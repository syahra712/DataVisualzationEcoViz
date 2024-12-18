from netCDF4 import Dataset
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px

# Paths to your NetCDF files
file_paths = {
    '2012': '/Users/admin/Downloads/Gridded Methane Data 2012/GEPA_Annual.nc',
    '2013': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2013.nc',
    '2014': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2014.nc',
    '2015': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2015.nc',
    '2016': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2016.nc',
    '2017': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2017.nc',
    '2018': '/Users/admin/Downloads/gridded GHGI/Gridded_GHGI_Methane_v2_2018.nc'
}

# Updated variables dictionary
variables = {
    'Mobile Combustion': 'emissions_1A_Combustion_Mobile',
    'Stationary Combustion': 'emissions_1A_Combustion_Stationary',
    'Natural Gas Production': 'emissions_1B2b_Natural_Gas_Production',
    'Enteric Fermentation': 'emissions_4A_Enteric_Fermentation',
    'Municipal Landfills': 'emissions_6A_Landfills_Municipal',
}

# Function to load data from a NetCDF file
def load_data(file_path, var_name):
    dataset = Dataset(file_path, mode='r')
    try:
        emissions_data = dataset.variables[var_name][:]
        latitudes = dataset.variables['lat'][:]
        longitudes = dataset.variables['lon'][:]
        latitudes_mesh, longitudes_mesh = np.meshgrid(latitudes, longitudes)
        df = pd.DataFrame({
            'Latitude': latitudes_mesh.flatten(),
            'Longitude': longitudes_mesh.flatten(),
            'Emissions': emissions_data.flatten()
        }).dropna()
    except KeyError:
        print(f"Variable '{var_name}' not found in the dataset.")
        df = pd.DataFrame()  # Return an empty DataFrame if the variable is not found
    dataset.close()
    return df

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div(
        html.H1('Methane Emissions Visualization', style={'textAlign': 'center'}),
        style={'padding': '20px', 'backgroundColor': '#f8f9fa'}
    ),
    html.Div([
        html.Div([
            html.Label('Select Year:', style={'fontSize': '18px'}),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': year, 'value': year} for year in file_paths.keys()],
                value='2012',  # Default year
                multi=True
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            html.Label('Select Emission Source:', style={'fontSize': '18px'}),
            dcc.Checklist(
                id='emission-checkboxes',
                options=[{'label': name, 'value': var} for name, var in variables.items()],
                value=['emissions_1A_Combustion_Mobile'],  # Default selection
                inline=True
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '20px'}),

    # Graph
    dcc.Graph(id='heatmap', style={'height': '70vh'}),  # Increased height for the map

    # Slider for the year
    dcc.Slider(
        id='year-slider',
        min=2012,
        max=2018,
        marks={i: f'{i}' for i in range(2012, 2019)},
        value=2012,
        step=1,
        tooltip={"placement": "bottom", "always_visible": True}
    )
], style={'margin': '0 auto', 'maxWidth': '1200px'})  # Centered layout

@app.callback(
    Output('heatmap', 'figure'),
    [Input('year-dropdown', 'value'), Input('emission-checkboxes', 'value'), Input('year-slider', 'value')]
)
def update_heatmap(selected_years, selected_vars, selected_year):
    df_list = []

    # Load data for selected years and variables
    for year in selected_years:
        if str(selected_year) in file_paths:
            for var_name in selected_vars:
                file_path = file_paths[str(selected_year)]
                df = load_data(file_path, var_name)
                if not df.empty:
                    df['Year'] = selected_year  # Add year information
                    df['Emission Source'] = var_name  # Add source information
                    df_list.append(df)

    if df_list:
        combined_df = pd.concat(df_list)
    else:
        combined_df = pd.DataFrame(columns=['Latitude', 'Longitude', 'Emissions'])

    # Create the heatmap
    fig = px.density_mapbox(
        combined_df,
        lat='Latitude',
        lon='Longitude',
        z='Emissions',
        radius=15,
        mapbox_style="open-street-map",
        title=f'Methane Emissions Visualization for {selected_year}',
        center={"lat": np.mean(combined_df['Latitude']), "lon": np.mean(combined_df['Longitude'])} if not combined_df.empty else {"lat": 0, "lon": 0},
        zoom=3 if not combined_df.empty else 0,
        opacity=0.6,
        color_continuous_scale=px.colors.sequential.Plasma
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
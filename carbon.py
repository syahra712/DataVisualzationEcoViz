# Let's modify the code according to the dataset structure

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

# Using the uploaded data
data = pd.read_csv("/Users/admin/Desktop/datasetsNASAPROGRAMS/annual-co2-emissions-per-country.csv")

# Sidebar for filtering
st.sidebar.header("Filter Options")

# Multi-country selection
countries = data['Entity'].unique()
selected_countries = st.sidebar.multiselect('Select Countries', countries, default=countries[:3])

# Year range selection
year_min = int(data['Year'].min())
year_max = int(data['Year'].max())
selected_years = st.sidebar.slider('Select Year Range', year_min, year_max, (year_min, year_max))

# Filter the data by selected countries and years
filtered_data = data[(data['Entity'].isin(selected_countries)) & 
                     (data['Year'] >= selected_years[0]) & (data['Year'] <= selected_years[1])]

# Display the title and dataset
st.title("🌍 CO2 Emissions Data Visualization and Analysis")
st.write(f"**CO2 Emissions for selected countries between {selected_years[0]} and {selected_years[1]}**")

# Non-Visualization Feature 1: Download filtered data as CSV
st.sidebar.markdown("### Download Data")
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(filtered_data)
st.sidebar.download_button(label="Download Filtered Data as CSV", 
                           data=csv, 
                           file_name='filtered_CO2_data.csv',
                           mime='text/csv')

# Non-Visualization Feature 2: Summary Statistics
st.subheader("📊 Summary Statistics")
st.write(f"**Total CO2 Emissions (selected period):** {filtered_data['Annual CO₂ emissions'].sum():,.2f} Metric Tons")
st.write(f"**Average CO2 Emissions (per year):** {filtered_data['Annual CO₂ emissions'].mean():,.2f} Metric Tons")
st.write(f"**Total Countries Selected:** {len(selected_countries)}")

# Non-Visualization Feature 3: Emissions percentage change
st.subheader("📈 Percentage Change in CO2 Emissions")
def calculate_percentage_change(df):
    df = df.sort_values('Year')
    df['Emissions Change (%)'] = df['Annual CO₂ emissions'].pct_change() * 100
    return df

# Display the percentage change for each country
for country in selected_countries:
    st.write(f"### {country}")
    country_data = filtered_data[filtered_data['Entity'] == country]
    country_data = calculate_percentage_change(country_data)
    st.dataframe(country_data[['Year', 'Annual CO₂ emissions', 'Emissions Change (%)']].dropna())

# Visualization Section
st.subheader("📊 CO2 Emissions Visualizations")

# Visualization selection
visualization_type = st.selectbox("Select Visualization Type", [
    "Bar Chart", "Stacked Bar Chart", "Pie Chart",
    "Box Plot", "Radar Chart", "Violin Plot"
])

# Bar Chart for CO2 Emissions
if visualization_type == "Bar Chart":
    fig, ax = plt.subplots()
    filtered_data.groupby('Entity')['Annual CO₂ emissions'].sum().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Total CO2 Emissions per Country')
    ax.set_ylabel('CO2 Emissions (Metric Tons)')
    ax.set_xlabel('Countries')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Stacked Bar Chart
elif visualization_type == "Stacked Bar Chart":
    fig, ax = plt.subplots()
    for country in selected_countries:
        country_data = filtered_data[filtered_data['Entity'] == country]
        ax.bar(country_data['Year'], country_data['Annual CO₂ emissions'], label=country, alpha=0.7)
    ax.set_title('CO2 Emissions (Stacked) per Country Over Time')
    ax.set_ylabel('CO2 Emissions (Metric Tons)')
    ax.set_xlabel('Year')
    ax.legend()
    st.pyplot(fig)

# Pie Chart of CO2 Emissions (Country-wise Contribution)
elif visualization_type == "Pie Chart":
    emissions_sum = filtered_data.groupby('Entity')['Annual CO₂ emissions'].sum()
    fig, ax = plt.subplots()
    ax.pie(emissions_sum, labels=emissions_sum.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
    ax.set_title('Country-wise Contribution to CO2 Emissions')
    st.pyplot(fig)

# Box Plot (Distribution of CO2 Emissions)
elif visualization_type == "Box Plot":
    fig, ax = plt.subplots()
    filtered_data.boxplot(column='Annual CO₂ emissions', by='Entity', ax=ax, patch_artist=True)
    ax.set_title('Distribution of CO2 Emissions per Country')
    ax.set_ylabel('CO2 Emissions (Metric Tons)')
    plt.suptitle('')
    st.pyplot(fig)

# Radar Chart (Country Comparisons)
elif visualization_type == "Radar Chart":
    fig = go.Figure()
    categories = filtered_data['Year'].unique()
    for country in selected_countries:
        country_data = filtered_data[filtered_data['Entity'] == country]
        fig.add_trace(go.Scatterpolar(
            r=country_data['Annual CO₂ emissions'],
            theta=country_data['Year'],
            fill='toself',
            name=country
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title="Country Comparisons on CO2 Emissions"
    )
    st.plotly_chart(fig)

# Violin Plot
elif visualization_type == "Violin Plot":
    fig = px.violin(filtered_data, y="Annual CO₂ emissions", x="Entity", box=True, points="all",
                    title="Violin Plot of CO2 Emissions Distribution per Country")
    st.plotly_chart(fig)

st.markdown("**Use the sidebar to filter the data and explore the visualizations!**")

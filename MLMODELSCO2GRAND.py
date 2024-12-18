import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
@st.cache
def load_and_preprocess_data():
    df = pd.read_csv('/Users/admin/Downloads/co2 Emissions.csv')
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[df["Fuel Type"] != "Natural Gas"]
    df.dropna(inplace=True)

    df_correlation = df[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']]
    z = np.abs(stats.zscore(df_correlation))
    df_new = df_correlation[(z < 1.9).all(axis=1)]
    df_new.reset_index(drop=True, inplace=True)

    X = df_new.drop(['CO2 Emissions(g/km)'], axis=1).astype(np.float32)
    y = df_new["CO2 Emissions(g/km)"].astype(np.float32)

    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    X["Engine Size(L)"] = X["Engine Size(L)"].round(2)
    X["Cylinders"] = X["Cylinders"].round(2)
    X["Fuel Consumption Comb (L/100 km)"] = X["Fuel Consumption Comb (L/100 km)"].round(2)

    return train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network model
def create_neural_network(input_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error', metrics=['mse'])
    return model

# Train and evaluate models
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    r2_train = model.score(X_train, y_train)
    model_r2_score = cross_val_score(model, X_train, y_train, cv=10, scoring="r2").mean()
    return train_rmse, test_rmse, r2_train, model_r2_score

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# App structure
st.title("CO2 Emissions Prediction Application")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Model Training", "Make Predictions"])

if page == "Model Training":
    st.header("Train and Evaluate Models")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
        "XGBoost": xgb.XGBRegressor(),
        "LightGBM": lgb.LGBMRegressor(),
        "CatBoost": CatBoostRegressor(verbose=0),
    }

    results = []
    for model_name, model in models.items():
        train_rmse, test_rmse, r2_train, model_r2_score = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
        results.append([model_name, train_rmse, test_rmse, r2_train, model_r2_score])

    # Neural Network
    nn_model = create_neural_network(X_train.shape[1])
    nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
    nn_train_rmse = np.sqrt(mean_squared_error(y_train, nn_model.predict(X_train)))
    nn_test_rmse = np.sqrt(mean_squared_error(y_test, nn_model.predict(X_test)))
    nn_r2_train = r2_score(y_train, nn_model.predict(X_train))
    nn_cross_val_r2 = np.mean(cross_val_score(RandomForestRegressor(), X_train, y_train, cv=10, scoring='r2'))
    results.append(["Neural Network", nn_train_rmse, nn_test_rmse, nn_r2_train, nn_cross_val_r2])

    results_df = pd.DataFrame(results, columns=["Model", "Training RMSE", "Testing RMSE", "R2 Train", "Cross-validated R2"])
    st.dataframe(results_df)

    # Plot Model Performance
    st.subheader("Model Performance Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=results_df.melt(id_vars=["Model"], value_vars=["Training RMSE", "Testing RMSE", "R2 Train", "Cross-validated R2"]),
                x="Model", y="value", hue="variable", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif page == "Make Predictions":
    st.header("Predict CO2 Emissions")

    engine_size = st.number_input("Engine Size (L):", min_value=0.0, step=0.1)
    cylinders = st.number_input("Cylinders:", min_value=1, step=1)
    fuel_consumption = st.number_input("Fuel Consumption (L/100 km):", min_value=0.0, step=0.1)

    if st.button("Predict"):
        new_data = pd.DataFrame({
            "Engine Size(L)": [engine_size],
            "Cylinders": [cylinders],
            "Fuel Consumption Comb (L/100 km)": [fuel_consumption]
        })

        new_data_normalized = (new_data - np.min(X_train)) / (np.max(X_train) - np.min(X_train))
        new_data_normalized["Engine Size(L)"] = new_data_normalized["Engine Size(L)"].round(2)
        new_data_normalized["Cylinders"] = new_data_normalized["Cylinders"].round(2)
        new_data_normalized["Fuel Consumption Comb (L/100 km)"] = new_data_normalized["Fuel Consumption Comb (L/100 km)"].round(2)

        predictions = {}
        for model_name, model in models.items():
            predictions[model_name] = model.predict(new_data_normalized)

        nn_prediction = nn_model.predict(new_data_normalized)
        predictions["Neural Network"] = nn_prediction.flatten()

        st.subheader("Predictions")
        for model_name, prediction in predictions.items():
            st.write(f"{model_name}: Predicted CO2 Emissions = {prediction[0]:.2f} g/km")

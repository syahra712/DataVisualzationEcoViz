import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

# Read and preprocess data
df = pd.read_csv('/Users/admin/Downloads/co2 Emissions.csv')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# Remove "Natural Gas" fuel type
df = df[df["Fuel Type"] != "Natural Gas"]

# Drop columns with NA values if any
df.dropna(inplace=True)

# Filter the dataset for relevant features
df_correlation = df[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']]
z = np.abs(stats.zscore(df_correlation))
df_new = df_correlation[(z < 1.9).all(axis=1)]
df_new.reset_index(drop=True, inplace=True)

# Split data into features (X) and target (y)
X = df_new.drop(['CO2 Emissions(g/km)'], axis=1).astype(np.float32)
y = df_new["CO2 Emissions(g/km)"].astype(np.float32)

# Normalize
X = (X - np.min(X)) / (np.max(X) - np.min(X))
X["Engine Size(L)"] = X["Engine Size(L)"].round(2)
X["Cylinders"] = X["Cylinders"].round(2)
X["Fuel Consumption Comb (L/100 km)"] = X["Fuel Consumption Comb (L/100 km)"].round(2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and evaluation
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    r2_train = model.score(X_train, y_train)
    model_r2_score = cross_val_score(model, X_train, y_train, cv=10, scoring="r2").mean()
    predictions = model.predict(X_test)  # Generate predictions
    return train_rmse, test_rmse, r2_train, model_r2_score, predictions

# Gradient Boosting Models
xgb_model = xgb.XGBRegressor()
lgb_model = lgb.LGBMRegressor()
catboost_model = CatBoostRegressor(verbose=0)

# Neural Network
def create_neural_network():
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error', metrics=['mse'])
    return model

# Stacking Models
base_models = [
    ('Random Forest', RandomForestRegressor()),
    ('Linear Regression', LinearRegression()),
    ('KNN', KNeighborsRegressor()),
    ('SVR', SVR())
]
stacking_model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

# Train and evaluate models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "XGBoost": xgb_model,
    "LightGBM": lgb_model,
    "CatBoost": catboost_model,
    "Stacking": stacking_model
}

# Initialize results
results = []
all_predictions = {}  # Dictionary to store predictions for each model

for model_name, model in models.items():
    train_rmse, test_rmse, r2_train, model_r2_score, predictions = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
    results.append([model_name, train_rmse, test_rmse, r2_train, model_r2_score])
    all_predictions[model_name] = predictions  # Store predictions

# Neural Network Evaluation
nn_model = create_neural_network()
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
nn_train_rmse = np.sqrt(mean_squared_error(y_train, nn_model.predict(X_train)))
nn_test_rmse = np.sqrt(mean_squared_error(y_test, nn_model.predict(X_test)))
nn_r2_train = r2_score(y_train, nn_model.predict(X_train))
nn_cross_val_r2 = np.mean(cross_val_score(RandomForestRegressor(), X_train, y_train, cv=10, scoring='r2'))
nn_predictions = nn_model.predict(X_test).flatten()  # Convert predictions to 1D array
results.append(["Neural Network", nn_train_rmse, nn_test_rmse, nn_r2_train, nn_cross_val_r2])
all_predictions["Neural Network"] = nn_predictions

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=["Model", "Training RMSE", "Testing RMSE", "R2 Train", "Cross-validated R2"])
print(results_df)

# Display predictions for all models
for model_name, predictions in all_predictions.items():
    print(f"\n{model_name} Predictions:")
    print(f"Predicted CO2 Emissions: {predictions[:5]}")
    print(f"Actual CO2 Emissions: {y_test[:5].values}")

# Plot Model Performance
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df.melt(id_vars=["Model"], value_vars=["Training RMSE", "Testing RMSE", "R2 Train", "Cross-validated R2"]),
            x="Model", y="value", hue="variable")
plt.title("Model Performance Comparison")
plt.xticks(rotation=45)
plt.show()

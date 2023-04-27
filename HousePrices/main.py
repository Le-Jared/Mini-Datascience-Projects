import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

# Load the datasets
train_data = pd.read_csv("/Users/jared/Documents/Code/Projects/Mini-Datascience-Projects/HousePrices/train.csv")
test_data = pd.read_csv("/Users/jared/Documents/Code/Projects/Mini-Datascience-Projects/HousePrices/test.csv")

# Combine the datasets for preprocessing and feature engineering
combined_data = pd.concat([train_data, test_data], sort=False)

# Handle missing values
for column in combined_data.columns:
    if combined_data[column].isnull().any():
        if combined_data[column].dtype == 'O':
            combined_data[column].fillna(combined_data[column].mode()[0], inplace=True)
        else:
            combined_data[column].fillna(combined_data[column].median(), inplace=True)

# Encode categorical features
combined_data = pd.get_dummies(combined_data)

# Split the combined_data back into train and test sets
train_data = combined_data[:len(train_data)]
test_data = combined_data[len(train_data):]

# Split the train_data into X (features) and y (target)
X = train_data.drop("SalePrice", axis=1)
y = np.log1p(train_data["SalePrice"])

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train the model
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_val, y_val)], verbose=False)

# Evaluate the model
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

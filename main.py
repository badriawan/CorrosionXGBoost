import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging


# Ganti dengan path file kamu
file_path = "/content/drive/MyDrive/S3 UTP/dataset/Soil_Pitting_Corrosion_Data_Final.xlsx"
df = pd.read_excel(file_path)

print(df.head())

X = df.drop(columns=['target'])  # ganti 'target' sesuai kolom kamu
y = df['target']

target_column = "target"

X = df.drop(columns=[target_column])
y = df[target_column]

X = pd.get_dummies(
    X,
    columns=["feature_11", "feature_12"],
    drop_first=True
)

print(X.dtypes)

feature_names = X.columns


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")


plt.figure(figsize=(8,5))
plt.barh(X.columns, model.feature_importances_)
plt.xlabel("Importance Score")
plt.title("Feature Importance - XGBoost")
plt.show()

# simpan feature names
feature_names = X.columns

# template inference
new_data = pd.DataFrame(
    np.zeros((1, len(feature_names))),
    columns=feature_names
)

new_data["feature_1"] = 7.2
new_data["feature_2"] = 45
new_data["feature_3"] = 30
new_data["feature_11_WTC"] = 1
new_data["feature_12_SC"] = 1

prediction = model.predict(new_data)
print("Prediksi corrosion:", prediction)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y,
    scoring='neg_root_mean_squared_error',
    cv=5
)

print("CV RMSE:", -scores.mean())


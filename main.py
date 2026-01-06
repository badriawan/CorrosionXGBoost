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
print(df.shape)


X = df.drop(columns=['target'])  # ganti 'target' sesuai kolom kamu
y = df['target']

target_column = "target"

X = df.drop(columns=[target_column])
y = df[target_column]

X = pd.get_dummies(
    X,
    columns=["ct", "Class"],
    drop_first=True
)

print(X.dtypes)

feature_names = X.columns


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=4282,
    max_depth=12,
    learning_rate=0.0216,
    subsample=0.6775,
    reg_lambda=0.8544,
    reg_alpha=0.2380,
    max_delta_step=10.0,
    colsample_bytree=0.9798,
    min_child_weight=3.0,
    gamma=0.3748,
    scale_pos_weight=0.0
    #objective='reg:squarederror',
    #random_state=42
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
plt.savefig("score.png", dpi=300, bbox_inches="tight")
plt.show()

# simpan feature names
feature_names = X.columns

# template inference
new_data = pd.DataFrame(
    np.zeros((1, len(feature_names))),
    columns=feature_names
)

new_data["t (years)"] = 20
new_data["pH"] = 4.5
new_data["pp (V)"] = -0.8
new_data["ct_WTC"] = 1
new_data["Class_SC"] = 1

prediction = model.predict(new_data)
print("Prediksi corrosion:", prediction)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y,
    scoring='neg_root_mean_squared_error',
    cv=5
)

print("CV RMSE:", -scores.mean())

import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

shap.summary_plot(shap_values, X_train, show=False)
plt.savefig("summary_plot.png", dpi=300, bbox_inches="tight")

shap.summary_plot(
    shap_values,
    X_train,
    plot_type="bar",
    show=False
)
plt.savefig("summary_2.png", dpi=300, bbox_inches="tight")


sample_idx = 0

shap.force_plot(
    explainer.expected_value,
    shap_values[sample_idx],
    X_train.iloc[sample_idx],
    matplotlib=True,
    show=False
)
plt.savefig("force.png", dpi=300, bbox_inches="tight")


shap.dependence_plot(
    "pH",  # ganti fitur utama
    shap_values,
    X_train,
    show=False

)
plt.savefig("dependence.png", dpi=300, bbox_inches="tight")


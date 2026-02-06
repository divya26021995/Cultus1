
Advanced Multivariate Time Series Forecasting
# LSTM + Optuna + ARIMA Baseline + SHAP Explainability

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import shap
import optuna
import warnings

# -------------------- Reproducibility ---------------------
np.random.seed(42)
tf.random.set_seed(42)
warnings.filterwarnings("ignore")

1. DATA GENERATION (Trend + Seasonality + AR + Noise)
def generate_data(n_steps=1500):
    t = np.arange(n_steps)

    trend = 0.004 * t
    seasonal_1 = np.sin(2 * np.pi * t / 24)
    seasonal_2 = np.sin(2 * np.pi * t / 365)
    noise = np.random.normal(0, 0.3, n_steps)
    f1 = trend + seasonal_1 + noise
    f2 = seasonal_2 + np.random.normal(0, 0.2, n_steps)
    f3 = np.roll(f1, 1) * 0.6
    f4 = np.random.normal(0, 1, n_steps)
    f5 = np.cos(2 * np.pi * t / 12)

    target = 0.5 * f1 + 0.3 * f2 + 0.2 * f3 + noise
    df = pd.DataFrame({
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "f4": f4,
        "f5": f5,
        "target": target
    })
    return df.dropna()
df = generate_data()

2. TRAIN / VALIDATION / TEST SPLIT
train_end = int(len(df) * 0.7)
val_end = int(len(df) * 0.85)

train_df = df[:train_end]
val_df = df[train_end:val_end]
test_df = df[val_end:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
val_scaled = scaler.transform(val_df)
test_scaled = scaler.transform(test_df)

3. SEQUENCE CREATION
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, :-1])
        y.append(data[i+seq_len, -1])
    return np.array(X), np.array(y)

4. OPTUNA HYPERPARAMETER OPTIMIZATION (VALIDATION RMSE)
def objective(trial):
    seq_len = trial.suggest_int("seq_len", 12, 48)
    units = trial.suggest_int("units", 32, 128)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    X_train, y_train = create_sequences(train_scaled, seq_len)
    X_val, y_val = create_sequences(val_scaled, seq_len)
    model = Sequential([
        LSTM(units, input_shape=(seq_len, X_train.shape[2])),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mse"
    )
    model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        verbose=0
    )
    val_preds = model.predict(X_val, verbose=0).flatten()
    return np.sqrt(mean_squared_error(y_val, val_preds))
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
best = study.best_params

5. FINAL LSTM MODEL
SEQ_LEN = best["seq_len"]
X_train, y_train = create_sequences(train_scaled, SEQ_LEN)
X_val, y_val = create_sequences(val_scaled, SEQ_LEN)
X_test, y_test = create_sequences(test_scaled, SEQ_LEN)
model = Sequential([
    LSTM(best["units"], return_sequences=True,
         input_shape=(SEQ_LEN, X_train.shape[2])),
    Dropout(0.2),
    LSTM(best["units"] // 2),
    Dense(1)
])
model.compile(
    optimizer=Adam(best["lr"]),
    loss="mse"
)
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    verbose=1
)
6. EVALUATION METRICS
def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape
lstm_preds = model.predict(X_test).flatten()
lstm_rmse, lstm_mae, lstm_mape = metrics(y_test, lstm_preds)

7. BASELINE MODEL (ARIMA)
arima = ARIMA(train_df["target"], order=(5, 1, 1))
arima_fit = arima.fit()
arima_forecast = arima_fit.forecast(steps=len(test_df))

arima_rmse, arima_mae, arima_mape = metrics(
    test_df["target"].values, arima_forecast
)

8. SHAP EXPLAINABILITY (TIME-AWARE)
background = X_train[:100]
explainer = shap.DeepExplainer(model, background)
shap_vals = explainer.shap_values(X_test[:20])[0]  # (samples, time, features)

# Aggregate SHAP values
mean_shap_time = np.mean(np.abs(shap_vals), axis=0)       # (time, features)
short_term = mean_shap_time[-3:].mean(axis=0)
long_term = mean_shap_time.mean(axis=0)
feature_names = ["f1", "f2", "f3", "f4", "f5"]
shap_summary = pd.DataFrame({
    "Feature": feature_names,
    "Short_Term_Importance": short_term,
    "Long_Term_Importance": long_term
}).sort_values("Long_Term_Importance", ascending=False)

9. RESULTS OUTPUT
print("\n===== BASELINE: ARIMA =====")
print(f"RMSE: {arima_rmse:.4f}")
print(f"MAE:  {arima_mae:.4f}")
print(f"MAPE: {arima_mape:.2f}%")

print("\n===== LSTM MODEL =====")
print(f"RMSE: {lstm_rmse:.4f}")
print(f"MAE:  {lstm_mae:.4f}")
print(f"MAPE: {lstm_mape:.2f}%")

print("\n===== SHAP FEATURE IMPORTANCE =====")
print(shap_summary)


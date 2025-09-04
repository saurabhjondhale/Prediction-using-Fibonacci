# fib_pipeline.py
# Requirements: numpy, pandas, scikit-learn, matplotlib
# pip install numpy pandas scikit-learn matplotlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------- Parameters ----------------
N = 40                 # number of Fibonacci terms to generate
window_size = 5        # how many previous values used to predict the next
test_fraction = 0.2    # fraction of samples kept as test
forecast_steps = 10    # recursive forecast horizon (how many steps ahead to predict)

# ---------------- 1) Generate Fibonacci sequence ----------------
fib = [0, 1]
for _ in range(2, N):
    fib.append(fib[-1] + fib[-2])
fib = np.array(fib, dtype=np.float64)

print("First 12 Fibonacci numbers:", fib[:12])

# ---------------- 2) Build supervised dataset ----------------
X, y = [], []
for i in range(window_size, len(fib)):
    X.append(fib[i-window_size:i])
    y.append(fib[i])

X = np.array(X)            # shape = (samples, window_size)
y = np.array(y).reshape(-1, 1)

print(f"Dataset size: X={X.shape}, y={y.shape}")

# ---------------- 3) Time-series train/test split (no shuffle) ----------------
split_idx = int((1 - test_fraction) * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# ---------------- 4) Scaling (fit on train only) ----------------
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_train_s = x_scaler.fit_transform(X_train)
X_test_s = x_scaler.transform(X_test)

y_train_s = y_scaler.fit_transform(y_train)
y_test_s = y_scaler.transform(y_test)

# ---------------- 5) Models ----------------
models = {
    "LinearRegression": LinearRegression(),
    "MLPRegressor": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)
}

trained = {}
results = []

for name, model in models.items():
    model.fit(X_train_s, y_train_s.ravel())
    trained[name] = model

    y_pred_s = model.predict(X_test_s).reshape(-1, 1)
    y_pred = y_scaler.inverse_transform(y_pred_s)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append((name, mae, mse, rmse, r2))

results_df = pd.DataFrame(results, columns=["model", "MAE", "MSE", "RMSE", "R2"]).set_index("model")
print("\nModel performance on test set:")
print(results_df)

# ---------------- 6) Plot actual vs predicted on test set ----------------
for name, model in trained.items():
    y_pred_s = model.predict(X_test_s).reshape(-1, 1)
    y_pred = y_scaler.inverse_transform(y_pred_s)
    plt.figure(figsize=(8,4))
    plt.title(f"Actual vs Predicted (Test) — {name}")
    plt.plot(range(len(y_test)), y_test.flatten(), label="Actual")
    plt.plot(range(len(y_pred)), y_pred.flatten(), label="Predicted", linestyle="--")
    plt.xlabel("Test sample index")
    plt.ylabel("Fibonacci value")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------- 7) Recursive forecasting (starting from the last observed window) ----------------
last_window = fib[-window_size:].reshape(1, -1)
print("\nLast observed window:", last_window.flatten())

forecasts = {}
for name, model in trained.items():
    window = last_window.copy()
    preds = []
    for step in range(forecast_steps):
        win_s = x_scaler.transform(window)
        pred_s = model.predict(win_s).reshape(-1, 1)
        pred = y_scaler.inverse_transform(pred_s).flatten()[0]
        preds.append(pred)
        # slide window: drop first value, append predicted
        window = np.roll(window, -1)
        window[0, -1] = pred
    forecasts[name] = np.array(preds)

# compute true next fibonacci numbers (for comparison)
true_ext = list(fib.copy())
for _ in range(forecast_steps):
    true_ext.append(true_ext[-1] + true_ext[-2])
true_next = np.array(true_ext[-forecast_steps:])

print("\nTrue next values (ground-truth):", true_next)
for name, pred in forecasts.items():
    print(f"\n{name} forecasted next {forecast_steps} values:\n", np.round(pred, 2))

# Plot recursive forecasts vs true
for name, pred in forecasts.items():
    plt.figure(figsize=(8,4))
    plt.title(f"Recursive forecast vs True — {name}")
    steps = np.arange(1, forecast_steps+1)
    plt.plot(steps, true_next, label="True Next", marker='o')
    plt.plot(steps, pred, label="Predicted (recursive)", marker='x', linestyle='--')
    plt.xlabel("Forecast step ahead")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

print("\nPipeline complete.")

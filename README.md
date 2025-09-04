# Prediction-using-Fibonacci
Prediction using Fibonacci


# fib_pipeline.py
# Requirements: numpy, pandas, scikit-learn, matplotlib
# pip install numpy pandas scikit-learn matplotlib


Baseline (window=2): LinearRegression will learn coefficients ≈ [1, 1] and give exact predictions (up to numeric precision). Good sanity check.

Window > 2: Models must infer the underlying recurrence from more inputs. LinearRegression may still find a linear relation; MLP can learn nonlinear combinations. Because Fibonacci grows exponentially, scaling is critical (we used MinMax).

Recursive forecasting (predict → append predicted → predict next) accumulates error. For deterministic Fibonacci, exactness is possible with correct features (last 2), but for ML models error will drift when forecasting many steps ahead.

Metrics: MAE, RMSE, R² — smaller MAE/RMSE and R² close to 1 indicate good fit. Beware that R² can be misleading when values grow rapidly; inspect plots too.

Try window_size=2 — confirm exactness.

Use log transform (log1p) to tame exponential growth for some models. Inverse via expm1.

Sequence models (LSTM/GRU) — for longer memory patterns; use Keras/TensorFlow. I can provide code if you want.

Walk-forward validation (time-series CV) — retrain as you move forward rather than train/test once.

Feature engineering for real data:

Use Fibonacci ratios (F(n-1)/F(n-2)), rolling means, diffs.

Use Fibonacci retracement levels as features for financial data.

Direct formula: For Fibonacci numbers you could use Binet’s formula to calculate nth term exactly (but that’s math, not ML).

Apply to real-world data: Extract turning points (peaks/valleys) from price/time-series, then use Fibonacci retracement levels as candidate support/resistance — feed those levels as engineered features to your ML model.


Optional: LSTM sketch (if you want diving into deep learning)

# LSTM sketch (requires tensorflow)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
# Prepare data shaped (samples, timesteps, features) => timesteps=window_size, features=1
X_train_l = X_train.reshape((X_train.shape[0], window_size, 1))
X_test_l = X_test.reshape((X_test.shape[0], window_size, 1))
# Build
model = Sequential()
model.add(LSTM(64, input_shape=(window_size, 1), return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train_l, y_train, epochs=200, batch_size=16, validation_split=0.1)
# Predict then inverse-transform if you scaled y



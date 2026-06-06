import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

print("--- TIME SERIES PREDICTION DEMO STARTED ---")

# 1. Generate synthetic sine wave data
def generate_sequence(length=50):
    x = np.linspace(0, 2*np.pi, length)
    return np.sin(x) + np.random.normal(scale=0.1, size=length)

sequences = [generate_sequence() for _ in range(1000)]
X, y = [], []
for seq in sequences:
    X.append(seq[:-1])   # input sequence
    y.append(seq[-1])    # target (next value)

X = np.array(X)
y = np.array(y)

# Reshape for LSTM [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# 2. Build LSTM model
model = models.Sequential([
    layers.LSTM(50, activation='tanh', input_shape=(X.shape[1], 1)),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, verbose=0)

# 3. Test on 3 samples
for i in range(3):
    seq = generate_sequence()
    X_test = seq[:-1].reshape((1, len(seq)-1, 1))
    y_actual = seq[-1]
    y_pred = model.predict(X_test)[0][0]

    plt.subplot(3,1,i+1)
    plt.plot(seq[:-1], label="Input Sequence")
    plt.scatter(len(seq)-1, y_actual, color="red", label="Actual Target")
    plt.scatter(len(seq)-1, y_pred, color="green", marker="x", label="Predicted Target")
    plt.title(f"Sample {i+1} Prediction: Actual={y_actual:.2f}, Predicted={y_pred:.2f}")
    plt.legend()

plt.tight_layout()
plt.show()
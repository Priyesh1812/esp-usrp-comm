import torch
import torch.nn as nn
import serial
import time
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import re
from collections import deque

# Global buffers for temperature history
temps = deque(maxlen=200)
sample_count = 0

# -----------------------------------
# Parse "Temperature: 22.5 °C" → 22.5
# -----------------------------------
def parse_temperature(raw):
    match = re.search(r"([-+]?\d*\.\d+|\d+)", raw)
    if match:
        return float(match.group(0))
    return None

# -----------------------------------
# Normalization (IMPORTANT)
# Temperature usually ~15–35°C
# Normalize to ~-1 .. +1 range
# -----------------------------------
def normalize(x):
    return (x - 25.0) / 10.0

def denormalize(x):
    return x * 10.0 + 25.0

# -----------------------------------
# LSTM Model
# -----------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# -----------------------------------
# Parameters
# -----------------------------------
PORT = "/dev/ttyUSB0"
BAUD = 115200

WINDOW = 5              # Short window works best for slow-changing temperature
PREDICT_FUTURE = 10     # Predict next 10 seconds
WARMUP = 40             # Wait 40 samples before training

temps = deque(maxlen=200)  # store last few minutes

model = LSTMForecaster()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# -----------------------------------
# Serial init
# -----------------------------------
def init_serial():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print(f"[OK] Connected to {PORT}")
    return ser

# -----------------------------------
# Train one small step
# -----------------------------------
def train_one_step():
    if len(temps) < WINDOW + 1:
        return None

    x_vals = list(temps)[-WINDOW-1:-1]
    y_val = list(temps)[-1]

    x = torch.tensor(x_vals, dtype=torch.float32).view(1, WINDOW, 1)
    y = torch.tensor([y_val], dtype=torch.float32)

    output = model(x)
    loss = criterion(output.squeeze(), y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# -----------------------------------
# Predict future values
# -----------------------------------
def predict_future():
    if len(temps) < WINDOW:
        return []

    data = list(temps)[-WINDOW:]
    seq = torch.tensor(data, dtype=torch.float32).view(1, WINDOW, 1)

    preds = []
    model.eval()

    with torch.no_grad():
        current = seq.clone()
        for _ in range(PREDICT_FUTURE):
            next_val = model(current).item()
            preds.append(next_val)
            current = torch.cat([current[:, 1:, :], torch.tensor([[[next_val]]])], dim=1)

    # Convert normalized preds → real temperatures
    preds = [denormalize(p) for p in preds]
    return preds

# -----------------------------------
# Main Loop
# -----------------------------------
def main():
    ser = init_serial()

    plt.ion()
    fig, ax = plt.subplots()
    line_real, = ax.plot([], [], label="Temperature")
    line_pred, = ax.plot([], [], label="Prediction")
    ax.legend()
    ax.grid(True)

    sample_count = 0

    while True:
        try:
            if ser.in_waiting > 0:
                raw = ser.readline().decode('utf-8', errors='ignore').strip()
                print("Raw:", raw)

                temp = parse_temperature(raw)
                if temp is None:
                    print("Could not parse temperature!")
                    continue

                print("Parsed:", temp)
                sample_count += 1

                # Normalized append
                temps.append(normalize(temp))

                # Wait for warmup
                if sample_count < WARMUP:
                    print(f"[WARMUP] collecting initial data… {sample_count}/{WARMUP}")
                    continue

                # Train multiple times to stabilize
                for _ in range(15):
                    loss = train_one_step()

                preds = predict_future()

                # Update plot
                real_vals = [denormalize(t) for t in temps]
                line_real.set_data(range(len(real_vals)), real_vals)

                if preds:
                    start = len(real_vals)
                    line_pred.set_data(range(start, start + len(preds)), preds)

                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.001)

                print(f"Temp: {temp:.2f}°C   Loss: {loss:.4f}   Next: {preds[:1] if preds else None}")

        except KeyboardInterrupt:
            print("\nExiting…")
            break

    ser.close()

if __name__ == "__main__":
    main()

import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import re
from scipy.signal import savgol_filter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from numpy.fft import rfft, rfftfreq

# -----------------------------
# CONFIG
# -----------------------------
PORT = "/dev/ttyUSB0"
BAUD = 115200

WINDOW = 200               # number of recent points to visualize
PREDICT_STEPS = 10         # future predictions (for smoothing models)

ENABLE_KALMAN = True
ENABLE_EXP_SMOOTHING = True
ENABLE_HOLT_WINTERS = True
ENABLE_SAVGOL = True
ENABLE_ROLLING_Z = True
ENABLE_AR = False          # requires statsmodels AR; optional

# -----------------------------
# Utility: Parse "Temperature: 23.93 °C"
# -----------------------------
def parse_temp(raw):
    match = re.search(r"([-+]?\d*\.\d+|\d+)", raw)
    return float(match.group(0)) if match else None

# -----------------------------
# 1) Kalman Filter
# -----------------------------
class KalmanFilter:
    def __init__(self, q=0.01, r=0.5):
        self.q = q      # process noise
        self.r = r      # measurement noise
        self.p = 1.0
        self.x = None

    def update(self, measurement):
        if self.x is None:
            self.x = measurement

        # prediction
        self.p += self.q

        # kalman gain
        k = self.p / (self.p + self.r)

        # update estimate
        self.x += k * (measurement - self.x)

        # update error covariance
        self.p *= (1 - k)

        return self.x

kf = KalmanFilter()

# -----------------------------
# Data Buffers
# -----------------------------        
 
  
   
    
     
      
       
        
         

          
           
            
             
              
               
                
                 
                  
                   
                    
                     
                      
                       
                        
                         
                          
                           
                            
                             
                              
                               
                                
                                     


                                      
                                       
                                        
                                         
                                          
                                           
                                            
                                             
                                              
                                               
                                                
                                                 
temps_raw = deque(maxlen=WINDOW)
temps_kf  = deque(maxlen=WINDOW)
temps_es  = deque(maxlen=WINDOW)
temps_hw  = deque(maxlen=WINDOW)
temps_sg  = deque(maxlen=WINDOW)
anomalies = deque(maxlen=WINDOW)

# -----------------------------
# Connect Serial
# -----------------------------
def open_serial():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print("[OK] Connected to ESP")
    return ser

ser = open_serial()

# -----------------------------
# Live Plot Setup
# -----------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(10,5))

line_raw,  = ax.plot([], [], label="Raw Data")
line_kf,   = ax.plot([], [], label="Kalman")
line_es,   = ax.plot([], [], label="Exp Smoothing")
line_hw,   = ax.plot([], [], label="Holt-Winters")
line_sg,   = ax.plot([], [], label="Savitzky-Golay")

scatter_anom = ax.scatter([], [], color='red', label="Anomalies")

ax.legend()
ax.grid(True)

# -----------------------------
# Rolling Anomaly Detector (Z-score)
# -----------------------------
def rolling_z_score(data, threshold=3):
    if len(data) < 20:
        return False

    arr = np.array(data)
    mean = arr.mean()
    std = arr.std() if arr.std() != 0 else 1e-6
    z = abs((arr[-1] - mean) / std)
    return z > threshold

# -----------------------------
# Prediction Helpers
# -----------------------------
def exp_smoothing_predict(data, alpha=0.3):
    if len(data) < 2:
        return data[-1]
    return alpha * data[-1] + (1-alpha) * data[-2]

def holt_winters_predict(data):
    if len(data) < 20:
        return data[-1]
    model = ExponentialSmoothing(data, trend=None, seasonal=None)
    fit = model.fit()
    return fit.forecast(PREDICT_STEPS)

# -----------------------------
# MAIN LOOP (NOTEBOOK STYLE)
# -----------------------------
while True:
    try:
        if ser.in_waiting > 0:
            raw = ser.readline().decode(errors="ignore").strip()
            temp = parse_temp(raw)
            if temp is None:
                continue

            temps_raw.append(temp)

            # KALMAN
            if ENABLE_KALMAN:
                kf_val = kf.update(temp)
                temps_kf.append(kf_val)

            # EXPONENTIAL SMOOTHING
            if ENABLE_EXP_SMOOTHING:
                es_val = exp_smoothing_predict(list(temps_raw))
                temps_es.append(es_val)

            # HOLT-WINTERS
            if ENABLE_HOLT_WINTERS:
                try:
                    hw_pred = holt_winters_predict(list(temps_raw))
                    temps_hw.append(hw_pred[0])
                except:
                    temps_hw.append(temp)

            # SAVITZKY-GOLAY
            if ENABLE_SAVGOL:
                if len(temps_raw) > 20:
                    sg = savgol_filter(list(temps_raw), window_length=11, polyorder=2)
                    temps_sg.clear()
                    temps_sg.extend(list(sg))
                else:
                    temps_sg.append(temp)

            # ANOMALY DETECTION
            if ENABLE_ROLLING_Z:
                is_anom = rolling_z_score(list(temps_raw))
                anomalies.append(temps_raw[-1] if is_anom else None)

            # -----------------------------
            # Update plot
            # -----------------------------
            xs = np.arange(len(temps_raw))

            line_raw.set_data(xs, temps_raw)
            line_kf.set_data(xs, temps_kf)
            line_es.set_data(xs, temps_es)
            line_hw.set_data(xs, temps_hw)
            line_sg.set_data(xs, temps_sg)

            # anomalies
            anom_x = [i for i,v in enumerate(anomalies) if v is not None]
            anom_y = [v for v in anomalies if v is not None]
            scatter_anom.set_offsets(np.c_[anom_x, anom_y])

            ax.relim()
            ax.autoscale_view()
            plt.title(f"Live Temperature Processing Notebook | Last: {temp:.2f}°C")
            plt.draw()
            plt.pause(0.001)

            print(f"Temp: {temp:.2f}°C")

    except KeyboardInterrupt:
        print("Exiting...")
        break

ser.close()

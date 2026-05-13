import serial
import time

# Replace with the correct serial port of your Jetson (e.g., /dev/ttyUSB0)
serial_port = '/dev/ttyUSB0'  
baud_rate = 115200  # Make sure this matches the baud rate of the ESP device

# Open the serial port
ser = serial.Serial(serial_port, baud_rate, timeout=1)

print(f"Reading from serial port {serial_port} at {baud_rate} baud...")

# Loop to continuously read data
try:
    while True:
        if ser.in_waiting > 0:  # Check if there is data waiting
            data = ser.readline()  # Read one line of data
            print(f"Received: {data.decode('utf-8').strip()}")  # Decode and strip to get clean output

        time.sleep(0.1)  # Small delay to avoid excessive CPU usage

except KeyboardInterrupt:
    print("\nExiting program...")
    
finally:
    ser.close()  # Make sure to close the serial port when done


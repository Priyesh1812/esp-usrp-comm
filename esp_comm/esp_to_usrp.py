import uhd
import numpy as np
import serial
import time
import struct
from lstm import parse_temperature, normalize, denormalize, temps, predict_future

# ============================================
# CONFIGURATION
# ============================================
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200

TX_FREQ = 915e6
TX_RATE = 1e6
TX_GAIN = 80
SAMPLES_PER_SYMBOL = 8

# ============================================
# MODULATION FUNCTIONS
# ============================================

def create_preamble():
    """Create sync preamble for packet detection"""
    barker = np.array([1, 1, 1, -1, -1, 1, -1, 1])
    return np.tile(barker, 8)

def bytes_to_iq_samples(data_bytes):
    """Convert data bytes to IQ samples for transmission"""
    preamble = b'\xAA\xAA\xAA\xAA'
    length = len(data_bytes).to_bytes(2, 'big')
    packet = preamble + length + data_bytes

    bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
    symbols = 2 * bits.astype(float) - 1

    sync_preamble = create_preamble()
    symbols = np.concatenate([sync_preamble, symbols])

    samples = np.repeat(symbols, SAMPLES_PER_SYMBOL)
    return samples.astype(np.complex64)

# ============================================
# MAIN PROGRAM
# ============================================

def main():
    print(f"Opening serial port {SERIAL_PORT} at {BAUD_RATE} baud...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Serial port opened successfully")

    # Setup USRP
    print("\nInitializing USRP...")
    usrp = uhd.usrp.MultiUSRP()
    usrp.set_tx_rate(TX_RATE)
    usrp.set_tx_freq(uhd.types.TuneRequest(TX_FREQ), 0)
    usrp.set_tx_gain(TX_GAIN, 0)

    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    streamer = usrp.get_tx_stream(stream_args)

    print(f"USRP initialized:")
    print(f"  Frequency: {usrp.get_tx_freq(0)/1e6} MHz")
    print(f"  Sample Rate: {usrp.get_tx_rate(0)/1e6} MHz")
    print(f"  Gain: {usrp.get_tx_gain(0)} dB")

    print("\nReading from ESP and continuously transmitting...\n")

    packet_count = 0

    # DEFAULT: transmit silence until first serial message
    last_samples = np.zeros(1024, dtype=np.complex64)

    try:
        while True:

            # -----------------------------
            # READ NEW SERIAL DATA WHEN AVAILABLE
            # -----------------------------
            if ser.in_waiting > 0:
                data = ser.readline()

                try:
                    decoded = data.decode('utf-8').strip()
                    print(f"ESP Data: {decoded}")
                except:
                    print(f"ESP Data: {len(data)} bytes (binary)")

                # Convert to IQ and store as last_samples
                last_samples = bytes_to_iq_samples(data)
                packet_count += 1

                print(f"Updated TX buffer from serial ({len(data)} bytes → "
                      f"{len(last_samples)} samples)")

            # -----------------------------
            # ALWAYS TRANSMIT last_samples
            # -----------------------------
            md = uhd.types.TXMetadata()
            md.has_time_spec = False
            streamer.send(last_samples, md)

            time.sleep(10)  # Keep TX pipe flowing

    except KeyboardInterrupt:
        print("\nStopping transmission...")

    finally:
        print("Sending end-of-burst...")
        md = uhd.types.TXMetadata()
        md.end_of_burst = True
        streamer.send(np.zeros(0, dtype=np.complex64), md)

        ser.close()
        print("Cleanup complete. Exited successfully.")

if __name__ == "__main__":
    main()

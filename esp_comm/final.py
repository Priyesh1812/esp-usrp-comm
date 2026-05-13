import os
os.environ["UHD_LOG_LEVEL"] = "off"
import uhd
import numpy as np
import serial
import time
import csv
import re
from lstm import (
    parse_temperature,
    normalize,
    temps,
    predict_future,
    train_one_step,
    WARMUP
)

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

TX_FREQ = 915e6
TX_RATE = 1e6
TX_GAIN = 80
SAMPLES_PER_SYMBOL = 8


def create_preamble():
    barker = np.array([1, 1, 1, -1, -1, 1, -1, 1])
    return np.tile(barker, 8)


def bytes_to_iq_samples(data_bytes):
    preamble = b'\xAA\xAA\xAA\xAA'
    length = len(data_bytes).to_bytes(2, "big")
    packet = preamble + length + data_bytes

    bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
    symbols = 2 * bits.astype(float) - 1

    sync_preamble = create_preamble()
    symbols = np.concatenate([sync_preamble, symbols])

    samples = np.repeat(symbols, SAMPLES_PER_SYMBOL)
    return samples.astype(np.complex64)


def main():
    print(f"Opening serial port {SERIAL_PORT} at {BAUD_RATE} baud...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    # -------- CSV LOGGER --------
    csv_file = open("sensor.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "time_ms","AQI_like","vNode_V",
        "est_AO_V","DO","Temp_C","RH_pct"
    ])
    # ----------------------------

    print("Initializing USRP...")
    usrp = uhd.usrp.MultiUSRP()
    usrp.set_tx_rate(TX_RATE)
    usrp.set_tx_freq(uhd.types.TuneRequest(TX_FREQ), 0)
    usrp.set_tx_gain(TX_GAIN, 0)

    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    streamer = usrp.get_tx_stream(stream_args)

    last_samples = np.zeros(1024, dtype=np.complex64)
    sample_count = 0

    try:
        while True:

            if ser.in_waiting > 0:
                raw = ser.readline()

                try:
                    decoded = raw.decode("utf-8").strip()
                except:
                    continue

                print(decoded)

                # --------- ROBUST CSV PARSER ---------
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", decoded)

                if len(nums) >= 7:
                    row = [
                        nums[0],  # time_ms
                        nums[1],  # AQI_like
                        nums[2],  # vNode
                        nums[3],  # est_AO
                        nums[4],  # DO
                        nums[5],  # Temp
                        nums[6],  # RH
                    ]
                    csv_writer.writerow(row)
                # -------------------------------------

                # -------- LSTM PART (unchanged) -------
                real_temp = parse_temperature(decoded)
                if real_temp is None:
                    continue

                temps.append(normalize(real_temp))
                sample_count += 1

                if sample_count < WARMUP:
                    continue

                for _ in range(15):
                    train_one_step()

                preds = predict_future()
                predicted = preds[0] if preds else -999.0

                tx_string = f"REAL:{real_temp:.3f},PRED:{predicted:.3f}"
                data_bytes = tx_string.encode("utf-8")
                last_samples = bytes_to_iq_samples(data_bytes)
                # --------------------------------------

            md = uhd.types.TXMetadata()
            md.has_time_spec = False
            streamer.send(np.tile(last_samples, 10), md)
            time.sleep(0.0005)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        csv_file.close()
        ser.close()

        md = uhd.types.TXMetadata()
        md.end_of_burst = True
        streamer.send(np.zeros(0, dtype=np.complex64), md)
        print("Cleanup complete.")


if __name__ == "__main__":
    main()

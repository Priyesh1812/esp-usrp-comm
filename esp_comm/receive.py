import uhd
import numpy as np
import threading

usrp = uhd.usrp.MultiUSRP()

# Configure both TX and RX
usrp.set_tx_rate(1e6)
usrp.set_rx_rate(1e6)
usrp.set_tx_freq(uhd.types.TuneRequest(915e6), 0)
usrp.set_rx_freq(uhd.types.TuneRequest(915e6), 0)
usrp.set_tx_gain(10, 0)  # LOW gain to avoid saturation!
usrp.set_rx_gain(30, 0)

# Create streams
tx_stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
tx_streamer = usrp.get_tx_stream(tx_stream_args)

rx_stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
rx_streamer = usrp.get_rx_stream(rx_stream_args)

running = True

def transmit():
    md = uhd.types.TXMetadata()
    sample_count = 0
    print("TX thread started")
    
    while running:
        # Generate tone
        chunk = 0.3 * np.exp(2j * np.pi * 0.05 * 
                             (np.arange(10000) + sample_count))
        tx_streamer.send(chunk.astype(np.complex64), md)
        sample_count += 10000
    
    # End burst
    md.end_of_burst = True
    tx_streamer.send(np.zeros(0, dtype=np.complex64), md)
    print("TX thread stopped")

def receive():
    # Start streaming
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    rx_streamer.issue_stream_cmd(stream_cmd)
    
    md = uhd.types.RXMetadata()
    print("RX thread started")
    
    while running:
        buffer = np.zeros(10000, dtype=np.complex64)
        rx_streamer.recv(buffer, md)
        power = np.mean(np.abs(buffer)**2)
        print(f"RX Power: {power:.6f}", end='\r')
    
    # Stop streaming
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    rx_streamer.issue_stream_cmd(stream_cmd)
    print("\nRX thread stopped")

# Start threads
tx_thread = threading.Thread(target=transmit)
rx_thread = threading.Thread(target=receive)

tx_thread.start()
rx_thread.start()

try:
    print("Running full-duplex. Press Ctrl+C to stop...")
    while True:
        pass
except KeyboardInterrupt:
    print("\nStopping...")
    running = False
    tx_thread.join()
    rx_thread.join()
    print("Done!")
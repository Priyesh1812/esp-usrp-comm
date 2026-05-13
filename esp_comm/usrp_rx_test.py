
import uhd
import numpy as np
import matplotlib.pyplot as plt

# Setup USRP
usrp = uhd.usrp.MultiUSRP()
usrp.set_rx_rate(2e6)  # 2 MHz sample rate
usrp.set_rx_freq(uhd.types.TuneRequest(98.5e6), 0)  # Your FM station
usrp.set_rx_gain(70, 0)

# Create stream
stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
streamer = usrp.get_rx_stream(stream_args)

# Start streaming
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
stream_cmd.stream_now = True
streamer.issue_stream_cmd(stream_cmd)

# Capture samples
num_samples = 200000
buffer = np.zeros(num_samples, dtype=np.complex64)
md = uhd.types.RXMetadata()
num_samps = streamer.recv(buffer, md)

print(f"Received {num_samps} samples")

# Stop streaming
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
streamer.issue_stream_cmd(stream_cmd)

# Compute FFT
fft = np.fft.fftshift(np.fft.fft(buffer))
freqs = np.fft.fftshift(np.fft.fftfreq(len(buffer), 1/2e6))  # In Hz
power_db = 20 * np.log10(np.abs(fft) + 1e-10)

# Plot
plt.figure(figsize=(14, 8))

# Plot 1: Power Spectrum
plt.subplot(2, 1, 1)
plt.plot(freqs/1e3, power_db)  # Convert to kHz
plt.xlabel('Frequency Offset (kHz)')
plt.ylabel('Power (dB)')
plt.title(f'FM Radio Power Spectrum - Center: 98.5 MHz')
plt.grid(True)
plt.xlim([freqs[0]/1e3, freqs[-1]/1e3])

# Plot 2: Time domain (first 1000 samples)
plt.subplot(2, 1, 2)
time = np.arange(1000) / 2e6 * 1e6  # Convert to microseconds
plt.plot(time, np.real(buffer[:1000]), label='I (Real)', alpha=0.7)
plt.plot(time, np.imag(buffer[:1000]), label='Q (Imag)', alpha=0.7)
plt.xlabel('Time (μs)')
plt.ylabel('Amplitude')
plt.title('Time Domain Signal (First 1000 samples)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('fm_spectrum.png', dpi=150)
print("Plot saved as fm_spectrum.png")
plt.show()
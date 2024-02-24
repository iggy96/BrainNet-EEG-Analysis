import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class AudioLatencyMeasurement:
    def __init__(self, fs=44100, buffer_size=4410, f0=100, f1=1000, chirp_duration=1.0):
        self.fs = fs
        self.buffer_size = buffer_size
        self.f0 = f0
        self.f1 = f1
        self.chirp_duration = chirp_duration
        
        # Generate the chirp
        self.chirp = self.generate_chirp(self.fs, self.chirp_duration, self.f0, self.f1)

    def generate_chirp(self, fs, duration, f0, f1):
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        chirp_signal = np.sin(2 * np.pi * f0 * t + (f1 - f0) * t**2 / (2 * duration))
        return chirp_signal

    def detect_chirp_start(self, chirp, signal):
        correlation = np.correlate(signal, chirp, 'valid')
        return np.argmax(correlation)

    def measure_latency(self, plot_flag=False):
        with sd.OutputStream(samplerate=self.fs) as player, sd.InputStream(samplerate=self.fs) as microphone:
            player.write(np.column_stack([self.chirp.astype(np.float32), self.chirp.astype(np.float32)]))  # Convert chirp to float32
            recorded_signal, _ = microphone.read(len(self.chirp) + self.buffer_size)
            
        played_chirp_start = self.detect_chirp_start(self.chirp, self.chirp)  # should be 0
        recorded_chirp_start = self.detect_chirp_start(self.chirp, recorded_signal[:, 0])  # taking one channel

        latency_samples = recorded_chirp_start - played_chirp_start
        latency_time = latency_samples / self.fs

        if plot_flag:
            self.plot_signals(recorded_signal)
        
        return latency_time


    def plot_signals(self, recorded_signal):
        t_chirp = np.arange(len(self.chirp)) / self.fs
        t_recorded = np.arange(len(recorded_signal)) / self.fs

        # Ensure that the chirp and recorded signals have the same length
        min_len = min(len(self.chirp), len(recorded_signal))
        chirp_signal = self.chirp[:min_len]
        recorded_signal = recorded_signal[:min_len]

        # Plotting chirp and recorded signals
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t_chirp, chirp_signal, label='Chirp Signal')
        plt.plot(t_recorded[:min_len], recorded_signal[:, 0], label='Recorded Signal')
        plt.title('Chirp and Recorded Signals')
        plt.xlabel('Time (in sec)')
        plt.ylabel('Amplitude')
        plt.legend()

        # Plotting the cross-correlation
        correlation = np.correlate(recorded_signal[:, 0], chirp_signal, 'full')
        lag = np.arange(-len(chirp_signal) + 1, len(chirp_signal)) / self.fs
        plt.subplot(2, 1, 2)
        plt.plot(lag, correlation)
        plt.title('Cross-Correlation between Chirp and Recorded Signal')
        plt.xlabel('Lag (in sec)')
        plt.ylabel('Cross-Correlation')
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":
    tool = AudioLatencyMeasurement()
    latency = tool.measure_latency(plot_flag=True)
    logger.info(f"Measured latency: {latency*1000:.2f} milliseconds")

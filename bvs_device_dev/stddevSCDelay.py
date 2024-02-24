import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf  # <-- Import soundfile
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class AudioLatencyMeasurement:

    def __init__(self, file_path='dev.wav', fs=16000, buffer_size=256):
        self.fs = fs
        self.buffer_size = buffer_size

        # Load audio data
        self.audio_out = self.load_audio_file(file_path)
        
        # Verify the length of audio_out is a multiple of buffer_size
        if len(self.audio_out) % buffer_size != 0:
            raise ValueError("audio_out length should be a multiple of buffer_size.")

        self.n_frames = len(self.audio_out) // buffer_size
        self.buffer = np.zeros((self.n_frames * buffer_size, 2))

    def load_audio_file(self, file_path):
        # Read the file using soundfile
        audio_data, _ = sf.read(file_path, dtype=np.float32, always_2d=True)
        
        # Calculate the remaining number of samples to reach a multiple of buffer_size
        remainder = len(audio_data) % self.buffer_size
        
        if remainder != 0:
            # Calculate the number of samples to pad
            pad_length = self.buffer_size - remainder
            
            # Pad the audio data with zeros
            audio_data = np.vstack((audio_data, np.zeros((pad_length, audio_data.shape[1]), dtype=np.float32)))
        
        return audio_data


    def stream_audio(self):
        with sd.OutputStream(samplerate=self.fs) as player, sd.InputStream(samplerate=self.fs) as microphone:
            for ind in range(self.n_frames):
                start_idx = ind * self.buffer_size
                end_idx = (ind + 1) * self.buffer_size

                try:
                    player.write(self.audio_out[start_idx:end_idx, :])
                    audio_in, _ = microphone.read(self.buffer_size)
                except Exception as e:
                    logger.error(f"Error streaming audio: {e}")
                    return

                if ind > 9:
                    sliced_audio_out = self.audio_out[start_idx:end_idx, 0]
                    sliced_audio_in = audio_in[:, 0][:len(sliced_audio_out)]
                    self.buffer[start_idx:end_idx, :] = np.column_stack((sliced_audio_out, sliced_audio_in))

    def compute_latency(self):
        rxy = np.abs(np.correlate(self.buffer[:, 0], self.buffer[:, 1], "full"))
        latency = (np.argmax(rxy) - len(self.buffer) + 1) / self.fs
        return latency, rxy

    def measure_latency(self, plot_flag=False):
        logger.info("Streaming audio...")
        self.stream_audio()
        latency, rxy = self.compute_latency()  # modified to get rxy
        
        if plot_flag:
            self.plot_signals(rxy)  # pass rxy as an argument to plot

        return latency

    def plot_signals(self, rxy):
        t = np.arange(self.buffer.shape[0]) / self.fs
        
        # Plotting the audio signals
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t, self.buffer)
        plt.title('Audio signals: Before audio player and after audio recorder')
        plt.legend(['Signal from audio file', 'Signal recorded (added latency of audio input and output)'])
        plt.xlabel('Time (in sec)')
        plt.ylabel('Audio signal')
        
        # Plotting the cross-correlation
        lag = np.arange(-len(self.buffer) + 1, len(self.buffer)) / self.fs
        plt.subplot(2, 1, 2)
        plt.plot(lag, rxy)
        plt.title('Cross-Correlation between played and received signals')
        plt.xlabel('Lag (in sec)')
        plt.ylabel('Cross-Correlation')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    tool = AudioLatencyMeasurement()
    latency = tool.measure_latency(plot_flag=True)
    logger.info(f"Measured latency: {latency*1000} milliseconds")









#%%
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

def generate_chirp(fs, duration, f0, f1):
    """Generate a linear chirp signal.
    
    Parameters:
    - fs: Sample rate
    - duration: Duration of the chirp in seconds
    - f0: Start frequency of the chirp
    - f1: End frequency of the chirp
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    chirp_signal = np.sin(2 * np.pi * f0 * t + (f1 - f0) * t**2 / (2 * duration))
    return chirp_signal

def detect_chirp_start(chirp, signal):
    """Detect the start of the chirp in the signal."""
    correlation = np.correlate(signal, chirp, 'valid')
    return np.argmax(correlation)

def measure_latency(fs=44100, duration=1.0, f0=100, f1=1000, buffer_size=4410):
    chirp = generate_chirp(fs, duration, f0, f1).astype(np.float32)

    
    # Play and record the chirp
    with sd.OutputStream(samplerate=fs) as player, sd.InputStream(samplerate=fs) as microphone:
        player.write(np.column_stack([chirp, chirp]))  # Playing in stereo
        recorded_signal, _ = microphone.read(len(chirp) + buffer_size)  # Add buffer size to ensure complete capture
        
    # Detect chirp in played and recorded signals
    played_chirp_start = detect_chirp_start(chirp, chirp)  # should be 0
    recorded_chirp_start = detect_chirp_start(chirp, recorded_signal[:, 0])  # taking one channel

    latency_samples = recorded_chirp_start - played_chirp_start
    latency_time = latency_samples / fs
    
    return latency_time

if __name__ == "__main__":
    latency = measure_latency()
    print(f"Measured latency: {latency*1000:.2f} milliseconds")


#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <fftw3.h>
#include <portaudio.h>

class AudioLatencyMeasurement {
public:
    static const int FS = 16000;
    static const int BUFFER_SIZE = 256;
    static constexpr double TONE_FREQ = 440.0; // 440 Hz (A4 note)
    static constexpr double TONE_DURATION = 5.0; // 5 seconds
    unsigned long currentFrame = 0;

    std::vector<std::array<double, 2>> audio_out;
    std::vector<std::array<double, 2>> buffer;

    AudioLatencyMeasurement() {
        generate_tone();

        int paddingFrames = BUFFER_SIZE - (audio_out.size() % BUFFER_SIZE);
        if (paddingFrames != BUFFER_SIZE) {
            for (int i = 0; i < paddingFrames; i++) {
                audio_out.push_back({0, 0});
            }
        }

        if (audio_out.size() % BUFFER_SIZE != 0) {
            throw std::runtime_error("audio_out length should be a multiple of buffer_size.");
        }

        buffer.resize(audio_out.size());
    }

    void generate_tone(double frequency = TONE_FREQ, double duration = TONE_DURATION) {
        int total_samples = duration * FS;
        audio_out.reserve(total_samples);
        for (int i = 0; i < total_samples; ++i) {
            double sample = sin(2 * M_PI * frequency * i / FS);
            audio_out.push_back({sample, sample});
        }
    }
    static int playCallback(const void *inputBuffer, void *outputBuffer, unsigned long framesPerBuffer,
                            const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void *userData ) {
        AudioLatencyMeasurement *tool = (AudioLatencyMeasurement*) userData;
        double (*out)[2] = (double (*)[2])outputBuffer;

        for (unsigned long i = 0; i < framesPerBuffer; i++) {
            out[i][0] = tool->audio_out[i + tool->currentFrame][0];
            out[i][1] = tool->audio_out[i + tool->currentFrame][1];
        }

        tool->currentFrame += framesPerBuffer;
        return paContinue;
    }

    static int recordCallback(const void *inputBuffer, void *outputBuffer, unsigned long framesPerBuffer,
                              const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void *userData ) {
        AudioLatencyMeasurement *tool = (AudioLatencyMeasurement*) userData;
        double (*in)[2] = (double (*)[2])inputBuffer;

        for (unsigned long i = 0; i < framesPerBuffer; i++) {
            tool->buffer[i + tool->currentFrame][0] = in[i][0];
            tool->buffer[i + tool->currentFrame][1] = in[i][1];
        }

        tool->currentFrame += framesPerBuffer;
        return paContinue;
    }

    double measureLatency() {
        currentFrame = 0;
        PaStream *playStream, *recordStream;

        Pa_Initialize();

        Pa_OpenDefaultStream(&playStream, 0, 2, paFloat32, FS, BUFFER_SIZE, playCallback, this);
        Pa_OpenDefaultStream(&recordStream, 2, 0, paFloat32, FS, BUFFER_SIZE, recordCallback, this);

        Pa_StartStream(playStream);
        Pa_StartStream(recordStream);

        while (currentFrame < audio_out.size()) {
            Pa_Sleep(100);
        }

        Pa_StopStream(playStream);
        Pa_StopStream(recordStream);

        Pa_CloseStream(playStream);
        Pa_CloseStream(recordStream);

        Pa_Terminate();

        return computeLatency();
    }


    double computeLatency() {
        int n = 2 * audio_out.size();  // Updated size
        fftw_complex *in1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
        fftw_complex *in2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
        fftw_complex *out1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
        fftw_complex *out2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
        fftw_complex *result = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);

        fftw_plan plan_forward1 = fftw_plan_dft_1d(n, in1, out1, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_plan plan_forward2 = fftw_plan_dft_1d(n, in2, out2, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_plan plan_backward = fftw_plan_dft_1d(n, result, result, FFTW_BACKWARD, FFTW_ESTIMATE);

        for (size_t i = 0; i < audio_out.size(); i++) {
            in1[i][0] = buffer[i][0];
            in1[i][1] = buffer[i][1];
            in2[i][0] = audio_out[i][0];
            in2[i][1] = audio_out[i][1];
        }
        for (size_t i = audio_out.size(); i < n; i++) {
            in1[i][0] = 0;
            in1[i][1] = 0;
            in2[i][0] = 0;
            in2[i][1] = 0;
        }

        fftw_execute(plan_forward1);
        fftw_execute(plan_forward2);

        for (size_t i = 0; i < n; i++) {
            result[i][0] = out1[i][0] * out2[i][0] + out1[i][1] * out2[i][1];  // Re(out1*conj(out2))
            result[i][1] = out1[i][0] * out2[i][1] - out1[i][1] * out2[i][0];  // Im(out1*conj(out2))
        }

        fftw_execute(plan_backward);

        double max_val = result[0][0];
        int max_index = 0;
        for (size_t i = 1; i < n; i++) {
            if (result[i][0] > max_val) {
                max_val = result[i][0];
                max_index = i;
            }
        }

        fftw_destroy_plan(plan_forward1);
        fftw_destroy_plan(plan_forward2);
        fftw_destroy_plan(plan_backward);
        fftw_free(in1);
        fftw_free(in2);
        fftw_free(out1);
        fftw_free(out2);
        fftw_free(result);

        double latency = max_index / (double)FS;
        return latency;
    }

};

int main() {
    AudioLatencyMeasurement tool;
    double latency = tool.measureLatency();
    std::cout << "Measured latency: " << latency * 1000 << " milliseconds" << std::endl;
    return 0;
}
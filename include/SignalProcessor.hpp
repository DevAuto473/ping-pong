#pragma once
// ============================================================================
//  SignalProcessor — remote photoplethysmography (rPPG) DSP pipeline.
//
//   raw green mean ─► circular buffer (value, timestamp)
//                  ─► uniform resampling (handles jittery camera FPS)
//                  ─► zero-mean detrending (moving-average baseline removal)
//                  ─► zero-phase Butterworth band-pass 0.75–2.5 Hz (filtfilt)
//                  ─► Hamming window + zero padding + cv::dft
//                  ─► spectral peak (parabolic interpolation) ─► BPM + SNR confidence
//                  ─► time-domain peak detection ─► IBI ─► RMSSD (HRV)
// ============================================================================
#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace vitals {

/// Fixed-capacity ring buffer (no allocations after construction).
template <typename T>
class CircularBuffer {
public:
    explicit CircularBuffer(std::size_t capacity = 0) : data_(capacity) {}

    void push(const T& v) {
        if (data_.empty()) return;
        data_[head_] = v;
        head_ = (head_ + 1) % data_.size();
        if (size_ < data_.size()) ++size_;
    }
    /// i = 0 is the oldest element.
    const T& operator[](std::size_t i) const {
        return data_[(head_ + data_.size() - size_ + i) % data_.size()];
    }
    const T& back() const { return (*this)[size_ - 1]; }
    std::size_t size() const { return size_; }
    std::size_t capacity() const { return data_.size(); }
    bool empty() const { return size_ == 0; }
    bool full() const { return size_ == data_.size(); }
    void clear() { head_ = size_ = 0; }

private:
    std::vector<T> data_;
    std::size_t head_ = 0;
    std::size_t size_ = 0;
};

/// Direct-form-II-transposed second-order section.
struct Biquad {
    double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
    static Biquad butterworthLowpass(double fc, double fs, double q = 0.70710678118654752);
    static Biquad butterworthHighpass(double fc, double fs, double q = 0.70710678118654752);
    /// Filters `x` in place, initial state = steady state for x[0] (reduces start-up transient).
    void process(std::vector<double>& x) const;
};

struct SignalConfig {
    std::size_t bufferSize = 512;   ///< 256–512 samples ≈ 8.5–17 s @ 30 FPS.
    std::size_t minSamples = 150;   ///< Samples before the first estimate (~5 s).
    double lowHz = 0.75;            ///< 45 BPM
    double highHz = 2.5;            ///< 150 BPM
    int filterOrderSections = 2;    ///< Biquads per edge (2 → 4th-order HP + 4th-order LP).
    double detrendWindowSec = 1.5;  ///< Moving-average baseline window.
    int dftMinSize = 2048;          ///< Zero-padded DFT length (spectral interpolation).
    double maxGapSec = 2.0;         ///< Pauses longer than this flush the buffer.
    double bpmSmoothing = 0.25;     ///< EMA weight for the reported BPM.
    std::size_t waveformSeconds = 8;///< Length of filtered waveform exported to the UI.
};

struct SignalMetrics {
    bool valid = false;         ///< Enough data for a heart-rate estimate.
    bool paused = false;        ///< Tracking lost: buffers frozen.
    double bpm = 0.0;           ///< Smoothed heart rate.
    double instantBpm = 0.0;    ///< Unsmoothed spectral peak.
    double confidence = 0.0;    ///< 0..1 from spectral SNR.
    double snrDb = 0.0;
    double rmssdMs = 0.0;       ///< HRV (RMSSD), 0 if insufficient beats.
    double meanIbiMs = 0.0;
    int beatCount = 0;
    double sampleRate = 0.0;    ///< Effective (measured) sampling rate.
    double bufferFill = 0.0;    ///< 0..1 progress of signal acquisition.
    std::vector<float> waveform;      ///< Filtered pulse, normalised to [-1, 1], oldest first.
    std::vector<float> beatMarkers;   ///< Beat positions inside `waveform`, as 0..1 fractions.
};

class SignalProcessor {
public:
    explicit SignalProcessor(const SignalConfig& cfg = SignalConfig{});

    /// Add one raw sample (mean green intensity of the forehead ROI) at time t [s].
    void push(double value, double t);
    /// Signal that tracking is lost at time t; samples stop until the next push().
    void pause(double t);
    void reset();

    /// Runs the full DSP pipeline on the current buffer (cheap: N ≤ 512).
    const SignalMetrics& process();
    const SignalMetrics& metrics() const { return metrics_; }

private:
    struct Sample { double v; double t; };

    bool resampleUniform(std::vector<double>& out, double& fs) const;
    static void detrend(std::vector<double>& x, int window);
    void bandpass(std::vector<double>& x, double fs) const;
    bool spectralPeak(const std::vector<double>& x, double fs,
                      double& peakHz, double& snrDb) const;
    void computeHrv(const std::vector<double>& x, double fs, double hrHz,
                    std::vector<double>& peakIdx);

    SignalConfig cfg_;
    CircularBuffer<Sample> buffer_;
    SignalMetrics metrics_;

    bool paused_ = false;
    double pauseStart_ = 0.0;
    double timeOffset_ = 0.0;   ///< Accumulated removed pause time (keeps timeline contiguous).
    double levelOffset_ = 0.0;  ///< DC re-alignment after a pause.
    double lastRaw_ = 0.0;
    double lastT_ = -1.0;
    double smoothedBpm_ = 0.0;
};

}  // namespace vitals

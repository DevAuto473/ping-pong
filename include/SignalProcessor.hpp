#pragma once
// ============================================================================
//  SignalProcessor — remote photoplethysmography (rPPG) DSP pipeline.
//
//   raw green mean ─► circular buffer (value, timestamp)
//                  ─► uniform resampling (handles jittery camera FPS)
//                  ─► zero-mean detrending (moving-average baseline removal)
//                  ─► zero-phase Butterworth band-pass 0.75–2.5 Hz (filtfilt)
//                  ─► Hamming window + zero padding + cv::dft
//                  ─► spectral peak (parabolic interpolation) ─► BPM
//                  ─► confidence = clamp(peak-lobe power / band power · norm, 0, 1)
//                  ─► time-domain peak detection ─► IBI ─► RMSSD (HRV)
//
//   Respiration (BrPM) uses a separate, longer buffer of the *undetrended*
//   green signal (the cardiac detrender would remove the breathing baseline):
//     normalise ─► linear detrend ─► band-pass 0.15–0.40 Hz (filtfilt)
//               ─► Hamming + cv::dft peak (+ zero-crossing cross-check)
// ============================================================================
#include <opencv2/core.hpp>

#include <cstddef>
#include <deque>
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

    // Confidence = clamp(lobePower / bandPower · confNorm, 0, 1).
    double confLobeHz = 0.12;       ///< Half-width of the "peak" lobe (Hamming main lobe).
    double confNorm = 1.0 / 0.80;   ///< A lobe holding ≥ 80 % of the band power ⇒ 100 %.

    // Respiratory rate (baseline modulation of the green channel).
    std::size_t respBufferSize = 1024; ///< ≈ 34 s @ 30 FPS (≥ 4 breaths at 7 BrPM).
    double respMinSeconds = 15.0;      ///< Data span required for the first estimate.
    double respLowHz = 0.15;           ///< 9 breaths/min
    double respHighHz = 0.40;          ///< 24 breaths/min
    double respLobeHz = 0.05;
    double respConfFull = 0.85;        ///< Lobe/band power ratio that maps to 100 %.
    double respRefLowHz = 0.45;        ///< Reference band for the noise floor (above breathing,
    double respRefHighHz = 0.70;       ///< below the cardiac band).
    double respMinPeakDb = 10.0;       ///< Peak must exceed the reference median by this much.
    double respMinConfidence = 0.55;   ///< Below this the estimate is reported as invalid.
    double respSmoothing = 0.30;
    std::size_t respStabilityFrames = 150;  ///< ≈ 5 s of consecutive estimates must agree.
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
    // Respiration (breaths per minute).
    bool respValid = false;
    double respRate = 0.0;          ///< Smoothed BrPM.
    double respConfidence = 0.0;    ///< 0..1 (clamped).
    double respProgress = 0.0;      ///< 0..1 acquisition progress toward respMinSeconds.

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

    struct Spectrum {
        double peakHz = 0.0;
        double ratio = 0.0;     ///< lobe power / total band power, 0..1
        double snrDb = 0.0;     ///< lobe power vs. remaining band power
        double peakToRefDb = 0.0; ///< peak bin vs. median bin of the reference band (if requested)
    };

    static bool resampleUniform(const CircularBuffer<Sample>& buf, std::vector<double>& out, double& fs);
    static void detrend(std::vector<double>& x, int window);
    static void removeLinearTrend(std::vector<double>& x);
    /// Zero-phase Butterworth band-pass; `sections` biquads per edge.
    static void bandpass(std::vector<double>& x, double fs, double lo, double hi, int sections);
    /// Hamming + zero-padded cv::dft; peak search and power ratio inside [lo, hi].
    /// If refHi > refLo, also computes the peak-to-median ratio against that reference band.
    static bool spectrum(const std::vector<double>& x, double fs, double lo, double hi,
                         double lobeHz, int nfftMin, Spectrum& out, double refLo = 0.0,
                         double refHi = 0.0);
    void processRespiration();
    void computeHrv(const std::vector<double>& x, double fs, double hrHz,
                    std::vector<double>& peakIdx);

    SignalConfig cfg_;
    CircularBuffer<Sample> buffer_;
    CircularBuffer<Sample> respBuffer_;   ///< Same timeline, longer history.
    SignalMetrics metrics_;

    bool paused_ = false;
    double pauseStart_ = 0.0;
    double timeOffset_ = 0.0;   ///< Accumulated removed pause time (keeps timeline contiguous).
    double levelOffset_ = 0.0;  ///< DC re-alignment after a pause.
    double lastRaw_ = 0.0;
    double lastT_ = -1.0;
    double smoothedBpm_ = 0.0;
    double smoothedResp_ = 0.0;
    std::deque<double> respHistory_;
};

}  // namespace vitals

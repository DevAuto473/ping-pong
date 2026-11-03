#pragma once
// ============================================================================
//  SignalProcessor — remote photoplethysmography (rPPG) DSP pipeline.
//
//   skin-masked mean RGB (forehead + cheeks) ─► circular buffer (BGR, timestamp)
//                  ─► uniform resampling at the *measured* rate (variable FPS safe)
//                  ─► POS projection (Wang et al., IEEE TBME 2017): per 1.6 s window
//                     normalise each channel by its temporal mean, project onto the
//                     plane orthogonal to skin tone, alpha-tune, overlap-add.
//                     Cancels intensity / auto-exposure changes common to R, G, B.
//                     (Green-only mode remains available for comparison.)
//                  ─► zero-mean detrending (moving-average baseline removal)
//                  ─► zero-phase Butterworth band-pass 0.75–2.5 Hz (filtfilt)
//                  ─► Hamming window + zero padding + cv::dft
//                  ─► peak = true local maximum of the full spectrum (never the
//                     band edge), weighted toward the previous estimate (tracking)
//                  ─► parabolic interpolation ─► BPM
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
#include <utility>
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

enum class PulseMethod {
    POS,    ///< Plane-Orthogonal-to-Skin chrominance (default, illumination robust)
    Green,  ///< Mean green channel only (legacy, for comparison)
};

struct SignalConfig {
    PulseMethod method = PulseMethod::POS;
    std::size_t bufferSize = 512;   ///< 256–512 samples ≈ 17–25 s @ 20–30 FPS.
    double minSeconds = 6.0;        ///< Data span before the first estimate (FPS independent).
    double posWindowSec = 1.6;      ///< POS sliding window (≈ one cardiac cycle at 40 BPM).
    double lowHz = 0.75;            ///< 45 BPM
    double highHz = 2.5;            ///< 150 BPM
    int filterOrderSections = 2;    ///< Biquads per edge (2 → 4th-order HP + 4th-order LP).
    double detrendWindowSec = 1.5;  ///< Moving-average baseline window.
    int dftMinSize = 2048;          ///< Zero-padded DFT length (spectral interpolation).
    double maxGapSec = 2.0;         ///< Pauses longer than this flush the buffer.
    double bpmSmoothing = 0.25;     ///< EMA weight for the reported BPM.
    std::size_t waveformSeconds = 8;///< Length of filtered waveform exported to the UI.

    // Peak tracking: a candidate's power is weighted by
    //   trackFloor + (1 − trackFloor) · exp(−(ΔBPM / trackSigmaBpm)²)
    // so a peak far from the previous estimate must be ≥ 1/trackFloor × stronger to win.
    double trackSigmaBpm = 12.0;
    double trackFloor = 0.5;
    double reacquireSec = 3.0;      ///< If the unweighted strongest peak disagrees for this long,
                                    ///< tracking re-locks onto it (prevents harmonic lock-in).

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
    double respStabilitySec = 5.0;     ///< Estimates over this span must agree (±1.5 BrPM).
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
    PulseMethod method = PulseMethod::POS;  ///< Method actually used for this estimate.
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

    /// Add one colour sample (mean B, G, R of the skin ROIs) at time t [s].
    void push(const cv::Vec3d& bgr, double t);
    /// Scalar sample (e.g. green mean). Forces Green mode until reset(): POS needs colour.
    void push(double value, double t);

    /// Skin-masked mean colour over several ROIs (pixel-count weighted). Pixels are kept
    /// when they fall in a broad YCrCb skin range and are neither clipped nor in deep
    /// shadow; if fewer than 25 % qualify (unusual lighting), the plain mean is used.
    /// Returns false if no ROI has usable pixels.
    static bool skinMean(const cv::Mat& bgrFrame, const std::vector<cv::Rect>& rois, cv::Vec3d& out);
    /// Signal that tracking is lost at time t; samples stop until the next push().
    void pause(double t);
    void reset();

    /// Runs the full DSP pipeline on the current buffer (cheap: N ≤ 512).
    const SignalMetrics& process();
    const SignalMetrics& metrics() const { return metrics_; }

private:
    struct Sample { cv::Vec3d c; double t; };  ///< c = (B, G, R)

    struct Spectrum {
        double peakHz = 0.0;
        double ratio = 0.0;     ///< lobe power / total band power, 0..1
        double snrDb = 0.0;     ///< lobe power vs. remaining band power
        double peakToRefDb = 0.0; ///< peak bin vs. median bin of the reference band (if requested)
        bool edgeOnly = false;    ///< band contained no true local maximum (only edge leakage)
        double rawPeakHz = 0.0;   ///< strongest local maximum without the tracking prior
    };

    /// Resamples channel `ch` (0 = B, 1 = G, 2 = R) onto a uniform grid at the measured rate.
    static bool resampleUniform(const CircularBuffer<Sample>& buf, int ch, std::vector<double>& out, double& fs);
    /// POS pulse synthesis from uniformly sampled B, G, R traces.
    static std::vector<double> pos(const std::vector<double>& b, const std::vector<double>& g,
                                   const std::vector<double>& r, int window);
    static void detrend(std::vector<double>& x, int window);
    static void removeLinearTrend(std::vector<double>& x);
    /// Zero-phase Butterworth band-pass; `sections` biquads per edge.
    static void bandpass(std::vector<double>& x, double fs, double lo, double hi, int sections);
    /// Hamming + zero-padded cv::dft; peak search and power ratio inside [lo, hi].
    /// If refHi > refLo, also computes the peak-to-median ratio against that reference band.
    /// Only true local maxima of the full spectrum are peak candidates, so monotonic leakage from
    /// below the band can never "latch" onto the lower edge. If prevHz > 0, candidates are
    /// weighted toward it (tracking) using trackSigmaHz / trackFloor.
    static bool spectrum(const std::vector<double>& x, double fs, double lo, double hi,
                         double lobeHz, int nfftMin, Spectrum& out, double refLo = 0.0,
                         double refHi = 0.0, double prevHz = 0.0, double trackSigmaHz = 0.2,
                         double trackFloor = 1.0);
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
    cv::Vec3d levelOffset_{0, 0, 0};  ///< Per-channel DC re-alignment after a pause.
    cv::Vec3d lastRaw_{0, 0, 0};
    bool scalarInput_ = false;        ///< A scalar push() happened: POS not possible.
    double lastT_ = -1.0;
    double smoothedBpm_ = 0.0;
    double disagreeSince_ = -1.0;  ///< Time the unweighted peak started disagreeing with tracking.
    double smoothedResp_ = 0.0;
    std::deque<std::pair<double, double>> respHistory_;  ///< (t, BrPM)
};

}  // namespace vitals

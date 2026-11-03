#pragma once
// ============================================================================
//  SignalProcessor — multi-ROI remote photoplethysmography (rPPG) DSP pipeline.
//
//  Per region (forehead, left cheek, right cheek):
//     skin-masked mean RGB ─► circular buffer (timestamped)
//       ─► uniform resampling at the *measured* rate (variable FPS safe)
//       ─► POS projection (Wang et al., IEEE TBME 2017)
//       ─► detrend ─► Butterworth band-pass 0.75–2.5 Hz (zero phase)
//       ─► Signal Quality Index  SQI = ½·SNR-term + ½·(1 − spectral entropy)-term
//          (zeroed for regions that are mostly non-skin / shadowed / not tracked)
//
//  Fusion: unit-variance region signals averaged with weights SQI² (regions under
//  the minimum SQI are discarded) ─► Hamming + cv::dft ─► true local spectral
//  maximum with tracking prior ─► BPM, confidence, beat times.
//
//  Side products (consumed by ClinicalAnalytics):
//    * morphology waveform: same fusion on a wide band (0.6 Hz … min(8 Hz, 0.45·fs)),
//      oriented as blood volume (intensity falls when volume rises) — for APG.
//    * confirmed beat times on a segment-aware timeline — for HRV / Poincaré.
//    * AC/DC of the red and blue channels — for the ratio-of-ratios SpO₂ surrogate.
//    * background (non-skin) pulse check — flags ambient light / screen flicker
//      that could masquerade as a pulse.
//    * respiration from the baseline modulation of the green channel.
// ============================================================================
#include <opencv2/core.hpp>

#include <array>
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

enum RoiIndex : int { kForehead = 0, kLeftCheek = 1, kRightCheek = 2, kRoiCount = 3 };

struct RoiSample {
    cv::Vec3d bgr{0, 0, 0};
    double skinFraction = 1.0;  ///< Fraction of ROI pixels passing the skin gate.
    bool valid = false;
};

/// Everything the pipeline needs from one video frame.
struct FrameSample {
    std::array<RoiSample, kRoiCount> roi{};
    cv::Vec3d background{0, 0, 0};  ///< Mean colour of non-face patches beside the head.
    bool hasBackground = false;
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

    // Peak tracking: candidate power weighted by trackFloor + (1−trackFloor)·exp(−(ΔBPM/σ)²).
    double trackSigmaBpm = 12.0;
    double trackFloor = 0.5;
    double reacquireSec = 3.0;      ///< Re-lock onto the strongest peak after this much disagreement.

    // Confidence = clamp(lobePower / bandPower · confNorm, 0, 1).
    double confLobeHz = 0.12;
    double confNorm = 1.0 / 0.80;

    // Multi-ROI quality / fusion.
    // SQI terms are chance-corrected: each maps its noise-only level to 0 and its clean-pulse
    // level to 1 (levels measured on band-limited noise vs. a two-harmonic pulse, 512 @ 30 FPS).
    double sqiRatioNoise = 0.30;    ///< Peak-lobe ratio of pure noise (≈ p75; the max-peak bias).
    double sqiRatioClean = 0.65;
    double sqiEntropyNoise = 0.92;  ///< Normalised spectral entropy of pure noise.
    double sqiEntropyClean = 0.60;
    double minRoiSqi = 0.15;        ///< Regions below this SQI are discarded from fusion.
    double minSkinFraction = 0.25;  ///< Regions with less skin (shadow, hair, occlusion) are dropped.
    double minRoiValidFraction = 0.9;

    // Pulse morphology (APG) band.
    double morphLowHz = 0.6;
    double morphHighHz = 8.0;       ///< Clamped to 0.45·fs.
    double beatConfirmSec = 1.0;    ///< Beats are emitted once this far from the window end.
    double minBeatConfidence = 0.3; ///< No beats are emitted below this HR confidence…
    double minBeatSqi = 0.75;       ///< …or below this fused SQI. Measured: SQI ≥ 0.8 → beat timing
                                    ///< SD ≤ 45 ms, no gross errors; SQI ≈ 0.5 → SD ≈ 130 ms with
                                    ///< half-cycle errors (would fake HRV / ectopy).

    // Ambient / screen flicker check.
    double ambientMinRatio = 0.5;   ///< Background lobe ratio at the face HR that raises a flag…
    double ambientMinRelAmp = 0.5;  ///< …if its relative AC is at least this fraction of the face's.

    // Respiratory rate (baseline modulation of the green channel).
    std::size_t respBufferSize = 1024;
    double respMinSeconds = 15.0;
    double respLowHz = 0.15;
    double respHighHz = 0.40;
    double respLobeHz = 0.05;
    double respConfFull = 0.85;
    double respRefLowHz = 0.45;
    double respRefHighHz = 0.70;
    double respMinPeakDb = 10.0;
    double respMinConfidence = 0.55;
    double respSmoothing = 0.30;
    double respStabilitySec = 5.0;
};

struct BeatEvent {
    double t = 0.0;     ///< Effective (pause-spliced) timeline, seconds.
    int segment = 0;    ///< Increments at every pause / reset: never pair beats across segments.
};

struct SignalMetrics {
    bool valid = false;
    bool paused = false;
    double bpm = 0.0;
    double instantBpm = 0.0;
    double confidence = 0.0;    ///< 0..1
    double snrDb = 0.0;
    double rmssdMs = 0.0;       ///< Window RMSSD (display only; session HRV lives in ClinicalAnalytics).
    double meanIbiMs = 0.0;
    int beatCount = 0;
    double sampleRate = 0.0;
    PulseMethod method = PulseMethod::POS;
    double bufferFill = 0.0;

    // Multi-ROI quality.
    std::array<double, kRoiCount> roiSqi{};     ///< 0..1 per region.
    std::array<double, kRoiCount> roiWeight{};  ///< Normalised fusion weights.
    double sqi = 0.0;                           ///< SQI of the fused signal.

    // Beats confirmed during the last process() call.
    std::vector<BeatEvent> newBeats;

    // Morphology waveform (blood-volume orientation), uniform at sampleRate, oldest first.
    std::vector<double> morphWave;

    // Ratio-of-ratios inputs.
    bool ratioValid = false;
    double acDcRed = 0.0, acDcBlue = 0.0, ratioRB = 0.0;

    // Ambient flicker / screen check.
    bool backgroundChecked = false;
    bool ambientSuspect = false;
    double backgroundRatio = 0.0;  ///< Background lobe ratio at the face HR.

    // Respiration.
    bool respValid = false;
    double respRate = 0.0;
    double respConfidence = 0.0;
    double respProgress = 0.0;

    std::vector<float> waveform;
    std::vector<float> beatMarkers;
};

class SignalProcessor {
public:
    explicit SignalProcessor(const SignalConfig& cfg = SignalConfig{});

    /// Full multi-ROI sample.
    void push(const FrameSample& s, double t);
    /// Single colour sample (treated as the forehead region only).
    void push(const cv::Vec3d& bgr, double t);
    /// Scalar sample (e.g. green mean). Forces Green mode until reset(): POS needs colour.
    void push(double value, double t);

    /// Skin-masked mean colour over several ROIs (pixel-count weighted). Returns false if empty.
    static bool skinMean(const cv::Mat& bgrFrame, const std::vector<cv::Rect>& rois, cv::Vec3d& out);
    /// Builds a FrameSample: skin-masked colour + skin fraction per ROI, plus background patches.
    static FrameSample sampleFrame(const cv::Mat& bgrFrame, const std::array<cv::Rect, kRoiCount>& rois,
                                   const std::vector<cv::Rect>& background);

    void pause(double t);
    void reset();

    const SignalMetrics& process();
    const SignalMetrics& metrics() const { return metrics_; }

private:
    struct Sample {
        std::array<cv::Vec3d, kRoiCount> c{};  ///< (B, G, R) per region
        std::array<float, kRoiCount> skin{};
        std::array<bool, kRoiCount> ok{};
        cv::Vec3d bg{0, 0, 0};
        bool bgOk = false;
        double t = 0.0;
    };

    struct Spectrum {
        double peakHz = 0.0;
        double ratio = 0.0;       ///< lobe power / total band power, 0..1
        double snrDb = 0.0;
        double peakToRefDb = 0.0;
        double entropy = 1.0;     ///< normalised spectral entropy in band, 0..1
        bool edgeOnly = false;
        double rawPeakHz = 0.0;
    };

    template <typename Getter>
    static bool resample(const CircularBuffer<Sample>& buf, Getter get, std::vector<double>& out, double& fs);
    static std::vector<double> pos(const std::vector<double>& b, const std::vector<double>& g,
                                   const std::vector<double>& r, int window);
    static void detrend(std::vector<double>& x, int window);
    static void removeLinearTrend(std::vector<double>& x);
    static void bandpass(std::vector<double>& x, double fs, double lo, double hi, int sections);
    static bool spectrum(const std::vector<double>& x, double fs, double lo, double hi,
                         double lobeHz, int nfftMin, Spectrum& out, double refLo = 0.0,
                         double refHi = 0.0, double prevHz = 0.0, double trackSigmaHz = 0.2,
                         double trackFloor = 1.0);
    static double rms(const std::vector<double>& x);
    static void normalise(std::vector<double>& x);

    /// Raw pulse trace of one region (POS or green), not yet filtered. Blood-volume orientation.
    std::vector<double> regionPulse(int roi, double fs) const;
    double sqiOf(const Spectrum& sp) const;
    void processRatio(const std::array<double, kRoiCount>& w, double f0);
    void processBackground(double f0, const std::vector<double>& faceGreenRel, double fs);
    void emitBeats(const std::vector<double>& peakIdx, double fs);
    void processRespiration();
    void computeHrv(const std::vector<double>& x, double fs, double hrHz, std::vector<double>& peakIdx);

    SignalConfig cfg_;
    CircularBuffer<Sample> buffer_;
    CircularBuffer<Sample> respBuffer_;
    SignalMetrics metrics_;

    bool paused_ = false;
    double pauseStart_ = 0.0;
    double timeOffset_ = 0.0;
    std::array<cv::Vec3d, kRoiCount> levelOffset_{};
    std::array<cv::Vec3d, kRoiCount> lastRaw_{};
    std::array<bool, kRoiCount> haveRaw_{};
    cv::Vec3d bgOffset_{0, 0, 0}, bgLast_{0, 0, 0};
    bool haveBg_ = false;
    bool scalarInput_ = false;
    double lastT_ = -1.0;
    double smoothedBpm_ = 0.0;
    double disagreeSince_ = -1.0;
    double smoothedResp_ = 0.0;
    std::deque<std::pair<double, double>> respHistory_;
    int segment_ = 0;
    double segmentStartT_ = -1e9;  ///< Effective time the current segment began.
    double lastBeatT_ = -1e9;
};

}  // namespace vitals

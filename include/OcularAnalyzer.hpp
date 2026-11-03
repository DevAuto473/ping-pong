#pragma once
// ============================================================================
//  OcularAnalyzer — appearance-based Eye Aspect Ratio (EAR) proxy, blink
//  dynamics and PERCLOS fatigue index from the two stabilised ocular patches.
//
//  YuNet provides only 5 landmarks (no eyelid contour), so the classic 6-point
//  EAR is not available. Instead:
//
//      EAR ≈ vertical aperture of the dark region (iris / pupil / lash line)
//            ─────────────────────────────────────────────────────────────
//                         horizontal width of the eye
//
//  The dark region is segmented per frame relative to the patch's own median
//  intensity (illumination-invariant). Per column of the central band we take
//  the longest vertical dark run; an open eye yields ≈ 0.25–0.35 of the eye
//  width, a closed one only the thin lash line (≈ 0.03–0.08).
//
//  Blink logic (all thresholds adaptive, relative to the rolling mean EAR of
//  *open-eye* frames):
//      closure  : EAR < 0.75 · mean   (exit at 0.85 · mean — hysteresis)
//      blink    : closure lasting 2 … 8 frames (scaled with FPS)
//      incomplete blink : blink whose minimum EAR stayed above 0.45 · mean
//      long closure     : closure longer than the maximum blink length
//      PERCLOS  : % of frames over the last 60 s with EAR < closedRatio · mean
// ============================================================================
#include <opencv2/core.hpp>

#include <deque>
#include <vector>

namespace vitals {

struct OcularConfig {
    cv::Size patchSize{64, 40};      ///< Eye patches are resampled to this size.
    double eyeWidthFraction = 0.83;  ///< Eye width as a fraction of the patch width.
    double darkRatio = 0.72;         ///< Pixel is "dark" if < darkRatio · patch median.
    double centralBand = 0.25;       ///< Columns in [band, 1-band] are evaluated.
    double emaWeight = 0.75;         ///< Weight of the new EAR sample (light smoothing).

    double meanWindowSec = 10.0;     ///< Rolling-mean EAR window (open-eye frames only).
    double warmupSec = 2.0;          ///< Time before metrics become valid.
    double blinkRatio = 0.75;        ///< Closure starts below this fraction of mean EAR.
    double recoverRatio = 0.85;      ///< … and ends above this one (hysteresis).
    double completeRatio = 0.62;     ///< Minimum EAR / mean below this → complete blink.
                                     ///< Tuned on real webcam data: the lash line keeps the
                                     ///< aperture proxy at ≈ 0.45–0.6 of open even when shut.
    double closedRatio = 0.35;       ///< PERCLOS: frame counts as closed below this ratio.

    int minBlinkFrames = 2;          ///< Blink duration limits, in frames at `referenceFps`.
    int maxBlinkFrames = 8;
    double referenceFps = 30.0;      ///< Frame limits scale with the measured FPS.

    double perclosWindowSec = 60.0;
    double rateWindowSec = 60.0;
};

struct OcularMetrics {
    bool valid = false;         ///< Warm-up finished and enough open-eye frames seen.
    bool paused = false;
    double ear = 0.0;           ///< Current EAR estimate (open ≈ 0.30).
    double meanEar = 0.0;       ///< Rolling mean of open-eye EAR.
    double threshold = 0.0;     ///< Adaptive blink threshold (0.75 · meanEar).
    double openness = 0.0;      ///< ear / meanEar.
    bool eyesClosed = false;    ///< Currently inside a closure.
    double blinkRate = 0.0;     ///< Blinks / minute (complete + incomplete), last 60 s.
    int totalBlinks = 0;
    int incompleteBlinks = 0;   ///< Session total.
    double incompleteRatio = 0.0;///< Incomplete / all blinks over the last 60 s.
    int longClosures = 0;       ///< Closures longer than the maximum blink (micro-sleep cue).
    double perclos = 0.0;       ///< 0..1 fraction of frames "closed" in the last 60 s.
    double lastBlinkDurationMs = 0.0;
    double lastBlinkMinRatio = 0.0; ///< Deepest EAR / mean of the last blink (diagnostic for tuning).
    long framesProcessed = 0;   ///< Frames that actually reached the analyzer (diagnostics).
    double sessionTrackedSec = 0.0; ///< Whole-session tracked time (session PERCLOS / blink rate).
    double sessionClosedSec = 0.0;
};

class OcularAnalyzer {
public:
    explicit OcularAnalyzer(const OcularConfig& cfg = OcularConfig{});

    /// grayFrame: full-frame 8-bit grayscale. Eye rects come from FaceTracker.
    const OcularMetrics& update(const cv::Mat& grayFrame, const cv::Rect& leftEye,
                                const cv::Rect& rightEye, double t);
    void pause(double t);
    void reset();
    const OcularMetrics& metrics() const { return metrics_; }

private:
    struct EarSample { double t; double v; };
    struct FrameFlag { double t; double dt; bool closed; };
    struct BlinkEvent { double t; bool complete; };

    /// Returns the EAR proxy of one eye patch, or < 0 if the ROI is unusable.
    double patchEar(const cv::Mat& gray, const cv::Rect& roi);
    void pushOpenSample(double ear, double t);
    void updateBlinkState(double ear, double t);
    void prune(double t);

    OcularConfig cfg_;
    OcularMetrics metrics_;

    cv::Mat patch_;
    std::vector<int> colRun_;
    std::vector<int> histogram_;

    std::deque<EarSample> openHistory_;   ///< Open-eye EAR samples for the rolling mean.
    double openSum_ = 0.0;
    std::deque<FrameFlag> frames_;        ///< Per-frame closed flags for PERCLOS / rate.
    double trackedSec_ = 0.0;
    double closedSec_ = 0.0;
    std::deque<BlinkEvent> blinks_;

    double smoothedEar_ = -1.0;
    double startT_ = -1.0;
    double lastT_ = -1.0;
    double fpsEma_ = 0.0;

    bool inClosure_ = false;
    int closureFrames_ = 0;
    double closureStartT_ = 0.0;
    double closureMinEar_ = 0.0;
};

}  // namespace vitals

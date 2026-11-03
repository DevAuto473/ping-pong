#pragma once
// ============================================================================
//  OcularAnalyzer — appearance-based eye-openness (EAR proxy), blink dynamics
//  and PERCLOS fatigue index from the two stabilised ocular patches.
//
//  With only 5 YuNet landmarks there are no eyelid contour points, so the
//  classic 6-point EAR cannot be computed. Instead each eye patch yields:
//    * contrast cue  : σ/μ of the patch (iris/sclera contrast vanishes when closed)
//    * aperture cue  : vertical spread of horizontal-edge energy (|∂I/∂y|),
//                      i.e. distance between upper and lower lid edges
//  Their product is normalised against an adaptive "eye open" baseline,
//  producing an openness ratio r ∈ [0, ~1.2] that is mapped to an EAR-like
//  scale (open ≈ 0.30).
// ============================================================================
#include <opencv2/core.hpp>

#include <deque>
#include <vector>

namespace vitals {

struct OcularConfig {
    cv::Size patchSize{64, 40};     ///< Eye patches are resampled to this size.
    double nominalOpenEar = 0.30;   ///< EAR assigned to a fully open eye.
    double blinkThreshold = 0.60;   ///< r below → closing (falling edge).
    double recoverThreshold = 0.75; ///< r above → re-opened (hysteresis).
    double completeThreshold = 0.35;///< Minimum r below this → complete blink.
    double minBlinkSec = 0.05;
    double maxBlinkSec = 0.50;      ///< Longer closures are counted as long closures.
    double perclosClosure = 0.80;   ///< P80: eye is "closed" when ≥80 % closed (r ≤ 0.2).
    double perclosWindowSec = 60.0;
    double baselineWindowSec = 12.0;
    double floorRatio = 0.45;       ///< Closed-eye cue level relative to open baseline.
    double smoothing = 0.55;        ///< EMA weight on the raw openness cue.
    double warmupSec = 3.0;
};

struct OcularMetrics {
    bool valid = false;         ///< Baseline established.
    bool paused = false;
    double ear = 0.0;           ///< EAR-scaled openness of the current frame.
    double openness = 0.0;      ///< r (0 closed … 1 baseline open).
    bool eyesClosed = false;
    double blinkRate = 0.0;     ///< Blinks / minute (complete + incomplete) over last 60 s.
    int totalBlinks = 0;
    int incompleteBlinks = 0;   ///< Total incomplete blinks this session.
    double incompleteRatio = 0.0;///< Incomplete / all blinks over the last 60 s.
    int longClosures = 0;       ///< Closures > maxBlinkSec (micro-sleep indicator).
    double perclos = 0.0;       ///< 0..1 fraction of tracked time eyes closed (P80) in last 60 s.
    double lastBlinkDurationMs = 0.0;
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
    struct TimedValue { double t; double v; };
    struct BlinkEvent { double t; bool complete; };
    struct ClosureSample { double t; double dt; bool closed; };

    double patchCue(const cv::Mat& gray, const cv::Rect& roi);
    void updateBaseline(double cue, double t);
    void updateBlinkState(double r, double t);
    void prune(double t);

    OcularConfig cfg_;
    OcularMetrics metrics_;

    cv::Mat patch_, patchF_, dy_;
    std::vector<double> sortScratch_;

    std::deque<TimedValue> cueHistory_;     ///< For adaptive baseline.
    std::deque<ClosureSample> closureHistory_; ///< Tracked-time samples for PERCLOS / rate.
    std::deque<BlinkEvent> blinks_;

    double smoothedCue_ = -1.0;
    double openBaseline_ = 0.0;
    double floor_ = 0.0;
    double startT_ = -1.0;
    double lastT_ = -1.0;
    bool paused_ = false;

    bool inClosure_ = false;
    double closureStart_ = 0.0;
    double closureMin_ = 1.0;
};

}  // namespace vitals

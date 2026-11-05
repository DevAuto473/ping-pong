#pragma once
// ============================================================================
//  QualityAdvisor — turns the tracker / DSP state into one actionable hint for
//  the person in front of the camera ("MOVE CLOSER", "ADD LIGHT", ...), and
//  DrowsinessMonitor — raises an alert on eye closures that look like
//  micro-sleeps or on a high PERCLOS.
//
//  Both are debounced: a hint appears only after its condition has held for
//  `onsetSec` and stays visible for `holdSec` after it clears, so the HUD never
//  flickers between messages frame by frame.
// ============================================================================
#include "FaceTracker.hpp"
#include "OcularAnalyzer.hpp"
#include "SignalProcessor.hpp"

#include <array>
#include <vector>

namespace vitals {

/// Ordered by priority: the first active one is shown.
enum class Advice : int {
    None = -1,
    HoldStill = 0,
    MoveCloser,
    FaceCamera,
    TooDark,
    TooBright,
    UncoverForehead,
    Flicker,
    LowFps,
    Count
};

/// Short upper-case text for the HUD / log ("" for Advice::None).
const char* adviceText(Advice a);
/// Machine-friendly identifier for CSV / JSON ("" for Advice::None).
const char* adviceId(Advice a);

struct AdviceConfig {
    float minInterOcularPx = 50.f;   ///< Smaller faces give too few skin pixels.
    float maxYawDeg = 25.f;          ///< Beyond this the cheeks / forehead foreshorten.
    double minLuma = 60.0;           ///< Skin luma (0–255) below this → under-exposed.
    double maxLuma = 215.0;          ///< … above this → glare / clipping.
    double minForeheadSkin = 0.25;   ///< Same as SignalConfig::minSkinFraction.
    double minFps = 15.0;            ///< Camera only (files are processed offline).
    double onsetSec = 1.0;           ///< Condition must persist this long …
    double holdSec = 1.5;            ///< … and the hint stays this long after it clears.
};

struct AdviceInput {
    const FaceObservation* face = nullptr;
    const FrameSample* sample = nullptr;   ///< May be null (no face this frame).
    const SignalMetrics* signal = nullptr;
    bool motionRejected = false;
    double fps = 0.0;
    bool offline = false;                  ///< Video file: FPS hint disabled.
};

class QualityAdvisor {
public:
    explicit QualityAdvisor(const AdviceConfig& cfg = AdviceConfig{});

    /// Instantaneous conditions (no debouncing), highest priority first.
    static std::vector<Advice> assess(const AdviceInput& in, const AdviceConfig& cfg);

    /// Debounced top hint for time `t` (seconds).
    Advice update(const AdviceInput& in, double t);
    Advice current() const { return current_; }
    void reset();

private:
    AdviceConfig cfg_;
    std::array<double, static_cast<int>(Advice::Count)> since_{};    ///< Onset time, < 0 = inactive.
    std::array<double, static_cast<int>(Advice::Count)> lastSeen_{}; ///< Last time the condition held.
    Advice current_ = Advice::None;
};

struct DrowsinessConfig {
    double closureAlertSec = 1.0;   ///< Eyes continuously closed this long → alert (micro-sleep).
    double perclosAlert = 0.15;     ///< PERCLOS (60 s) at or above this → alert.
    double longClosureHoldSec = 4.0;///< A completed long closure keeps the alert this long.
};

class DrowsinessMonitor {
public:
    explicit DrowsinessMonitor(const DrowsinessConfig& cfg = DrowsinessConfig{});

    /// Returns true while the alert is active.
    bool update(const OcularMetrics& om, double t);
    bool active() const { return active_; }
    /// Number of distinct alert episodes (rising edges) this session.
    int episodes() const { return episodes_; }
    /// Reason of the current alert ("EYES CLOSED", "HIGH PERCLOS", "LONG EYE CLOSURE").
    const char* reason() const { return reason_; }
    void reset();

private:
    DrowsinessConfig cfg_;
    double closedSince_ = -1.0;
    int lastLongClosures_ = 0;
    double longClosureT_ = -1e9;
    bool active_ = false;
    int episodes_ = 0;
    const char* reason_ = "";
};

}  // namespace vitals

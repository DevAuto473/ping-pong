#pragma once
// ============================================================================
//  ClinicalAnalytics — session-level aggregation of rPPG / ocular telemetry and
//  derived research metrics, plus the end-of-session English report.
//
//  Inputs are the per-frame outputs of SignalProcessor and OcularAnalyzer; this
//  module owns no image processing. Heavy analyses (HRV spectrum, APG template)
//  run on a fixed cadence (analysisPeriodSec), not every frame.
//
//  IMPORTANT — scientific limits (also printed in the report):
//   * Blood pressure: a single-site camera PPG has no validated mapping to mmHg
//     without per-person cuff calibration. Only waveform morphology features are
//     reported; the mmHg field reads "N/A".
//   * SpO2: RGB cameras lack the 660/940 nm bands of pulse oximeters. The
//     red/blue ratio-of-ratios R is real; the % value from A − B·R is
//     UNCALIBRATED unless A and B are fitted against a reference oximeter. Values
//     are never clamped into the normal range (that would hide everything).
//   * Rhythm: only inter-beat-interval irregularity can be screened; "normal
//     sinus rhythm" requires an ECG and is never claimed.
//   * LF/HF: standard practice is 5 min; shorter recordings are flagged.
// ============================================================================
#include "OcularAnalyzer.hpp"
#include "SignalProcessor.hpp"

#include <string>
#include <vector>

namespace vitals {

struct ClinicalConfig {
    double minHrConfidence = 0.35;  ///< Frames below this HR confidence are excluded from statistics.
    double analysisPeriodSec = 2.0; ///< Cadence of HRV-spectrum / APG / SpO2 analyses.

    // Inter-beat intervals.
    double ibiMinMs = 333.0;        ///< 180 BPM
    double ibiMaxMs = 1500.0;       ///<  40 BPM
    double nnTolerance = 0.20;      ///< |IBI − local median| ≤ 20 % → normal-to-normal (NN).
    int localMedianHalfWidth = 5;

    // Ectopic screening: premature interval followed by a compensatory one.
    double prematureRatio = 0.80;
    double compensatoryRatio = 1.10;
    int minEctopicForFlag = 2;

    // Frequency-domain HRV.
    double hrvMinSeconds = 60.0;    ///< Minimum contiguous NN span for LF/HF.
    double hrvStandardSeconds = 300.0;
    double tachogramHz = 4.0;
    double welchSegmentSec = 64.0;
    double maxNnGapSec = 3.0;       ///< Larger gaps split the tachogram.

    // APG / morphology.
    int apgMinBeats = 5;
    int apgTemplatePoints = 100;
    double apgMinSeconds = 8.0;

    // SpO2 = A − B·R (UNCALIBRATED defaults; fit A, B against a reference oximeter).
    double spo2A = 110.0;
    double spo2B = 25.0;
    double spo2PlausibleMin = 80.0;
};

struct HrvSummary {
    int beats = 0;              ///< Beats received.
    int intervals = 0;          ///< Physiologically plausible IBIs.
    int nnCount = 0;            ///< Normal-to-normal intervals.
    bool timeValid = false;
    double meanNnMs = 0.0, sdnnMs = 0.0, rmssdMs = 0.0;

    bool poincareValid = false;
    double sd1Ms = 0.0, sd2Ms = 0.0;
    int ectopicBeats = 0;
    bool irregular = false;

    bool freqValid = false;
    double analysedSec = 0.0;   ///< Length of the tachogram used for LF/HF.
    double lfMs2 = 0.0, hfMs2 = 0.0, lfHf = 0.0;
    double rsaBrpm = 0.0;       ///< HF-band peak frequency (respiratory sinus arrhythmia).
};

struct ApgSummary {
    bool valid = false;         ///< a and b waves found (b/a available).
    double ba = 0.0;
    bool agiValid = false;      ///< c, d and e waves found as well.
    double agi = 0.0;           ///< (b − c − d − e) / a
    double riseTimeMs = 0.0;    ///< Foot → systolic peak.
    double crestRatio = 0.0;    ///< Rise time / cardiac cycle.
    bool riValid = false;
    double reflectionIndex = 0.0;  ///< Diastolic (or inflection) height / systolic height.
    int beatsUsed = 0;
};

class ClinicalAnalytics {
public:
    explicit ClinicalAnalytics(const ClinicalConfig& cfg = ClinicalConfig{});

    /// Call once per frame after SignalProcessor::process() and OcularAnalyzer::update().
    void update(double t, const SignalMetrics& sm, const OcularMetrics& om, bool faceTracked,
                bool motionRejected);

    const HrvSummary& hrv() const { return hrv_; }
    const ApgSummary& apg() const { return apgLatest_; }
    /// Uncalibrated SpO2 surrogate and its ratio R; false if no pulsatile R/B data.
    bool spo2(double& percent, double& ratio) const;

    /// One-line runtime telemetry for --log.
    std::string statusLine(double t, const SignalMetrics& sm, const OcularMetrics& om,
                           bool motionRejected) const;
    /// End-of-session English report.
    std::string report() const;
    /// Machine-readable session summary (JSON object; unavailable values are null).
    std::string reportJson() const;

    /// CSV time series (one row per call, typically 1 Hz). Empty cells = not available.
    static std::string csvHeader();
    std::string csvRow(double t, const SignalMetrics& sm, const OcularMetrics& om, bool faceTracked,
                       bool motionRejected, const char* advice, bool drowsy) const;

    /// Discards all session statistics (keeps the configuration).
    void reset() { *this = ClinicalAnalytics(cfg_); }
    /// Drowsiness alert episodes, reported in the session summary.
    void setDrowsinessEpisodes(int n) { drowsyEpisodes_ = n; }

    // Pure analysis kernels (public for unit tests).
    static HrvSummary analyseHrv(const std::vector<BeatEvent>& beats, const ClinicalConfig& cfg);
    static ApgSummary analyseApg(const std::vector<double>& morph, double fs, double hrHz,
                                 const ClinicalConfig& cfg);

private:
    void periodicAnalysis(const SignalMetrics& sm);
    /// Session respiration rate. The validated baseline-modulation estimate (peak prominence,
    /// zero-crossing agreement, stability) is preferred; the RSA peak of the HRV spectrum is
    /// only a fallback, and is rejected when it sits on the HF band edge (noise, not breathing).
    bool respiration(double& brpm, std::string& source) const;
    static double median(std::vector<double> v);

    ClinicalConfig cfg_;

    double tFirst_ = -1.0, tLast_ = -1.0, lastAnalysis_ = -1e9;
    long frames_ = 0, trackedFrames_ = 0, motionFrames_ = 0;
    double sqiSum_ = 0.0;
    long sqiCount_ = 0;
    double hrSum_ = 0.0, hrMin_ = 1e9, hrMax_ = 0.0;
    long hrCount_ = 0;
    long ambientChecked_ = 0, ambientSuspect_ = 0;

    std::vector<BeatEvent> beats_;
    std::vector<double> ratioSamples_, respSamples_;
    std::vector<double> baSamples_, agiSamples_, riseSamples_, crestSamples_, riSamples_;

    int drowsyEpisodes_ = 0;

    HrvSummary hrv_;
    ApgSummary apgLatest_;
    OcularMetrics ocular_;
};

}  // namespace vitals

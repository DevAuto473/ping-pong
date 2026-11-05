// Tests for the ROI lock / DC re-alignment, acquisition tips, drowsiness alert and the
// CSV / JSON exporters (no camera needed).
#include "ClinicalAnalytics.hpp"
#include "FaceTracker.hpp"
#include "QualityAdvisor.hpp"
#include "SignalProcessor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

using namespace vitals;

namespace {
int gFailures = 0;
#define EXPECT(cond, ...)                                      \
    do {                                                       \
        if (!(cond)) {                                         \
            ++gFailures;                                       \
            std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                          \
            std::printf("\n");                                 \
        }                                                      \
    } while (0)

constexpr double kPi = 3.14159265358979323846;

/// YuNet-layout detection: box + 5 landmarks + score, shifted by (dx, dy).
cv::Mat detection(float dx, float dy) {
    return (cv::Mat_<float>(1, 15) << 500 + dx, 200 + dy, 200, 240, 560 + dx, 300 + dy, 640 + dx, 300 + dy,
            600 + dx, 350 + dy, 570 + dx, 400 + dy, 630 + dx, 400 + dy, 0.95f);
}

// ---------------------------------------------------------------------------
void testRoiLock() {
    FaceTracker tr(FaceTrackerConfig{});
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> jit(-0.8f, 0.8f);
    double t = 0.0;
    for (int i = 0; i < 30; ++i, t += 1.0 / 30) tr.ingest(detection(jit(rng), jit(rng)), cv::Mat(), {1280, 720}, t);
    const cv::Rect f0 = tr.last().forehead, c0 = tr.last().leftCheek;
    int changes = 0;
    for (int i = 0; i < 150; ++i, t += 1.0 / 30) {
        const FaceObservation& o = tr.ingest(detection(jit(rng), jit(rng)), cv::Mat(), {1280, 720}, t);
        changes += (o.forehead != f0) + (o.leftCheek != c0);
    }
    std::printf("  ROI lock: %d ROI changes over 150 jittered frames\n", changes);
    EXPECT(changes == 0, "sub-pixel detector jitter moved the rPPG ROIs %d times", changes);

    // A genuine move (20 px) must be followed.
    for (int i = 0; i < 30; ++i, t += 1.0 / 30) tr.ingest(detection(20.f, 0.f), cv::Mat(), {1280, 720}, t);
    const int moved = tr.last().forehead.x - f0.x;
    EXPECT(moved >= 15 && moved <= 25, "forehead followed a 20 px move by %d px", moved);

    // Disabled lock: the ROI follows the jitter (sanity check of the test itself).
    FaceTrackerConfig off;
    off.roiLockIod = 0.f;
    FaceTracker tr2(off);
    int changes2 = 0;
    cv::Rect prev;
    for (int i = 0; i < 150; ++i, t += 1.0 / 30) {
        const FaceObservation& o = tr2.ingest(detection(3.f * jit(rng), 3.f * jit(rng)), cv::Mat(), {1280, 720}, t);
        changes2 += i > 0 && o.forehead != prev;
        prev = o.forehead;
    }
    EXPECT(changes2 > 20, "without the lock the ROI should move (%d)", changes2);
}

// ---------------------------------------------------------------------------
/// Pulse at 72 BPM with a colour step every 3 s (ROI relocation); returns final confidence.
double runSteps(bool flagMoves, double& bpm) {
    SignalProcessor sp;
    std::mt19937 rng(9);
    std::normal_distribution<double> n(0.0, 0.05);
    cv::Vec3d offset(0, 0, 0);
    for (int i = 0; i < 30 * 25; ++i) {
        const double t = i / 30.0;
        const double p = 0.5 * (1.0 - std::cos(2.0 * kPi * 1.2 * t));
        bool moved = false;
        if (i > 0 && i % 90 == 0) {
            offset += cv::Vec3d(6.0, 9.0, 7.0) * ((i / 90) % 2 ? 1.0 : -0.6);  // new pixel set
            moved = true;
        }
        FrameSample s;
        for (int k = 0; k < kRoiCount; ++k) {
            const cv::Vec3d c(100 * (1 - 0.003 * p) + n(rng), 120 * (1 - 0.01 * p) + n(rng),
                              150 * (1 - 0.004 * p) + n(rng));
            s.roi[k] = RoiSample{c + offset, 0.9, true, flagMoves && moved};
        }
        sp.push(s, t);
        sp.process();
    }
    bpm = sp.metrics().bpm;
    return sp.metrics().confidence;
}

void testRoiStepRealign() {
    double bpmFlag = 0, bpmRaw = 0;
    const double cFlag = runSteps(true, bpmFlag), cRaw = runSteps(false, bpmRaw);
    std::printf("  ROI steps: flagged conf %.2f (%.1f bpm)  unflagged conf %.2f (%.1f bpm)\n", cFlag, bpmFlag, cRaw,
                bpmRaw);
    EXPECT(std::abs(bpmFlag - 72.0) < 2.0, "bpm with re-alignment %.1f", bpmFlag);
    EXPECT(cFlag > 0.8, "confidence with re-alignment %.2f", cFlag);
    EXPECT(cFlag >= cRaw - 1e-9, "re-alignment should not lower confidence (%.2f vs %.2f)", cFlag, cRaw);
}

// ---------------------------------------------------------------------------
void testAdvisor() {
    FaceObservation face;
    face.valid = true;
    face.interOcular = 90.f;
    FrameSample sample;
    for (int k = 0; k < kRoiCount; ++k) sample.roi[k] = RoiSample{cv::Vec3d(110, 140, 180), 0.9, true};
    SignalMetrics sm;
    AdviceInput in;
    in.face = &face;
    in.sample = &sample;
    in.signal = &sm;
    in.fps = 30.0;
    const AdviceConfig cfg;
    EXPECT(QualityAdvisor::assess(in, cfg).empty(), "good conditions should give no tip");

    in.motionRejected = true;
    face.interOcular = 30.f;
    face.yawDeg = 40.f;
    auto v = QualityAdvisor::assess(in, cfg);
    EXPECT(v.size() == 3 && v[0] == Advice::HoldStill && v[1] == Advice::MoveCloser && v[2] == Advice::FaceCamera,
           "priority order wrong (%zu tips)", v.size());
    in.motionRejected = false;
    face.interOcular = 90.f;
    face.yawDeg = 0.f;

    for (int k = 0; k < kRoiCount; ++k) sample.roi[k].bgr = cv::Vec3d(20, 30, 40);
    v = QualityAdvisor::assess(in, cfg);
    EXPECT(v.size() == 1 && v[0] == Advice::TooDark, "dark scene not detected");
    for (int k = 0; k < kRoiCount; ++k) sample.roi[k].bgr = cv::Vec3d(230, 235, 240);
    v = QualityAdvisor::assess(in, cfg);
    EXPECT(v.size() == 1 && v[0] == Advice::TooBright, "glare not detected");
    for (int k = 0; k < kRoiCount; ++k) sample.roi[k].bgr = cv::Vec3d(110, 140, 180);
    sample.roi[kForehead].skinFraction = 0.1;
    v = QualityAdvisor::assess(in, cfg);
    EXPECT(v.size() == 1 && v[0] == Advice::UncoverForehead, "covered forehead not detected");
    sample.roi[kForehead].skinFraction = 0.9;
    in.fps = 10.0;
    EXPECT(QualityAdvisor::assess(in, cfg).size() == 1, "low fps not detected");
    in.offline = true;
    EXPECT(QualityAdvisor::assess(in, cfg).empty(), "low fps must be ignored for video files");
    in.fps = 30.0;
    face.valid = false;
    in.motionRejected = true;
    EXPECT(QualityAdvisor::assess(in, cfg).empty(), "no tips without a face");
    face.valid = true;

    // Debouncing: a single-frame glitch never shows; a persistent condition does, and is held.
    QualityAdvisor adv;
    double t = 0.0;
    in.motionRejected = true;
    EXPECT(adv.update(in, t) == Advice::None, "tip shown before onset");
    in.motionRejected = false;
    for (int i = 0; i < 90; ++i) t += 1.0 / 30, adv.update(in, t);
    EXPECT(adv.current() == Advice::None, "one-frame glitch produced a tip");
    in.motionRejected = true;
    for (int i = 0; i < 40; ++i) t += 1.0 / 30, adv.update(in, t);
    EXPECT(adv.current() == Advice::HoldStill, "persistent motion should show HOLD STILL");
    in.motionRejected = false;
    for (int i = 0; i < 15; ++i) t += 1.0 / 30, adv.update(in, t);
    EXPECT(adv.current() == Advice::HoldStill, "tip should be held for a moment after clearing");
    for (int i = 0; i < 60; ++i) t += 1.0 / 30, adv.update(in, t);
    EXPECT(adv.current() == Advice::None, "tip should disappear after the hold time");
}

// ---------------------------------------------------------------------------
void testDrowsiness() {
    DrowsinessMonitor dm;
    OcularMetrics om;
    om.valid = true;
    double t = 0.0;
    for (int i = 0; i < 60; ++i, t += 1.0 / 30) dm.update(om, t);
    EXPECT(!dm.active(), "alert while alert");
    om.eyesClosed = true;  // a normal blink (≈ 200 ms) must not alarm
    for (int i = 0; i < 6; ++i, t += 1.0 / 30) dm.update(om, t);
    EXPECT(!dm.active(), "a blink triggered the alert");
    for (int i = 0; i < 30; ++i, t += 1.0 / 30) dm.update(om, t);
    EXPECT(dm.active() && std::string(dm.reason()) == "EYES CLOSED", "1.2 s closure not alerted");
    om.eyesClosed = false;
    om.longClosures = 1;  // the analyzer classifies the closure once the eye re-opens
    dm.update(om, t);
    EXPECT(dm.active() && std::string(dm.reason()) == "LONG EYE CLOSURE", "long closure hold missing");
    for (int i = 0; i < 150; ++i, t += 1.0 / 30) dm.update(om, t);
    EXPECT(!dm.active(), "alert should clear after the hold time");
    om.perclos = 0.2;
    dm.update(om, t);
    EXPECT(dm.active() && std::string(dm.reason()) == "HIGH PERCLOS", "high PERCLOS not alerted");
    EXPECT(dm.episodes() == 2, "episodes %d (expected 2)", dm.episodes());
    om.valid = false;
    EXPECT(!dm.update(om, t), "alert without valid ocular data");
}

// ---------------------------------------------------------------------------
int countChar(const std::string& s, char c) { return static_cast<int>(std::count(s.begin(), s.end(), c)); }

void testExporters() {
    ClinicalAnalytics ca;
    SignalMetrics sm;
    OcularMetrics om;
    const std::string header = ClinicalAnalytics::csvHeader();
    std::string row = ca.csvRow(1.5, sm, om, false, false, "", false);
    EXPECT(countChar(header, ',') == countChar(row, ','), "CSV header/row column mismatch (%d vs %d)",
           countChar(header, ','), countChar(row, ','));
    EXPECT(row.rfind("1.500,0,0,,", 0) == 0, "invalid metrics should give empty cells: %s", row.c_str());

    sm.valid = true;
    sm.bpm = 71.94;
    sm.confidence = 0.9;
    om.valid = true;
    om.perclos = 0.05;
    row = ca.csvRow(2.0, sm, om, true, true, adviceId(Advice::HoldStill), true);
    EXPECT(countChar(header, ',') == countChar(row, ','), "CSV column mismatch on a valid row");
    EXPECT(row.find(",71.9,") != std::string::npos, "bpm missing from CSV row: %s", row.c_str());
    EXPECT(row.find("hold_still,1") != std::string::npos, "advice / drowsy missing: %s", row.c_str());

    // JSON of an empty session: valid structure with nulls.
    std::string js = ca.reportJson();
    EXPECT(countChar(js, '{') == countChar(js, '}'), "unbalanced JSON braces");
    EXPECT(js.find("\"mean_bpm\": null") != std::string::npos, "empty session should report null HR");
    EXPECT(js.find("nan") == std::string::npos && js.find("inf") == std::string::npos, "non-finite JSON value");

    // After a real session the JSON carries numbers, and reset() clears them.
    sm.respValid = true;
    sm.respRate = 15.0;
    for (int i = 0; i < 90; ++i) ca.update(i / 30.0, sm, om, true, false);
    js = ca.reportJson();
    EXPECT(js.find("\"respiration_brpm\": 15.0") != std::string::npos, "respiration missing from JSON");
    EXPECT(ca.report().find("15.0 BrPM (Derived from rPPG baseline modulation)") != std::string::npos,
           "validated respiration should be reported");
    EXPECT(js.find("\"mean_bpm\": 71.9") != std::string::npos, "session HR missing from JSON:\n%s", js.c_str());
    ca.reset();
    EXPECT(ca.reportJson().find("\"mean_bpm\": null") != std::string::npos, "reset() did not clear the session");
}

}  // namespace

int main() {
    std::printf("[roi lock]\n");
    testRoiLock();
    std::printf("[roi step re-alignment]\n");
    testRoiStepRealign();
    std::printf("[acquisition tips]\n");
    testAdvisor();
    std::printf("[drowsiness alert]\n");
    testDrowsiness();
    std::printf("[csv / json export]\n");
    testExporters();
    std::printf(gFailures ? "%d FAILURE(S)\n" : "ALL TESTS PASSED\n", gFailures);
    return gFailures ? 1 : 0;
}

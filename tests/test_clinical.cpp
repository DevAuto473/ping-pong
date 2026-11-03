// Regression tests for multi-ROI fusion, SQI, ambient check, SpO2 ratio, head-pose speed,
// HRV (time / frequency / Poincaré / ectopic), APG morphology and the session report.
#include "ClinicalAnalytics.hpp"
#include "FaceTracker.hpp"
#include "OcularAnalyzer.hpp"
#include "SignalProcessor.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

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
using namespace vitals;

// Periodic blood-volume template: systolic + reflected diastolic wave (phase 0..1).
double bvTemplate(double ph) {
    ph -= std::floor(ph);
    auto g = [](double x, double mu, double s) {
        double d = x - mu;
        d -= std::round(d);  // periodic
        return std::exp(-0.5 * d * d / (s * s));
    };
    return g(ph, 0.18, 0.06) + 0.45 * g(ph, 0.50, 0.09);
}

/// Beat train with RR(t) = base + modulation; returns beat times.
std::vector<double> beatTimes(double seconds, double baseMs, double ampMs, double modHz, std::mt19937& rng,
                              double jitterMs = 3.0) {
    std::normal_distribution<double> j(0.0, jitterMs);
    std::vector<double> t{0.0};
    while (t.back() < seconds) {
        const double rr = baseMs + ampMs * std::sin(2 * kPi * modHz * t.back()) + j(rng);
        t.push_back(t.back() + rr / 1000.0);
    }
    return t;
}

std::vector<BeatEvent> toEvents(const std::vector<double>& t, int seg = 1) {
    std::vector<BeatEvent> e;
    for (double x : t) e.push_back({x, seg});
    return e;
}

// ---------------------------------------------------------------------------
void testHrvTimeDomainExact() {
    // Known IBIs: 800, 820, 790, 810, 800, 830, 780 ... → compare with direct formulas.
    const std::vector<double> ibi{800, 820, 790, 810, 800, 830, 780, 805, 815, 795, 800, 810};
    std::vector<double> t{0.0};
    for (double v : ibi) t.push_back(t.back() + v / 1000.0);
    const HrvSummary h = ClinicalAnalytics::analyseHrv(toEvents(t), ClinicalConfig{});
    double mean = 0, ss = 0, sd = 0;
    for (double v : ibi) mean += v;
    mean /= ibi.size();
    for (std::size_t i = 1; i < ibi.size(); ++i) ss += (ibi[i] - ibi[i - 1]) * (ibi[i] - ibi[i - 1]);
    const double rmssd = std::sqrt(ss / (ibi.size() - 1));
    for (double v : ibi) sd += (v - mean) * (v - mean);
    const double sdnn = std::sqrt(sd / (ibi.size() - 1));
    std::printf("  exact: RMSSD %.2f (expect %.2f)  SDNN %.2f (expect %.2f)  NN %d\n", h.rmssdMs, rmssd, h.sdnnMs,
                sdnn, h.nnCount);
    EXPECT(h.timeValid, "time domain invalid");
    EXPECT(std::abs(h.rmssdMs - rmssd) < 1e-6, "RMSSD %.4f vs %.4f", h.rmssdMs, rmssd);
    EXPECT(std::abs(h.sdnnMs - sdnn) < 1e-6, "SDNN %.4f vs %.4f", h.sdnnMs, sdnn);
    EXPECT(h.ectopicBeats == 0, "no ectopy expected");
}

void testLfHf() {
    std::mt19937 rng(1);
    // Respiratory sinus arrhythmia at 0.25 Hz (15 BrPM) → HF dominant.
    HrvSummary hf = ClinicalAnalytics::analyseHrv(toEvents(beatTimes(150, 850, 45, 0.25, rng)), ClinicalConfig{});
    // Mayer-wave (baroreflex) modulation at 0.1 Hz → LF dominant.
    HrvSummary lf = ClinicalAnalytics::analyseHrv(toEvents(beatTimes(150, 850, 45, 0.10, rng)), ClinicalConfig{});
    std::printf("  RSA 0.25 Hz : LF/HF %.2f  RSA-derived resp %.1f BrPM (expect 15.0)  span %.0f s\n", hf.lfHf,
                hf.rsaBrpm, hf.analysedSec);
    std::printf("  Mayer 0.10 Hz: LF/HF %.2f\n", lf.lfHf);
    EXPECT(hf.freqValid && lf.freqValid, "frequency domain invalid");
    EXPECT(hf.lfHf < 0.5, "HF-modulated series should be parasympathetic (LF/HF %.2f)", hf.lfHf);
    EXPECT(lf.lfHf > 2.0, "LF-modulated series should be sympathetic (LF/HF %.2f)", lf.lfHf);
    EXPECT(std::abs(hf.rsaBrpm - 15.0) < 1.0, "RSA resp %.1f", hf.rsaBrpm);

    // Too short → no LF/HF.
    HrvSummary shortRun = ClinicalAnalytics::analyseHrv(toEvents(beatTimes(40, 850, 45, 0.25, rng)), ClinicalConfig{});
    EXPECT(!shortRun.freqValid, "40 s must not produce LF/HF");
}

void testEctopicAndMissedBeats() {
    std::mt19937 rng(2);
    std::vector<double> t = beatTimes(120, 800, 10, 0.25, rng);
    // Inject 4 premature beats (beat arrives 30 % early, next interval compensatory).
    int injected = 0;
    for (std::size_t i = 20; i + 2 < t.size() && injected < 4; i += 25, ++injected) t[i] -= 0.24;
    HrvSummary h = ClinicalAnalytics::analyseHrv(toEvents(t), ClinicalConfig{});
    std::printf("  injected 4 PVC-like events -> ectopic %d irregular %d  SD1/SD2 %.2f\n", h.ectopicBeats,
                h.irregular ? 1 : 0, h.sd1Ms / std::max(1e-9, h.sd2Ms));
    EXPECT(h.ectopicBeats == 4, "ectopic %d vs 4", h.ectopicBeats);
    EXPECT(h.irregular, "should flag irregularity");

    // Missed detections (one beat dropped) must NOT be called ectopic.
    std::vector<double> m = beatTimes(120, 800, 10, 0.25, rng);
    for (std::size_t i = 30; i < m.size(); i += 30) m.erase(m.begin() + static_cast<std::ptrdiff_t>(i));
    HrvSummary hm = ClinicalAnalytics::analyseHrv(toEvents(m), ClinicalConfig{});
    std::printf("  dropped beats -> ectopic %d (expect 0), NN %d\n", hm.ectopicBeats, hm.nnCount);
    EXPECT(hm.ectopicBeats == 0, "missed beats misread as ectopic (%d)", hm.ectopicBeats);
    EXPECT(!hm.irregular, "missed beats flagged irregular");

    // Intervals must never be formed across segments (tracking gaps).
    std::vector<BeatEvent> seg = toEvents(beatTimes(30, 800, 10, 0.25, rng), 1);
    for (const BeatEvent& b : toEvents(beatTimes(30, 800, 10, 0.25, rng), 2)) seg.push_back({b.t + 30.2, 2});
    HrvSummary hs = ClinicalAnalytics::analyseHrv(seg, ClinicalConfig{});
    EXPECT(hs.intervals == static_cast<int>(seg.size()) - 2, "cross-segment interval created (%d)", hs.intervals);
}

// ---------------------------------------------------------------------------
void testApgMorphology(double fs) {
    const double hr = 70.0, T = 60.0 / hr;
    std::vector<double> x;
    for (double t = 0; t < 20.0; t += 1.0 / fs) x.push_back(bvTemplate(t / T));
    // Ground truth from the analytic template at high resolution (same definitions).
    const int N = 2000;
    std::vector<double> tp(N);
    int foot = 0;
    for (int k = 0; k < N; ++k) {
        tp[k] = bvTemplate(static_cast<double>(k) / N);
        if (tp[k] < tp[foot]) foot = k;
    }
    std::vector<double> c(N);
    for (int k = 0; k < N; ++k) c[k] = bvTemplate(static_cast<double>(foot + k) / N);
    int sp = 0;
    for (int k = 0; k < N; ++k) if (c[k] > c[sp]) sp = k;
    const double base = c[0];
    int dia = -1;
    for (int k = sp + N / 10; k < 85 * N / 100; ++k)
        if (c[k] > c[k - 1] && c[k] >= c[k + 1]) { dia = k; break; }
    const double riTrue = (c[dia] - base) / (c[sp] - base);
    const double riseTrue = static_cast<double>(sp) / N * T * 1000.0;
    // b/a truth from the continuous second derivative.
    std::vector<double> acc(N, 0.0);
    for (int k = 1; k + 1 < N; ++k) acc[k] = c[k + 1] - 2 * c[k] + c[k - 1];
    int ia = 1;
    for (int k = 1; k < sp; ++k) if (acc[k] > acc[ia]) ia = k;
    int ib = ia + 1;
    for (int k = ia + 1; k < sp + N / 10; ++k) if (acc[k] < acc[ib]) ib = k;
    const double baTrue = acc[ib] / acc[ia];

    const ApgSummary a = ClinicalAnalytics::analyseApg(x, fs, hr / 60.0, ClinicalConfig{});
    std::printf("  APG @ %2.0f fps: b/a %.2f (true %.2f)  rise %.0f ms (true %.0f)  RI %.2f (true %.2f)  beats %d"
                "  AGI %s\n", fs, a.ba, baTrue, a.riseTimeMs, riseTrue, a.reflectionIndex, riTrue, a.beatsUsed,
                a.agiValid ? std::to_string(a.agi).c_str() : "n/a");
    EXPECT(a.valid, "APG not resolved");
    EXPECT(a.ba < 0.0, "b/a must be negative (%.2f)", a.ba);
    EXPECT(std::abs(a.ba - baTrue) < 0.25, "b/a %.2f vs %.2f", a.ba, baTrue);
    EXPECT(std::abs(a.riseTimeMs - riseTrue) < 40.0, "rise %.0f vs %.0f ms", a.riseTimeMs, riseTrue);
    EXPECT(a.riValid && std::abs(a.reflectionIndex - riTrue) < 0.12, "RI %.2f vs %.2f", a.reflectionIndex, riTrue);
}

// ---------------------------------------------------------------------------
struct RoiSim {
    double pulseScale = 1.0;   // relative pulse amplitude
    double noise = 0.2;        // additive noise (8-bit levels)
    double skin = 0.9;
};

void runRoiStream(SignalProcessor& sp, double seconds, double hr, const RoiSim (&r)[kRoiCount], bool bgFlicker,
                  std::vector<double>* bpmOut = nullptr) {
    std::mt19937 rng(31);
    std::normal_distribution<double> n01(0.0, 1.0);
    const double dc[3] = {110, 130, 180}, pv[3] = {0.53, 0.77, 0.33};
    for (double t = 0; t < seconds; t += 1.0 / 30) {
        const double bv = bvTemplate(t * hr / 60.0);
        FrameSample s;
        for (int k = 0; k < kRoiCount; ++k) {
            cv::Vec3d c;
            for (int ch = 0; ch < 3; ++ch)
                c[ch] = dc[ch] * (1.0 - 0.004 * r[k].pulseScale * pv[ch] * bv) + r[k].noise * n01(rng);
            s.roi[k] = RoiSample{c, r[k].skin, true};
        }
        const double flick = bgFlicker ? 1.0 + 0.01 * std::sin(2 * kPi * hr / 60.0 * t) : 1.0;
        s.background = cv::Vec3d(60, 70, 80) * flick + cv::Vec3d(n01(rng), n01(rng), n01(rng)) * 0.3;
        s.hasBackground = true;
        sp.push(s, t);
        const auto& m = sp.process();
        if (bpmOut && m.valid) bpmOut->push_back(m.bpm);
    }
}

void testMultiRoiFusion() {
    SignalProcessor sp;
    // Forehead clean, left cheek drowned in noise, right cheek mostly non-skin (shadow / hair).
    const RoiSim r[kRoiCount] = {{1.0, 0.05, 0.9}, {1.0, 6.0, 0.9}, {1.0, 0.05, 0.10}};
    runRoiStream(sp, 25, 72, r, false);
    const auto& m = sp.metrics();
    std::printf("  SQI F %.2f L %.2f R %.2f | weights %.2f %.2f %.2f | fused SQI %.2f | HR %.1f\n", m.roiSqi[0],
                m.roiSqi[1], m.roiSqi[2], m.roiWeight[0], m.roiWeight[1], m.roiWeight[2], m.sqi, m.bpm);
    EXPECT(m.roiSqi[0] > m.roiSqi[1] + 0.2, "noisy cheek should have clearly lower SQI");
    EXPECT(m.roiWeight[2] == 0.0, "non-skin region must be discarded");
    EXPECT(m.roiWeight[0] > 0.7, "clean forehead should dominate the fusion (%.2f)", m.roiWeight[0]);
    EXPECT(std::abs(m.bpm - 72.0) < 2.0, "fused HR %.1f", m.bpm);
    EXPECT(!m.ambientSuspect, "static background must not be flagged");

    // All three good → all contribute.
    SignalProcessor sp2;
    const RoiSim g[kRoiCount] = {{1.0, 0.3, 0.9}, {1.0, 0.3, 0.9}, {1.0, 0.3, 0.9}};
    runRoiStream(sp2, 25, 72, g, false);
    const auto& m2 = sp2.metrics();
    std::printf("  equal regions: SQI %.2f %.2f %.2f | weights %.2f %.2f %.2f | HR %.1f\n", m2.roiSqi[0], m2.roiSqi[1],
                m2.roiSqi[2], m2.roiWeight[0], m2.roiWeight[1], m2.roiWeight[2], m2.bpm);
    EXPECT(m2.roiWeight[0] > 0.2 && m2.roiWeight[1] > 0.2 && m2.roiWeight[2] > 0.2, "equal regions should share");
}

void testAmbientFlicker() {
    SignalProcessor sp;
    const RoiSim r[kRoiCount] = {{1.0, 0.2, 0.9}, {1.0, 0.2, 0.9}, {1.0, 0.2, 0.9}};
    runRoiStream(sp, 25, 72, r, true);
    const auto& m = sp.metrics();
    std::printf("  background flickering at the pulse rate -> ambient suspect %d (bg ratio %.2f)\n",
                m.ambientSuspect ? 1 : 0, m.backgroundRatio);
    EXPECT(m.ambientSuspect, "background modulated at HR must be flagged");
}

void testRatioOfRatios() {
    SignalProcessor sp;
    const RoiSim r[kRoiCount] = {{1.0, 0.05, 0.9}, {1.0, 0.05, 0.9}, {1.0, 0.05, 0.9}};
    runRoiStream(sp, 25, 72, r, false);
    const auto& m = sp.metrics();
    const double expect = 0.33 / 0.53;  // pulse vector R / B
    std::printf("  ratio-of-ratios R = %.3f (expect %.3f) acdc red %.5f blue %.5f\n", m.ratioRB, expect, m.acDcRed,
                m.acDcBlue);
    EXPECT(m.ratioValid && std::abs(m.ratioRB - expect) < 0.06, "R %.3f vs %.3f", m.ratioRB, expect);
}

// ---------------------------------------------------------------------------
void testHeadPoseSpeed() {
    FaceTrackerConfig c;
    FaceTracker tr(c);
    auto det = [](float noseDx) {
        cv::Mat d = (cv::Mat_<float>(1, 15) << 500, 200, 200, 240, 560, 300, 640, 300, 600.f + noseDx, 350, 570, 400,
                     630, 400, 0.95f);
        return d;
    };
    float stillMax = 0, turnMax = 0;
    double t = 0;
    for (int i = 0; i < 30; ++i, t += 1.0 / 30) stillMax = std::max(stillMax, tr.ingest(det(0), cv::Mat(), {1280, 720}, t).angularSpeedDps);
    for (int i = 0; i < 15; ++i, t += 1.0 / 30)  // turn ≈ 45° in 0.5 s
        turnMax = std::max(turnMax, tr.ingest(det(static_cast<float>(i) * 2.7f), cv::Mat(), {1280, 720}, t).angularSpeedDps);
    std::printf("  head still: %.1f deg/s   head turning: %.1f deg/s (yaw %.0f deg)\n", stillMax, turnMax,
                tr.last().yawDeg);
    EXPECT(stillMax < 5.0, "still head speed %.1f", stillMax);
    EXPECT(turnMax > 30.0, "turning head speed %.1f should exceed the 30 deg/s freeze threshold", turnMax);
}

// ---------------------------------------------------------------------------
ClinicalAnalytics runSession(double noise, double seconds, std::string* report) {
    SignalProcessor sp;
    OcularAnalyzer oa;
    ClinicalAnalytics ca;
    std::mt19937 rng(41);
    std::normal_distribution<double> n01(0.0, 1.0);
    const double dc[3] = {110, 130, 180}, pv[3] = {0.53, 0.77, 0.33};
    double phase = 0.0, t = 0.0;
    const double dt = 1.0 / 30;
    for (int i = 0; i < static_cast<int>(30 * seconds); ++i, t += dt) {
        const double hr = 72.0 + 5.0 * std::sin(2 * kPi * 0.25 * t);  // RSA
        phase += hr / 60.0 * dt;
        const double bv = bvTemplate(phase);
        FrameSample s;
        for (int k = 0; k < kRoiCount; ++k) {
            cv::Vec3d c;
            for (int ch = 0; ch < 3; ++ch) c[ch] = dc[ch] * (1.0 - 0.004 * pv[ch] * bv) + noise * n01(rng);
            s.roi[k] = RoiSample{c, 0.9, true};
        }
        sp.push(s, t);
        const auto& sm = sp.process();
        ca.update(t, sm, oa.metrics(), true, false);
    }
    if (report) *report = ca.report();
    return ca;
}

void testSessionReport() {
    // 100 s of clean three-region signal with RSA (±5 BPM at 0.25 Hz) → full report.
    std::string rep;
    const ClinicalAnalytics ca = runSession(0.05, 100, &rep);
    std::printf("%s", rep.c_str());
    const HrvSummary& h = ca.hrv();
    EXPECT(h.timeValid && h.nnCount > 80, "NN count %d", h.nnCount);
    EXPECT(h.freqValid, "LF/HF should be available after 100 s");
    EXPECT(h.freqValid && std::abs(h.rsaBrpm - 15.0) < 1.5, "RSA respiration %.1f vs 15", h.rsaBrpm);
    EXPECT(!h.irregular, "clean rhythm flagged irregular");
    for (const char* key : {"CLINICAL VITALS & TELEMETRY REPORT", "[ CARDIOVASCULAR & HEMODYNAMIC PROFILE ]",
                            "[ AUTONOMIC & HRV ANALYSIS ]", "[ RESPIRATORY & OCULAR FATIGUE ]", "Session Duration:",
                            "Motion Rejected Frames:", "Vascular Aging Index:", "UNCALIBRATED",
                            "FOR RESEARCH & MONITORING PURPOSES ONLY"})
        EXPECT(rep.find(key) != std::string::npos, "report is missing '%s'", key);
    EXPECT(rep.find("Normal Sinus Rhythm") == std::string::npos, "report must not claim an ECG rhythm diagnosis");
}

void testWeakSessionNoFalseAlarm() {
    // Same physiology, but noisy (fused SQI ≈ 0.5): beat timing is unreliable, so HRV and the
    // arrhythmia screen must say N/A instead of reporting fake variability or ectopy.
    std::string rep;
    const ClinicalAnalytics ca = runSession(0.20, 100, &rep);
    std::printf("  weak signal: beats logged %d, HRV valid %d, ectopic flag %d\n", ca.hrv().beats,
                ca.hrv().timeValid ? 1 : 0, ca.hrv().irregular ? 1 : 0);
    EXPECT(!ca.hrv().irregular, "weak signal produced a false ectopy alarm");
    EXPECT(rep.find("Suspected Ectopic") == std::string::npos, "report shows false ectopy");
}
}  // namespace

int main() {
    std::printf("[HRV]\n");
    testHrvTimeDomainExact();
    testLfHf();
    testEctopicAndMissedBeats();
    std::printf("[APG / morphology]\n");
    testApgMorphology(30.0);
    testApgMorphology(20.0);
    std::printf("[Multi-ROI / SQI / ambient / SpO2 ratio]\n");
    testMultiRoiFusion();
    testAmbientFlicker();
    testRatioOfRatios();
    std::printf("[Head pose]\n");
    testHeadPoseSpeed();
    std::printf("[Session report]\n");
    testSessionReport();
    testWeakSessionNoFalseAlarm();
    std::printf(gFailures ? "%d FAILURE(S)\n" : "ALL TESTS PASSED\n", gFailures);
    return gFailures ? 1 : 0;
}

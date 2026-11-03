// Self-contained regression tests for the DSP and ocular modules (no camera needed).
#include "OcularAnalyzer.hpp"
#include "SignalProcessor.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdio>
#include <random>

namespace {
int gFailures = 0;
#define EXPECT(cond, ...)                                   \
    do {                                                    \
        if (!(cond)) {                                      \
            ++gFailures;                                    \
            std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                       \
            std::printf("\n");                              \
        }                                                   \
    } while (0)

constexpr double kPi = 3.14159265358979323846;

/// Synthetic rPPG: pulsatile beats with HRV + drift + noise, jittered sampling.
void testHeartRate(double bpm, double fpsNominal) {
    vitals::SignalProcessor sp;
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.15), jitter(0.0, 0.004), ibiJit(0.0, 0.025);
    double t = 0.0, nextBeat = 0.0, phase = 0.0;
    std::vector<double> beats;
    for (double b = 0; b < 30.0;) { beats.push_back(b); b += 60.0 / bpm + ibiJit(rng); }
    for (int i = 0; i < static_cast<int>(25 * fpsNominal); ++i) {
        t += 1.0 / fpsNominal + jitter(rng);
        double pulse = 0.0;
        for (double b : beats) pulse += std::exp(-(t - b) * (t - b) / (2 * 0.07 * 0.07));
        const double v = 120.0 + 0.8 * pulse + 3.0 * std::sin(2 * kPi * 0.04 * t) + noise(rng);
        sp.push(v, t);
        sp.process();
    }
    (void)nextBeat; (void)phase;
    const auto& m = sp.metrics();
    std::printf("  HR %.0f bpm @ %.0f fps -> %.1f bpm (inst %.1f) conf %.2f snr %.1f dB rmssd %.1f ms "
                "ibi %.0f ms beats %d fs %.2f\n",
                bpm, fpsNominal, m.bpm, m.instantBpm, m.confidence, m.snrDb, m.rmssdMs, m.meanIbiMs,
                m.beatCount, m.sampleRate);
    EXPECT(m.valid, "metrics not valid");
    EXPECT(std::abs(m.bpm - bpm) < 3.0, "bpm %.1f vs %.1f", m.bpm, bpm);
    EXPECT(m.confidence > 0.5, "confidence %.2f", m.confidence);
    EXPECT(m.rmssdMs > 5.0 && m.rmssdMs < 120.0, "rmssd %.1f", m.rmssdMs);
    EXPECT(std::abs(m.meanIbiMs - 60000.0 / bpm) < 60.0, "ibi %.1f", m.meanIbiMs);
}

void testPauseResume() {
    vitals::SignalProcessor sp;
    double t = 0.0;
    for (int i = 0; i < 300; ++i, t += 1.0 / 30) sp.push(100 + std::sin(2 * kPi * 1.0 * t), t);
    sp.pause(t);
    EXPECT(sp.process().paused, "should be paused");
    t += 1.0;  // short gap: buffer kept
    sp.push(140 + std::sin(2 * kPi * t), t);
    sp.process();
    EXPECT(sp.metrics().valid && std::abs(sp.metrics().bpm - 60.0) < 4.0, "short gap broke estimate (%.1f)",
           sp.metrics().bpm);
    sp.pause(t);
    t += 5.0;  // long gap: buffer flushed
    sp.push(100, t);
    sp.process();
    EXPECT(!sp.metrics().valid && sp.metrics().bufferFill < 0.05, "long gap should reset buffer");
}

cv::Mat eyeImage(double openness, int gazeDx = 0, double gain = 1.0) {
    cv::Mat img(480, 640, CV_8UC1, cv::Scalar(170));  // skin
    for (cv::Point c : {cv::Point(220, 240), cv::Point(420, 240)}) {
        cv::line(img, c + cv::Point(-36, -34), c + cv::Point(36, -38), cv::Scalar(80), 4, cv::LINE_AA);  // eyebrow
        const int a = 34, b = static_cast<int>(16 * openness);
        if (b >= 2) {
            cv::ellipse(img, c, {a, b}, 0, 0, 360, cv::Scalar(235), cv::FILLED, cv::LINE_AA);  // sclera
            cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
            cv::ellipse(mask, c, {a, b}, 0, 0, 360, cv::Scalar(255), cv::FILLED);
            cv::Mat iris = img.clone();
            cv::circle(iris, c + cv::Point(gazeDx, 0), 13, cv::Scalar(50), cv::FILLED, cv::LINE_AA);
            iris.copyTo(img, mask);
            cv::ellipse(img, c, {a, b}, 0, 180, 360, cv::Scalar(70), 2, cv::LINE_AA);       // upper lid
        } else {
            cv::line(img, c - cv::Point(a, 0), c + cv::Point(a, 0), cv::Scalar(90), 2, cv::LINE_AA);
        }
    }
    if (gain != 1.0) img.convertTo(img, CV_8UC1, gain);
    return img;
}

void testOcular() {
    vitals::OcularAnalyzer oa;
    const cv::Rect le(180, 215, 80, 50), re(380, 215, 80, 50);
    std::mt19937 rng(7);
    std::normal_distribution<double> n(0.0, 2.0);
    double t = 0.0;
    int blinksIn = 0, incompleteIn = 0;
    for (int i = 0; i < 30 * 40; ++i, t += 1.0 / 30) {
        double open = 1.0;
        const int k = i % 90;  // a blink every 3 s
        if (i > 150 && k < 6) {
            const bool partial = (i / 90) % 4 == 0;
            const double depth = partial ? 0.5 : 1.0;
            open = 1.0 - depth * std::sin(kPi * (k + 0.5) / 6.0);
            if (k == 0) { ++blinksIn; if (partial) ++incompleteIn; }
        }
        if (i > 900 && i < 960) open = 0.0;  // 2 s closure (micro-sleep)
        cv::Mat img = eyeImage(open);
        cv::Mat noise(img.size(), CV_16SC1);
        cv::randn(noise, 0, 2);
        cv::Mat img16;
        img.convertTo(img16, CV_16SC1);
        img16 += noise;
        img16.convertTo(img, CV_8UC1);
        oa.update(img, le, re, t);
    }
    const auto& m = oa.metrics();
    std::printf("  blinks in %d (incomplete %d) -> detected %d (incomplete %d) rate %.1f/min perclos %.1f%% "
                "long %d\n",
                blinksIn, incompleteIn, m.totalBlinks, m.incompleteBlinks, m.blinkRate, m.perclos * 100.0,
                m.longClosures);
    EXPECT(m.valid, "ocular not valid");
    EXPECT(std::abs(m.totalBlinks - blinksIn) <= 2, "blinks %d vs %d", m.totalBlinks, blinksIn);
    EXPECT(m.incompleteBlinks >= incompleteIn - 1 && m.incompleteBlinks <= incompleteIn + 2,
           "incomplete %d vs %d", m.incompleteBlinks, incompleteIn);
    EXPECT(m.longClosures == 1, "long closures %d", m.longClosures);
    EXPECT(m.perclos > 0.02 && m.perclos < 0.12, "perclos %.3f", m.perclos);
}

/// Confidence must stay inside [0, 1] and fall monotonically with noise.
void testConfidence() {
    double prev = 2.0;
    for (double sigma : {0.1, 0.5, 1.5, 4.0, 20.0}) {
        vitals::SignalProcessor sp;
        std::mt19937 rng(11);
        std::normal_distribution<double> noise(0.0, sigma);
        double t = 0.0;
        double minC = 9, maxC = -9;
        for (int i = 0; i < 600; ++i, t += 1.0 / 30) {
            sp.push(120.0 + 0.8 * std::sin(2 * kPi * 1.2 * t) + noise(rng), t);
            const auto& m = sp.process();
            if (m.valid) { minC = std::min(minC, m.confidence); maxC = std::max(maxC, m.confidence); }
        }
        const auto& m = sp.metrics();
        std::printf("  noise sigma %5.1f -> conf %5.1f%% (range %.2f..%.2f) snr %.1f dB\n", sigma,
                    m.confidence * 100.0, minC, maxC, m.snrDb);
        EXPECT(minC >= 0.0 && maxC <= 1.0, "confidence out of [0,1]: %.3f..%.3f", minC, maxC);
        EXPECT(m.confidence <= prev + 1e-9, "confidence not monotonic (%.3f after %.3f)", m.confidence, prev);
        prev = m.confidence;
        if (sigma <= 0.5) EXPECT(m.confidence > 0.9, "clean signal should be ~100%% (%.2f)", m.confidence);
        if (sigma >= 20.0) EXPECT(m.confidence < 0.5, "noise-only should be low (%.2f)", m.confidence);
    }
}

/// Breathing modulates the baseline at 0.25 Hz (15 BrPM) under pulse + drift + noise.
void testRespiration(double brpm, double fps) {
    vitals::SignalProcessor sp;
    std::mt19937 rng(5);
    std::normal_distribution<double> noise(0.0, 0.25);
    double t = 0.0;
    for (int i = 0; i < static_cast<int>(45 * fps); ++i, t += 1.0 / fps) {
        const double v = 120.0 + 0.6 * std::sin(2 * kPi * (brpm / 60.0) * t + 0.7)      // breathing
                         + 0.5 * std::sin(2 * kPi * 1.2 * t) + 2.0 * std::sin(2 * kPi * 0.03 * t)  // pulse, drift
                         + noise(rng);
        sp.push(v, t);
        sp.process();
    }
    const auto& m = sp.metrics();
    std::printf("  resp %4.1f BrPM @ %.0f fps -> %.1f BrPM conf %.0f%% valid %d\n", brpm, fps, m.respRate,
                m.respConfidence * 100.0, m.respValid ? 1 : 0);
    EXPECT(m.respValid, "respiration not valid");
    EXPECT(std::abs(m.respRate - brpm) < 2.0, "resp %.1f vs %.1f", m.respRate, brpm);
    EXPECT(m.respConfidence >= 0.0 && m.respConfidence <= 1.0, "resp conf out of range");
}

void testRespirationNoiseOnly() {
    vitals::SignalProcessor sp;
    std::mt19937 rng(9);
    std::normal_distribution<double> noise(0.0, 1.0);
    double t = 0.0;
    int validFrames = 0, total = 0;
    for (int i = 0; i < 30 * 90; ++i, t += 1.0 / 30) {
        sp.push(120.0 + noise(rng), t);
        const auto& m = sp.process();
        if (t > 25.0) { ++total; if (m.respValid) ++validFrames; }
    }
    std::printf("  resp noise-only: reported valid in %d / %d frames, conf %.0f%%\n", validFrames, total,
                sp.metrics().respConfidence * 100.0);
    EXPECT(validFrames < total / 4, "noise reported as respiration in %d / %d frames", validFrames, total);
}

void testRespirationNotReadyEarly() {
    vitals::SignalProcessor sp;
    double t = 0.0;
    for (int i = 0; i < 300; ++i, t += 1.0 / 30) { sp.push(100 + std::sin(2 * kPi * 0.25 * t), t); sp.process(); }
    EXPECT(!sp.metrics().respValid && sp.metrics().respProgress < 1.0, "resp must wait for %d s of data", 15);
}

/// Harder ocular scenario: exposure drift, gaze shifts, 15 / 30 / 60 FPS, blink-free start.
void testOcularRobust(double fps) {
    vitals::OcularAnalyzer oa;
    const cv::Rect le(180, 215, 80, 50), re(380, 215, 80, 50);
    std::mt19937 rng(21);
    const int frames = static_cast<int>(60 * fps);
    int blinksIn = 0;
    int lastBlinkStart = -1000;
    double t = 0.0;
    for (int i = 0; i < frames; ++i, t += 1.0 / fps) {
        double open = 1.0;
        // A 180 ms blink every ~3.3 s starting at t = 5 s (first 5 s: no blink at all).
        const double ts = t - 5.0;
        const double period = 3.3;
        if (ts >= 0.0) {
            const double ph = std::fmod(ts, period);
            if (ph < 0.18) {
                open = 1.0 - std::sin(kPi * ph / 0.18);
                if (i - lastBlinkStart > static_cast<int>(fps)) { ++blinksIn; lastBlinkStart = i; }
            }
        }
        const double gain = 1.0 + 0.15 * std::sin(2 * kPi * 0.07 * t);             // exposure drift
        const int gaze = static_cast<int>(8 * std::sin(2 * kPi * 0.11 * t));        // gaze wander
        cv::Mat img = eyeImage(open, gaze, gain), noise(img.size(), CV_16SC1), i16;
        cv::randn(noise, 0, 3);
        img.convertTo(i16, CV_16SC1); i16 += noise; i16.convertTo(img, CV_8UC1);
        oa.update(img, le, re, t);
    }
    const auto& m = oa.metrics();
    std::printf("  [%2.0f fps] blinks in %d -> detected %d  rate %.1f/min  perclos %.1f%%  ear %.2f mean %.2f thr %.2f"
                "  long %d\n", fps, blinksIn, m.totalBlinks, m.blinkRate, m.perclos * 100.0, m.ear, m.meanEar,
                m.threshold, m.longClosures);
    EXPECT(m.valid, "ocular not valid");
    EXPECT(m.framesProcessed > frames - 5, "frames reached analyzer: %ld / %d", m.framesProcessed, frames);
    EXPECT(m.meanEar > 0.15 && m.meanEar < 0.5, "mean EAR %.3f not in the plausible open-eye range", m.meanEar);
    EXPECT(std::abs(m.totalBlinks - blinksIn) <= 2, "blinks %d vs %d", m.totalBlinks, blinksIn);
    EXPECT(m.blinkRate > 10.0 && m.blinkRate < 25.0, "blink rate %.1f", m.blinkRate);
    EXPECT(m.perclos > 0.0 && m.perclos < 0.10, "perclos %.3f", m.perclos);
}

/// Eyes shut for most of the window must produce a high PERCLOS (drowsy) and long closures.
void testOcularDrowsy() {
    vitals::OcularAnalyzer oa;
    const cv::Rect le(180, 215, 80, 50), re(380, 215, 80, 50);
    double t = 0.0;
    for (int i = 0; i < 30 * 60; ++i, t += 1.0 / 30) {
        // 6 s open, then cycles of 2 s closed / 3 s open.
        const double ts = t - 6.0;
        const bool closed = ts > 0.0 && std::fmod(ts, 5.0) < 2.0;
        oa.update(eyeImage(closed ? 0.0 : 1.0), le, re, t);
    }
    const auto& m = oa.metrics();
    std::printf("  drowsy: perclos %.1f%% long closures %d blinks %d\n", m.perclos * 100.0, m.longClosures,
                m.totalBlinks);
    EXPECT(m.perclos > 0.25 && m.perclos < 0.45, "perclos %.3f (expected ≈ 0.36)", m.perclos);
    EXPECT(m.longClosures >= 8, "long closures %d", m.longClosures);
}

void testOcularPauseGap() {
    vitals::OcularAnalyzer oa;
    const cv::Rect le(180, 215, 80, 50), re(380, 215, 80, 50);
    double t = 0.0;
    for (int i = 0; i < 150; ++i, t += 1.0 / 30) oa.update(eyeImage(1.0), le, re, t);
    const long before = oa.metrics().framesProcessed;
    oa.pause(t);
    t += 8.0;  // face lost for 8 s with the eyes "closed" — must be ignored entirely
    for (int i = 0; i < 90; ++i, t += 1.0 / 30) oa.update(eyeImage(1.0), le, re, t);
    const auto& m = oa.metrics();
    EXPECT(m.framesProcessed == before + 90, "frame counting after pause");
    EXPECT(m.totalBlinks == 0 && m.longClosures == 0, "a tracking gap must not create blink events");
}

/// Webcam-like colour signal: weak pulse (0.3 %) along the skin pulse vector, buried under a
/// stronger *common-mode* illumination flicker at 60 BPM, auto-exposure steps and a large
/// drift just below the band (0.6 Hz), sampled at a jittery ~19.9 FPS with dropped frames.
struct WebcamSim {
    double trueBpm;
    double illumHz = 1.0;      // 60 BPM illumination flicker (in band!)
    double illumAmp = 0.02;    // 2 % — ~9× the green pulse amplitude
    double driftAmp = 0.03;    // 3 % baseline swing at 0.6 Hz (below the band)
    bool aeSteps = true;       // ±1.5 % auto-exposure square wave (period 7 s)
};

double runWebcam(const WebcamSim& sim, vitals::PulseMethod method, double fpsNominal,
                 double* conf = nullptr, int seconds = 30) {
    vitals::SignalConfig cfg;
    cfg.method = method;
    vitals::SignalProcessor sp(cfg);
    std::mt19937 rng(17);
    std::normal_distribution<double> jitter(0.0, 0.006), noise(0.0, 0.25);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const double dc[3] = {110.0, 130.0, 180.0};          // B, G, R skin tone
    const double pv[3] = {0.53, 0.77, 0.33};             // pulse vector (B, G, R), de Haan 2013
    const double a = 0.003;
    double t = 0.0;
    while (t < seconds) {
        t += 1.0 / fpsNominal + jitter(rng);
        if (u(rng) < 0.05) continue;                     // dropped frame
        const double ph = 2 * kPi * sim.trueBpm / 60.0 * t;
        const double pulse = std::sin(ph) + 0.4 * std::sin(2 * ph + 0.5);
        const double exposure = 1.0 + (sim.aeSteps && std::fmod(t, 7.0) >= 3.5 ? 0.03 : 0.0);  // AE steps
        const double illum = (1.0 + sim.illumAmp * std::sin(2 * kPi * sim.illumHz * t) +
                              sim.driftAmp * std::sin(2 * kPi * 0.6 * t)) * exposure;
        cv::Vec3d c;
        for (int k = 0; k < 3; ++k) c[k] = dc[k] * (1.0 + a * pv[k] * pulse) * illum + noise(rng);
        sp.push(c, t);
        sp.process();
    }
    if (conf) *conf = sp.metrics().confidence;
    return sp.metrics().valid ? sp.metrics().bpm : 0.0;
}

void testPosVsIllumination() {
    for (double bpm : {58.0, 84.0, 110.0}) {
        WebcamSim sim{bpm};
        double cPos = 0, cGreen = 0;
        const double pos = runWebcam(sim, vitals::PulseMethod::POS, 19.9, &cPos);
        const double green = runWebcam(sim, vitals::PulseMethod::Green, 19.9, &cGreen);
        std::printf("  true %5.1f bpm @ ~19.9 fps (flicker 60 bpm, AE steps, drift): POS %5.1f (conf %3.0f%%)"
                    "   green %5.1f (conf %3.0f%%)\n", bpm, pos, cPos * 100, green, cGreen * 100);
        EXPECT(std::abs(pos - bpm) < 3.0, "POS %.1f vs %.1f", pos, bpm);
        EXPECT(pos > 47.0, "POS latched at the lower band edge (%.1f)", pos);
    }
    // The case motivating POS: green-only follows the 60 BPM illumination flicker.
    WebcamSim sim{84.0};
    const double green = runWebcam(sim, vitals::PulseMethod::Green, 19.9);
    EXPECT(std::abs(green - 84.0) > 5.0, "expected green-only to fail here (%.1f) - test is not discriminative", green);
}

void testVariableFps() {
    for (double fps : {15.0, 19.9, 24.0, 30.0}) {
        WebcamSim sim{76.0, 1.0, 0.0, 0.0, false};
        const double pos = runWebcam(sim, vitals::PulseMethod::POS, fps);
        std::printf("  76 bpm @ %4.1f fps -> %.1f bpm\n", fps, pos);
        EXPECT(std::abs(pos - 76.0) < 2.0, "fps %.1f: %.1f bpm", fps, pos);
    }
}

void testNoEdgeLatch() {
    // Huge baseline power just below the band (0.6 Hz, 10 %) must not drag the estimate to 45 BPM.
    WebcamSim sim{92.0, 1.0, 0.0, 0.10, false};
    const double green = runWebcam(sim, vitals::PulseMethod::Green, 30.0);
    std::printf("  strong 0.6 Hz baseline, pulse 92 bpm -> green %.1f bpm\n", green);
    EXPECT(std::abs(green - 92.0) < 3.0, "latched: %.1f", green);
}

void testTransientRejection() {
    // Steady 70 BPM; a 2 s burst of strong 105 BPM interference must not yank the reading.
    vitals::SignalProcessor sp;
    std::mt19937 rng(23);
    std::normal_distribution<double> noise(0.0, 0.1);
    double t = 0.0, worst = 0.0;
    for (int i = 0; i < 30 * 40; ++i, t += 1.0 / 30) {
        double v = 120.0 + 0.6 * std::sin(2 * kPi * (70.0 / 60.0) * t) + noise(rng);
        if (t > 25.0 && t < 27.0) v += 1.5 * std::sin(2 * kPi * 1.75 * t);
        sp.push(v, t);
        const auto& m = sp.process();
        if (t > 15.0 && m.valid) worst = std::max(worst, std::abs(m.bpm - 70.0));
    }
    std::printf("  2 s interference burst: max deviation of reported HR %.1f bpm\n", worst);
    EXPECT(worst < 6.0, "reported HR jumped by %.1f bpm", worst);
}
}  // namespace

int main() {
    std::printf("[SignalProcessor]\n");
    testHeartRate(52, 30);
    testHeartRate(72, 30);
    testHeartRate(95, 25);
    testHeartRate(130, 30);
    testPauseResume();
    std::printf("[POS / webcam conditions]\n");
    testPosVsIllumination();
    testVariableFps();
    testNoEdgeLatch();
    testTransientRejection();
    std::printf("[Confidence]\n");
    testConfidence();
    std::printf("[Respiration]\n");
    testRespiration(15, 30);
    testRespiration(10, 30);
    testRespiration(22, 25);
    testRespirationNotReadyEarly();
    testRespirationNoiseOnly();
    std::printf("[OcularAnalyzer]\n");
    testOcular();
    testOcularRobust(15);
    testOcularRobust(30);
    testOcularRobust(60);
    testOcularDrowsy();
    testOcularPauseGap();
    std::printf(gFailures ? "%d FAILURE(S)\n" : "ALL TESTS PASSED\n", gFailures);
    return gFailures ? 1 : 0;
}

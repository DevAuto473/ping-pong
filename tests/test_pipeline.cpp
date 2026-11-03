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

cv::Mat eyeImage(double openness) {
    cv::Mat img(480, 640, CV_8UC1, cv::Scalar(170));  // skin
    for (cv::Point c : {cv::Point(220, 240), cv::Point(420, 240)}) {
        const int a = 34, b = static_cast<int>(16 * openness);
        if (b >= 2) {
            cv::ellipse(img, c, {a, b}, 0, 0, 360, cv::Scalar(235), cv::FILLED, cv::LINE_AA);  // sclera
            cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
            cv::ellipse(mask, c, {a, b}, 0, 0, 360, cv::Scalar(255), cv::FILLED);
            cv::Mat iris = img.clone();
            cv::circle(iris, c, 13, cv::Scalar(50), cv::FILLED, cv::LINE_AA);
            iris.copyTo(img, mask);
            cv::ellipse(img, c, {a, b}, 0, 180, 360, cv::Scalar(70), 2, cv::LINE_AA);       // upper lid
        } else {
            cv::line(img, c - cv::Point(a, 0), c + cv::Point(a, 0), cv::Scalar(90), 2, cv::LINE_AA);
        }
    }
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
}  // namespace

int main() {
    std::printf("[SignalProcessor]\n");
    testHeartRate(52, 30);
    testHeartRate(72, 30);
    testHeartRate(95, 25);
    testHeartRate(130, 30);
    testPauseResume();
    std::printf("[OcularAnalyzer]\n");
    testOcular();
    std::printf(gFailures ? "%d FAILURE(S)\n" : "ALL TESTS PASSED\n", gFailures);
    return gFailures ? 1 : 0;
}

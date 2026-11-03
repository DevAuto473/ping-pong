// ============================================================================
//  vitals_monitor — contactless rPPG heart rate + ocular fatigue monitor.
// ============================================================================
#include "FaceTracker.hpp"
#include "OcularAnalyzer.hpp"
#include "SignalProcessor.hpp"
#include "TelemetryUI.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

#ifndef VITALS_DEFAULT_MODEL
#define VITALS_DEFAULT_MODEL "models/face_detection_yunet_2023mar.onnx"
#endif

namespace {

std::atomic<bool> gStop{false};
void onSignal(int) { gStop = true; }

using Clock = std::chrono::steady_clock;

double secondsSince(const Clock::time_point& t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

const char* kKeys =
    "{help h usage ? |      | print this message                                   }"
    "{camera c       | 0    | camera index                                         }"
    "{video v        |      | video file instead of a camera (offline analysis)    }"
    "{model m        | " VITALS_DEFAULT_MODEL " | path to YuNet ONNX model         }"
    "{width          | 1280 | requested capture width                              }"
    "{height         | 720  | requested capture height                             }"
    "{fps            | 30   | requested capture frame rate                         }"
    "{buffer         | 512  | rPPG buffer length in samples (256-512)              }"
    "{mirror         | true | mirror the preview (selfie view)                     }"
    "{headless       | false| no window; print telemetry to stdout                 }"
    "{log            | 0    | print telemetry every N seconds (0 = off)            }"
    "{max-motion     | 0.08 | face motion (x IOD per frame) that pauses sampling   }";

/// Camera / video source with automatic reconnection.
class FrameSource {
public:
    FrameSource(int camIndex, std::string video, int w, int h, double fps)
        : cam_(camIndex), video_(std::move(video)), w_(w), h_(h), fps_(fps) {}

    bool isFile() const { return !video_.empty(); }

    bool open() {
        cap_.release();
        if (isFile()) {
            cap_.open(video_);
        } else {
            cap_.open(cam_, cv::CAP_ANY);
            if (cap_.isOpened()) {
                cap_.set(cv::CAP_PROP_FRAME_WIDTH, w_);
                cap_.set(cv::CAP_PROP_FRAME_HEIGHT, h_);
                cap_.set(cv::CAP_PROP_FPS, fps_);
                cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);  // low latency (ignored by some backends)
            }
        }
        failures_ = 0;
        return cap_.isOpened();
    }

    /// Returns false for an empty/failed read. `fatal` is set on end-of-file.
    bool read(cv::Mat& frame, bool& fatal) {
        fatal = false;
        if (!cap_.isOpened()) {
            ++failures_;
            return false;
        }
        bool ok = false;
        try {
            ok = cap_.read(frame) && !frame.empty();
        } catch (const cv::Exception& e) {
            std::cerr << "[capture] exception: " << e.what() << '\n';
            ok = false;
        }
        if (ok) {
            failures_ = 0;
            return true;
        }
        if (isFile()) fatal = true;          // end of stream
        else ++failures_;
        return false;
    }

    /// Timestamp of the last frame in seconds (media clock for files).
    double timestamp(const Clock::time_point& t0, long frameIndex) const {
        if (isFile()) {
            const double ms = cap_.get(cv::CAP_PROP_POS_MSEC);
            if (ms > 0.0) return ms / 1000.0;
            const double f = cap_.get(cv::CAP_PROP_FPS);
            return frameIndex / (f > 1.0 ? f : 30.0);
        }
        return secondsSince(t0);
    }

    int consecutiveFailures() const { return failures_; }
    void release() { cap_.release(); }

private:
    cv::VideoCapture cap_;
    int cam_;
    std::string video_;
    int w_, h_;
    double fps_;
    int failures_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv, kKeys);
    parser.about("Contactless rPPG heart-rate & ocular fatigue monitor (OpenCV " CV_VERSION ")");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    const int camIndex = parser.get<int>("camera");
    const std::string video = parser.has("video") ? parser.get<std::string>("video") : "";
    const std::string modelPath = parser.get<std::string>("model");
    const bool mirror = parser.get<bool>("mirror");
    const bool headless = parser.get<bool>("headless");
    const double logEvery = parser.get<double>("log");
    const double maxMotion = parser.get<double>("max-motion");
    const int bufferLen = std::max(256, std::min(512, parser.get<int>("buffer")));
    if (!parser.check()) {
        parser.printErrors();
        return 2;
    }

    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    try {
        vitals::FaceTrackerConfig tcfg;
        tcfg.modelPath = modelPath;
        vitals::FaceTracker tracker(tcfg);

        vitals::SignalConfig scfg;
        scfg.bufferSize = static_cast<std::size_t>(bufferLen);
        vitals::SignalProcessor signal(scfg);
        vitals::OcularAnalyzer ocular;
        vitals::TelemetryUI ui;

        FrameSource source(camIndex, video, parser.get<int>("width"), parser.get<int>("height"),
                           parser.get<double>("fps"));
        if (!source.open()) {
            std::cerr << "[capture] cannot open " << (source.isFile() ? video : "camera " + std::to_string(camIndex))
                      << (source.isFile() ? "" : " — will keep retrying") << '\n';
            if (source.isFile()) return 1;
        }

        const std::string window = "Vitals Monitor";
        if (!headless) cv::namedWindow(window, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);

        const auto t0 = Clock::now();
        auto lastFrameTime = Clock::now();
        auto lastReconnect = Clock::now();
        double fps = 0.0, lastLog = 0.0;
        double reconnectDelay = 0.5;
        long frameIndex = 0;
        cv::Mat frame, gray;
        cv::Size lastSize(1280, 720);

        while (!gStop) {
            bool eof = false;
            if (!source.read(frame, eof)) {
                if (eof) break;
                // ---- Camera unavailable: pause everything, retry with backoff.
                signal.pause(secondsSince(t0));
                ocular.pause(secondsSince(t0));
                tracker.reset();
                // Tolerate a few transient read failures before re-opening the device.
                if (source.consecutiveFailures() > 5) {
                    if (secondsSince(lastReconnect) > reconnectDelay) {
                        std::cerr << "[capture] camera lost — reconnecting (retry in " << reconnectDelay << " s)\n";
                        lastReconnect = Clock::now();
                        if (source.open()) {
                            std::cerr << "[capture] camera reconnected\n";
                            reconnectDelay = 0.5;
                        } else {
                            reconnectDelay = std::min(reconnectDelay * 2.0, 8.0);
                        }
                    }
                }
                if (!headless) {
                    cv::imshow(window, ui.renderNoSignal(lastSize, "Check the USB connection. Retrying...", 0.0));
                    const int key = cv::waitKey(30) & 0xFF;
                    if (key == 27 || key == 'q' || key == 'Q') break;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                }
                continue;
            }
            ++frameIndex;
            if (mirror && !source.isFile()) cv::flip(frame, frame, 1);
            lastSize = frame.size();
            const double t = source.timestamp(t0, frameIndex);

            // ---- FPS (EMA of instantaneous rate, wall clock).
            const auto now = Clock::now();
            const double dt = std::chrono::duration<double>(now - lastFrameTime).count();
            lastFrameTime = now;
            if (dt > 0.0) fps = fps <= 0.0 ? 1.0 / dt : 0.9 * fps + 0.1 / dt;

            // ---- Tracking + feature extraction.
            const vitals::FaceObservation face = tracker.update(frame);
            vitals::SystemState state = vitals::SystemState::Acquiring;
            std::string status;
            if (face.valid) {
                const bool stable = face.motion <= maxMotion;
                if (stable) {
                    const cv::Scalar m = cv::mean(frame(face.forehead));
                    signal.push(m[1], t);  // green channel (BGR index 1)
                } else {
                    // Gross motion corrupts the rPPG trace: hold the buffer briefly.
                    signal.pause(t);
                }
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                ocular.update(gray, face.leftEye, face.rightEye, t);
            } else {
                signal.pause(t);
                ocular.pause(t);
                state = vitals::SystemState::FaceLost;
                status = "NO FACE DETECTED - BUFFERS PAUSED";
            }
            const vitals::SignalMetrics& sm = signal.process();
            if (face.valid) state = sm.valid ? vitals::SystemState::Running : vitals::SystemState::Acquiring;

            const vitals::OcularMetrics& om = ocular.metrics();
            if (logEvery > 0.0 && t - lastLog >= logEvery) {
                lastLog = t;
                std::printf("t=%6.1fs fps=%5.1f face=%d hr=%5.1f bpm conf=%3.0f%% rmssd=%5.1fms "
                            "resp=%4.1f brpm(%3.0f%%) | eyes: frames=%ld ear=%.3f mean=%.3f thr=%.3f "
                            "blinks=%d rate=%4.1f/min incompl=%d long=%d perclos=%4.1f%%\n",
                            t, fps, face.valid ? 1 : 0, sm.valid ? sm.bpm : 0.0, sm.confidence * 100.0,
                            sm.rmssdMs, sm.respValid ? sm.respRate : 0.0, sm.respConfidence * 100.0,
                            om.framesProcessed, om.ear, om.meanEar, om.threshold, om.totalBlinks,
                            om.blinkRate, om.incompleteBlinks, om.longClosures, om.perclos * 100.0);
                std::fflush(stdout);
            }
            if (headless) continue;

            // ---- Render.
            vitals::UiFrameState ui_state;
            ui_state.state = state;
            ui_state.fps = fps;
            ui_state.sessionSeconds = secondsSince(t0);
            ui_state.face = &face;
            ui_state.signal = &sm;
            ui_state.ocular = &om;
            ui_state.statusMessage = status;
            ui.render(frame, ui_state);
            cv::imshow(window, frame);

            const int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q' || key == 'Q') break;
            if (key == 'r' || key == 'R') {
                signal.reset();
                ocular.reset();
                tracker.reset();
            }
            // Window closed with the [x] button.
            if (cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1.0) break;
        }

        source.release();
        if (!headless) cv::destroyAllWindows();
        std::cout << "Session ended cleanly.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[fatal] " << e.what() << '\n';
        return 1;
    }
}

// ============================================================================
//  vitals_monitor — contactless rPPG heart rate + ocular fatigue monitor.
// ============================================================================
#include "ClinicalAnalytics.hpp"
#include "FaceTracker.hpp"
#include "OcularAnalyzer.hpp"
#include "QualityAdvisor.hpp"
#include "SignalProcessor.hpp"
#include "TelemetryUI.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <exception>
#include <fstream>
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
    "{method         | pos  | pulse extraction: pos (illumination robust) or green }"
    "{save           |      | write the HUD preview to a video file (e.g. out.mp4)  }"
    "{log            | 0    | print telemetry every N seconds (0 = off)            }"
    "{max-motion     | 0.08 | face motion (x IOD per frame) that pauses sampling   }"
    "{max-rot-speed  | 30   | head rotation speed (deg/s) that pauses sampling     }"
    "{spo2-a         | 110  | SpO2 = A - B*R calibration intercept (uncalibrated)  }"
    "{spo2-b         | 25   | SpO2 = A - B*R calibration slope (uncalibrated)      }"
    "{report         | true | print the clinical telemetry report on exit         }"
    "{report-file    |      | also write the text report to this file              }"
    "{json           |      | write a machine-readable JSON session summary        }"
    "{csv            |      | write a time series (CSV) of all measurements        }"
    "{csv-interval   | 1.0  | seconds between CSV rows                             }"
    "{tips           | true | show acquisition tips (move closer, add light, ...)   }"
    "{drowsy-alert   | true | flash an alert on micro-sleeps / high PERCLOS        }";

/// Local time stamp for file names, e.g. 20261105_143012.
std::string timeStamp() {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return buf;
}

bool writeTextFile(const std::string& path, const std::string& content, const char* what) {
    std::ofstream f(path, std::ios::binary);
    if (f) f << content;
    if (!f) {
        std::cerr << "[" << what << "] cannot write '" << path << "'\n";
        return false;
    }
    std::cout << "[" << what << "] saved to " << path << '\n';
    return true;
}

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
            // Many backends report 0 ms for the first frame: that is a valid time stamp.
            if (ms > 0.0 || (ms == 0.0 && frameIndex <= 1)) return ms / 1000.0;
            const double f = cap_.get(cv::CAP_PROP_FPS);
            return frameIndex / (f > 1.0 ? f : 30.0);
        }
        return secondsSince(t0);
    }

    double fileFps() const {
        const double f = cap_.get(cv::CAP_PROP_FPS);
        return f > 1.0 ? f : 30.0;
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
    const double maxRotSpeed = parser.get<double>("max-rot-speed");
    const bool printReport = parser.get<bool>("report");
    const std::string savePath = parser.has("save") ? parser.get<std::string>("save") : "";
    const int bufferArg = parser.get<int>("buffer");
    const int bufferLen = std::max(256, std::min(512, bufferArg));
    const std::string method = parser.get<std::string>("method");
    const std::string reportFile = parser.has("report-file") ? parser.get<std::string>("report-file") : "";
    const std::string jsonPath = parser.has("json") ? parser.get<std::string>("json") : "";
    const std::string csvPath = parser.has("csv") ? parser.get<std::string>("csv") : "";
    const double csvInterval = parser.get<double>("csv-interval");
    const bool showTips = parser.get<bool>("tips");
    const bool drowsyAlert = parser.get<bool>("drowsy-alert");
    const int reqW = parser.get<int>("width"), reqH = parser.get<int>("height");
    const double reqFps = parser.get<double>("fps");
    if (!parser.check()) {
        parser.printErrors();
        return 2;
    }
    // ---- Argument validation: fail early with a clear message instead of misbehaving later.
    {
        std::string err;
        if (method != "pos" && method != "green") err = "--method must be 'pos' or 'green' (got '" + method + "')";
        else if (camIndex < 0) err = "--camera must be >= 0";
        else if (reqW <= 0 || reqH <= 0) err = "--width / --height must be positive";
        else if (!(reqFps > 0.0)) err = "--fps must be positive";
        else if (!(maxMotion > 0.0)) err = "--max-motion must be positive";
        else if (!(maxRotSpeed > 0.0)) err = "--max-rot-speed must be positive";
        else if (logEvery < 0.0) err = "--log must be >= 0";
        else if (!(csvInterval > 0.0)) err = "--csv-interval must be positive";
        if (!err.empty()) {
            std::cerr << "[args] " << err << "\n";
            return 2;
        }
        if (bufferArg != bufferLen)
            std::cerr << "[args] --buffer " << bufferArg << " clamped to " << bufferLen << " (valid 256-512)\n";
    }

    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    try {
        vitals::FaceTrackerConfig tcfg;
        tcfg.modelPath = modelPath;
        vitals::FaceTracker tracker(tcfg);

        vitals::SignalConfig scfg;
        scfg.bufferSize = static_cast<std::size_t>(bufferLen);
        scfg.method = method == "green" ? vitals::PulseMethod::Green : vitals::PulseMethod::POS;
        vitals::SignalProcessor signal(scfg);
        vitals::OcularAnalyzer ocular;
        vitals::ClinicalConfig ccfg;
        ccfg.spo2A = parser.get<double>("spo2-a");
        ccfg.spo2B = parser.get<double>("spo2-b");
        vitals::ClinicalAnalytics analytics(ccfg);
        vitals::TelemetryUI ui;
        vitals::QualityAdvisor advisor;
        vitals::DrowsinessMonitor drowsy;

        std::ofstream csv;
        if (!csvPath.empty()) {
            csv.open(csvPath, std::ios::binary);
            if (!csv) {
                std::cerr << "[csv] cannot create '" << csvPath << "'\n";
                return 1;
            }
            csv << vitals::ClinicalAnalytics::csvHeader() << '\n';
        }

        FrameSource source(camIndex, video, reqW, reqH, reqFps);
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
        cv::VideoWriter writer;  // opened lazily on the first rendered frame
        bool saveTried = false;
        cv::Size lastSize(1280, 720);
        double lastT = -1.0, lastCsv = -1e9;
        bool userPaused = false, hudVisible = true;
        std::array<cv::Rect, vitals::kRoiCount> prevRois{};

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
            double t = source.timestamp(t0, frameIndex);
            // Guard against non-monotonic clocks (seek glitches, broken container time stamps).
            if (lastT >= 0.0 && t <= lastT) t = lastT + 1.0 / (source.isFile() ? source.fileFps() : 30.0);
            lastT = t;

            // ---- FPS (EMA of instantaneous rate, wall clock).
            const auto now = Clock::now();
            const double dt = std::chrono::duration<double>(now - lastFrameTime).count();
            lastFrameTime = now;
            if (dt > 0.0) fps = fps <= 0.0 ? 1.0 / dt : 0.9 * fps + 0.1 / dt;

            // ---- Tracking + feature extraction.
            const vitals::FaceObservation face = tracker.update(frame, t);
            vitals::SystemState state = vitals::SystemState::Acquiring;
            std::string status;
            bool motionRejected = false;
            vitals::FrameSample sample;
            bool haveSample = false;
            if (userPaused) {
                signal.pause(t);
                ocular.pause(t);
                state = vitals::SystemState::UserPaused;
            } else if (face.valid) {
                // Translation and head-rotation speed both corrupt the rPPG trace:
                // freeze the buffers (they resume, spliced, once the head is still).
                motionRejected = face.motion > maxMotion || face.angularSpeedDps > maxRotSpeed;
                const std::array<cv::Rect, vitals::kRoiCount> rois{face.forehead, face.leftCheek, face.rightCheek};
                sample = vitals::SignalProcessor::sampleFrame(frame, rois, {face.backgroundLeft, face.backgroundRight});
                haveSample = true;
                // A relocated (lock-released) ROI sees a different pixel set: flag it so the
                // processor removes the colour step instead of treating it as signal.
                for (int k = 0; k < vitals::kRoiCount; ++k)
                    sample.roi[k].moved = prevRois[k].area() > 0 && rois[k] != prevRois[k];
                prevRois = rois;
                if (!motionRejected) signal.push(sample, t);
                else signal.pause(t);
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                ocular.update(gray, face.leftEye, face.rightEye, t);
            } else {
                prevRois = {};
                signal.pause(t);
                ocular.pause(t);
                state = vitals::SystemState::FaceLost;
                status = "NO FACE DETECTED - BUFFERS PAUSED";
            }
            const vitals::SignalMetrics& sm = signal.process();
            if (face.valid && !userPaused)
                state = sm.valid ? vitals::SystemState::Running : vitals::SystemState::Acquiring;

            const vitals::OcularMetrics& om = ocular.metrics();
            if (!userPaused) analytics.update(t, sm, om, face.valid, motionRejected);

            // ---- Acquisition tips + drowsiness alert (both debounced).
            vitals::AdviceInput ain;
            ain.face = &face;
            ain.sample = haveSample ? &sample : nullptr;
            ain.signal = &sm;
            ain.motionRejected = motionRejected;
            ain.fps = fps;
            ain.offline = source.isFile();
            const vitals::Advice advice = userPaused ? vitals::Advice::None : advisor.update(ain, t);
            const bool wasDrowsy = drowsy.active();
            const bool isDrowsy = drowsyAlert && !userPaused && drowsy.update(om, t);
            analytics.setDrowsinessEpisodes(drowsy.episodes());
            if (isDrowsy && !wasDrowsy) {
                std::cerr << "[alert] t=" << t << " s DROWSINESS: " << drowsy.reason() << (headless ? "\a" : "")
                          << '\n';
            }

            if (logEvery > 0.0 && t - lastLog >= logEvery) {
                lastLog = t;
                std::cout << analytics.statusLine(t, sm, om, motionRejected);
                if (advice != vitals::Advice::None) std::cout << " | tip: " << vitals::adviceText(advice);
                std::cout << std::endl;
            }
            if (csv.is_open() && t - lastCsv >= csvInterval) {
                lastCsv = t;
                csv << analytics.csvRow(t, sm, om, face.valid, motionRejected, vitals::adviceId(advice), isDrowsy)
                    << '\n';
            }
            if (headless && savePath.empty()) continue;

            // ---- Render.
            vitals::UiFrameState ui_state;
            ui_state.state = state;
            ui_state.fps = fps;
            ui_state.sessionSeconds = source.isFile() ? t : secondsSince(t0);
            ui_state.face = &face;
            ui_state.signal = &sm;
            ui_state.ocular = &om;
            ui_state.statusMessage = status;
            if (showTips && advice != vitals::Advice::None) ui_state.advice = vitals::adviceText(advice);
            if (isDrowsy) ui_state.alert = drowsy.reason();
            if (hudVisible) ui.render(frame, ui_state);
            if (!savePath.empty()) {
                if (!writer.isOpened() && !saveTried) {
                    saveTried = true;
                    const double outFps = source.isFile() ? source.fileFps() : std::max(10.0, std::min(fps, 60.0));
                    if (!writer.open(savePath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), outFps, frame.size())) {
                        std::cerr << "[save] cannot create '" << savePath << "' — continuing without saving\n";
                    } else {
                        std::cout << "[save] writing HUD preview to " << savePath << '\n';
                    }
                }
                if (writer.isOpened()) writer.write(frame);
            }
            if (headless) continue;
            cv::imshow(window, frame);

            const int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q' || key == 'Q') break;
            if (key == 'r' || key == 'R') {
                signal.reset();
                ocular.reset();
                tracker.reset();
                analytics.reset();
                advisor.reset();
                drowsy.reset();
                prevRois = {};
                std::cout << "[reset] measurements cleared\n";
            } else if (key == 'p' || key == 'P' || key == ' ') {
                userPaused = !userPaused;
                std::cout << (userPaused ? "[pause] measurement paused\n" : "[pause] measurement resumed\n");
            } else if (key == 'h' || key == 'H') {
                hudVisible = !hudVisible;
            } else if (key == 's' || key == 'S') {
                const std::string shot = "vitals_" + timeStamp() + ".png";
                if (cv::imwrite(shot, frame)) std::cout << "[snapshot] saved " << shot << '\n';
                else std::cerr << "[snapshot] cannot write " << shot << '\n';
            }
            // Window closed with the [x] button.
            if (cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1.0) break;
        }

        if (writer.isOpened()) {
            writer.release();
            std::cout << "[save] preview saved to " << savePath << '\n';
        }
        source.release();
        if (!headless) cv::destroyAllWindows();
        // ESC / 'q' / window close / end of video / Ctrl+C all end up here.
        if (csv.is_open()) {
            csv.close();
            std::cout << "[csv] time series saved to " << csvPath << '\n';
        }
        const std::string report = analytics.report();
        if (printReport) std::cout << report << std::flush;
        int rc = 0;
        if (!reportFile.empty() && !writeTextFile(reportFile, report, "report")) rc = 1;
        if (!jsonPath.empty() && !writeTextFile(jsonPath, analytics.reportJson(), "json")) rc = 1;
        std::cout << "Session ended cleanly.\n";
        return rc;
    } catch (const std::exception& e) {
        std::cerr << "[fatal] " << e.what() << '\n';
        return 1;
    }
}

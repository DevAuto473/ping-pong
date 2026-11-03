#pragma once
// ============================================================================
//  TelemetryUI — dark medical-style HUD composited directly onto the video.
// ============================================================================
#include "FaceTracker.hpp"
#include "OcularAnalyzer.hpp"
#include "SignalProcessor.hpp"

#include <opencv2/core.hpp>

#include <string>

namespace vitals {

enum class SystemState { Running, Acquiring, FaceLost, CameraLost };

struct UiFrameState {
    SystemState state = SystemState::Acquiring;
    double fps = 0.0;
    double sessionSeconds = 0.0;
    const FaceObservation* face = nullptr;
    const SignalMetrics* signal = nullptr;
    const OcularMetrics* ocular = nullptr;
    std::string statusMessage;
};

class TelemetryUI {
public:
    TelemetryUI() = default;

    /// Draws the complete HUD on `frame` (BGR, modified in place).
    void render(cv::Mat& frame, const UiFrameState& s) const;

    /// Full-screen placeholder used while the camera is unavailable.
    cv::Mat renderNoSignal(const cv::Size& size, const std::string& message,
                           double fps) const;

private:
    void drawRois(cv::Mat& frame, const FaceObservation& face) const;
    void drawHeader(cv::Mat& frame, const UiFrameState& s) const;
    void drawCards(cv::Mat& frame, const cv::Rect& panel, const UiFrameState& s) const;
    void drawWaveform(cv::Mat& frame, const cv::Rect& area, const SignalMetrics& m,
                      bool paused) const;
    void drawBanner(cv::Mat& frame, const std::string& text, const cv::Scalar& color) const;
};

}  // namespace vitals

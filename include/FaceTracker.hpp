#pragma once
// ============================================================================
//  FaceTracker — YuNet (cv::FaceDetectorYN) face + 5-landmark inference with
//  temporally stabilised forehead / ocular regions of interest.
// ============================================================================
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include <array>
#include <string>

namespace vitals {

/// Exponential Moving Average over an axis-aligned floating-point rectangle.
/// A large displacement (re-acquisition / fast head move) snaps the filter to
/// the new measurement instead of dragging the box slowly across the frame.
class RectEMA {
public:
    explicit RectEMA(float alpha = 0.25f, float snapRatio = 0.6f)
        : alpha_(alpha), snapRatio_(snapRatio) {}

    const cv::Rect2f& update(const cv::Rect2f& measurement);
    void reset() { initialised_ = false; }
    bool initialised() const { return initialised_; }
    const cv::Rect2f& value() const { return state_; }
    void setAlpha(float a) { alpha_ = a; }

private:
    float alpha_;
    float snapRatio_;
    bool initialised_ = false;
    cv::Rect2f state_;
};

struct FaceTrackerConfig {
    std::string modelPath;
    int detectWidth = 320;          ///< Inference width (frame is downscaled; aspect preserved).
    float scoreThreshold = 0.80f;
    float nmsThreshold = 0.30f;
    int topK = 50;
    float roiAlpha = 0.25f;         ///< EMA weight of new measurement for ROI boxes (0..1].
    float landmarkAlpha = 0.35f;    ///< EMA weight for landmark points.
    int maxMissedFrames = 6;        ///< Detector misses tolerated (coasting) before "lost".
    int backend = 0;                ///< cv::dnn::Backend
    int target = 0;                 ///< cv::dnn::Target
};

/// Per-frame tracking output. All rectangles are clipped to the frame.
struct FaceObservation {
    bool valid = false;             ///< A face is tracked (detected or coasting).
    bool detectedThisFrame = false; ///< false while coasting through a short miss.
    float score = 0.f;
    cv::Rect2f face;
    /// Image-space landmarks: 0 = right eye, 1 = left eye, 2 = nose tip,
    /// 3 = right mouth corner, 4 = left mouth corner (subject's perspective).
    std::array<cv::Point2f, 5> landmarks{};
    cv::Rect forehead;
    cv::Rect leftEye;               ///< Eye appearing on the image-left side.
    cv::Rect rightEye;              ///< Eye appearing on the image-right side.
    /// Malar (cheek) skin patches below each eye, lateral to the nose and above the mouth
    /// corners: strongly perfused, so they add rPPG signal to the forehead.
    cv::Rect leftCheek;
    cv::Rect rightCheek;
    float interOcular = 0.f;        ///< Pixels, stabilised.
    float rollDeg = 0.f;
    float motion = 0.f;             ///< Normalised per-frame face displacement (×inter-ocular).
    /// Approximate head pose from the 5 landmarks (degrees). Yaw from the nose offset along
    /// the eye axis, pitch relative to the session's own neutral nose/mouth geometry.
    float yawDeg = 0.f;
    float pitchDeg = 0.f;
    /// Max angular speed over yaw / pitch / roll (deg/s, smoothed); 0 without timestamps.
    float angularSpeedDps = 0.f;
    /// Non-face patches beside the head (ambient-light / screen-flicker reference).
    cv::Rect backgroundLeft;
    cv::Rect backgroundRight;
};

class FaceTracker {
public:
    /// An empty cfg.modelPath creates an "ingest-only" tracker (no network loaded).
    explicit FaceTracker(const FaceTrackerConfig& cfg);

    /// Runs detection on a BGR frame and returns the stabilised observation.
    /// `t` (seconds) enables head-pose velocity; pass < 0 if unknown.
    FaceObservation update(const cv::Mat& bgrFrame, double t = -1.0);

    /// Feeds externally computed detections in YuNet layout (N×15 CV_32F, frame
    /// coordinates) through the same selection / stabilisation / ROI pipeline.
    /// `bgrFrame` may be empty (hairline then falls back to the geometric prior).
    FaceObservation ingest(const cv::Mat& yunetFaces, const cv::Mat& bgrFrame, const cv::Size& frameSize,
                           double t = -1.0);

    void reset();
    const FaceObservation& last() const { return last_; }

private:
    struct RawFace {
        cv::Rect2f box;
        std::array<cv::Point2f, 5> lm;
        float score = 0.f;
    };

    bool detect(const cv::Mat& bgr, RawFace& out);
    bool selectFace(const cv::Mat& faces, float invScale, RawFace& out) const;
    FaceObservation integrate(bool found, const RawFace& raw, const cv::Mat& frame,
                              const cv::Size& frameSize, double t);
    void computePose(double t);
    float estimateHairline(const cv::Mat& bgr, const cv::Point2f& eyeMid, float iod);
    void computeRois(const cv::Mat& frame, const cv::Size& frameSize);

    FaceTrackerConfig cfg_;
    cv::Ptr<cv::FaceDetectorYN> detector_;
    cv::Size detectorInput_{0, 0};
    cv::Mat scaled_, padded_, ycc_, rowMeans_;

    RectEMA faceEma_, foreheadEma_, leftEyeEma_, rightEyeEma_, leftCheekEma_, rightCheekEma_;
    std::array<cv::Point2f, 5> lmState_{};
    bool lmInitialised_ = false;
    float iodState_ = 0.f;
    float hairState_ = 0.f;
    float pitchRef_ = -1.f;   ///< Neutral nose/mouth ratio (slow EMA).
    float prevYaw_ = 0.f, prevPitch_ = 0.f, prevRoll_ = 0.f;
    double prevPoseT_ = -1.0;   ///< Smoothed eye-line → hairline distance (px).

    int missed_ = 0;
    cv::Point2f prevCenter_{-1.f, -1.f};
    FaceObservation last_;
};

}  // namespace vitals

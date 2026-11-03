#include "FaceTracker.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace vitals {

namespace {

constexpr float kPi = 3.14159265358979f;

cv::Rect clipRect(const cv::Rect2f& r, const cv::Size& size) {
    const int x0 = std::max(0, static_cast<int>(std::lround(r.x)));
    const int y0 = std::max(0, static_cast<int>(std::lround(r.y)));
    const int x1 = std::min(size.width, static_cast<int>(std::lround(r.x + r.width)));
    const int y1 = std::min(size.height, static_cast<int>(std::lround(r.y + r.height)));
    if (x1 <= x0 || y1 <= y0) return {};
    return {x0, y0, x1 - x0, y1 - y0};
}

cv::Rect2f centeredRect(const cv::Point2f& c, float w, float h) {
    return {c.x - 0.5f * w, c.y - 0.5f * h, w, h};
}

inline float norm2(const cv::Point2f& p) { return std::sqrt(p.x * p.x + p.y * p.y); }

}  // namespace

// ---------------------------------------------------------------------------
// RectEMA
// ---------------------------------------------------------------------------
const cv::Rect2f& RectEMA::update(const cv::Rect2f& m) {
    if (!initialised_) {
        state_ = m;
        initialised_ = true;
        return state_;
    }
    // Snap on large jumps: centre displacement relative to box size.
    const cv::Point2f dc((m.x + 0.5f * m.width) - (state_.x + 0.5f * state_.width),
                         (m.y + 0.5f * m.height) - (state_.y + 0.5f * state_.height));
    const float scale = std::max(1.f, std::max(state_.width, state_.height));
    if (norm2(dc) > snapRatio_ * scale) {
        state_ = m;
        return state_;
    }
    const float a = alpha_;
    state_.x += a * (m.x - state_.x);
    state_.y += a * (m.y - state_.y);
    state_.width += a * (m.width - state_.width);
    state_.height += a * (m.height - state_.height);
    return state_;
}

// ---------------------------------------------------------------------------
// FaceTracker
// ---------------------------------------------------------------------------
FaceTracker::FaceTracker(const FaceTrackerConfig& cfg)
    : cfg_(cfg),
      faceEma_(cfg.roiAlpha),
      foreheadEma_(cfg.roiAlpha),
      leftEyeEma_(cfg.roiAlpha),
      rightEyeEma_(cfg.roiAlpha),
      leftCheekEma_(cfg.roiAlpha),
      rightCheekEma_(cfg.roiAlpha) {
    if (cfg_.modelPath.empty()) return;  // ingest-only mode (external detector / tests)
    {
        std::ifstream f(cfg_.modelPath, std::ios::binary);
        if (!f.good()) {
            throw std::runtime_error(
                "YuNet model not found at '" + cfg_.modelPath +
                "'.\nDownload it with:\n  cmake -DVITALS_DOWNLOAD_MODEL=ON <build-dir>\n"
                "or manually from:\n  https://github.com/opencv/opencv_zoo/tree/main/models/"
                "face_detection_yunet\nand pass --model=<path>.");
        }
    }
    try {
        detector_ = cv::FaceDetectorYN::create(cfg_.modelPath, "", cv::Size(320, 320),
                                               cfg_.scoreThreshold, cfg_.nmsThreshold,
                                               cfg_.topK, cfg_.backend, cfg_.target);
    } catch (const cv::Exception& e) {
        throw std::runtime_error(
            std::string("Failed to load YuNet model: ") + e.what() +
            "\nNote: face_detection_yunet_2023mar.onnx requires OpenCV >= 4.8.0; "
            "use face_detection_yunet_2022mar.onnx with OpenCV 4.5.4 – 4.7.x.");
    }
    if (detector_.empty()) throw std::runtime_error("cv::FaceDetectorYN::create returned null");
}

void FaceTracker::reset() {
    faceEma_.reset();
    foreheadEma_.reset();
    leftEyeEma_.reset();
    rightEyeEma_.reset();
    leftCheekEma_.reset();
    rightCheekEma_.reset();
    lmInitialised_ = false;
    iodState_ = 0.f;
    hairState_ = 0.f;
    missed_ = 0;
    prevCenter_ = {-1.f, -1.f};
    last_ = FaceObservation{};
}

bool FaceTracker::detect(const cv::Mat& bgr, RawFace& out) {
    const float scale = bgr.cols > cfg_.detectWidth
                            ? static_cast<float>(cfg_.detectWidth) / static_cast<float>(bgr.cols)
                            : 1.f;
    const cv::Mat* input = &bgr;
    if (scale < 1.f) {
        cv::resize(bgr, scaled_, cv::Size(), scale, scale, cv::INTER_AREA);
        input = &scaled_;
    }
    // YuNet's feature pyramid needs both input dimensions divisible by 32
    // (OpenCV >= 4.8 pads internally, older versions assert). Padding on the
    // bottom/right keeps landmark coordinates unchanged.
    const int padW = (32 - input->cols % 32) % 32;
    const int padH = (32 - input->rows % 32) % 32;
    if (padW || padH) {
        cv::copyMakeBorder(*input, padded_, 0, padH, 0, padW, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        input = &padded_;
    }
    if (input->size() != detectorInput_) {
        detector_->setInputSize(input->size());
        detectorInput_ = input->size();
    }

    cv::Mat faces;
    try {
        detector_->detect(*input, faces);
    } catch (const cv::Exception& e) {
        throw std::runtime_error(
            std::string("YuNet inference failed: ") + e.what() +
            "\nThis usually means the model does not match the OpenCV version: "
            "face_detection_yunet_2023mar.onnx requires OpenCV >= 4.8.0 "
            "(use face_detection_yunet_2022mar.onnx with OpenCV 4.5.4 - 4.7.x).");
    }
    return selectFace(faces, 1.f / scale, out);
}

bool FaceTracker::selectFace(const cv::Mat& faces, float inv, RawFace& out) const {
    if (faces.empty() || faces.rows == 0 || faces.cols < 15 || faces.type() != CV_32F) return false;

    // Select the dominant face: largest area weighted by score, with a bonus
    // for continuity with the currently tracked face (avoids identity swaps).
    int best = -1;
    float bestScore = -1.f;
    for (int i = 0; i < faces.rows; ++i) {
        const float* r = faces.ptr<float>(i);
        const cv::Rect2f box(r[0] * inv, r[1] * inv, r[2] * inv, r[3] * inv);
        float s = box.area() * r[14];
        if (last_.valid) {
            const cv::Rect2f inter = box & last_.face;
            const float uni = box.area() + last_.face.area() - inter.area();
            if (uni > 0.f) s *= 1.f + 2.f * inter.area() / uni;
        }
        if (s > bestScore) {
            bestScore = s;
            best = i;
        }
    }
    const float* r = faces.ptr<float>(best);
    out.box = cv::Rect2f(r[0] * inv, r[1] * inv, r[2] * inv, r[3] * inv);
    for (int k = 0; k < 5; ++k) out.lm[k] = cv::Point2f(r[4 + 2 * k] * inv, r[5 + 2 * k] * inv);
    out.score = r[14];
    return true;
}

float FaceTracker::estimateHairline(const cv::Mat& bgr, const cv::Point2f& eyeMid, float d) {
    // Skin reference: patch on the nose bridge / upper cheeks below the eye line.
    const cv::Size fs = bgr.size();
    const cv::Rect ref = clipRect(centeredRect(eyeMid + cv::Point2f(0.f, 0.35f * d), 0.35f * d, 0.25f * d), fs);
    // Vertical scan strip above the brows, centred on the forehead.
    const float yTop = eyeMid.y - 1.45f * d, yBot = eyeMid.y - 0.45f * d;
    const cv::Rect strip = clipRect(cv::Rect2f(eyeMid.x - 0.3f * d, yTop, 0.6f * d, yBot - yTop), fs);
    const float fallback = 1.0f * d;
    if (ref.area() < 9 || strip.height < 4 || strip.width < 3) return fallback;

    cv::cvtColor(bgr(ref), ycc_, cv::COLOR_BGR2YCrCb);
    const cv::Scalar r0 = cv::mean(ycc_);
    cv::cvtColor(bgr(strip), ycc_, cv::COLOR_BGR2YCrCb);
    cv::reduce(ycc_, rowMeans_, 1, cv::REDUCE_AVG, CV_32F);  // strip.height × 1 × 3

    // Walk upwards from the brows; the hairline is the first run of rows whose
    // colour departs from skin (chroma distance + weighted luma difference).
    int run = 0;
    for (int y = rowMeans_.rows - 1; y >= 0; --y) {
        const cv::Vec3f v = rowMeans_.at<cv::Vec3f>(y, 0);
        const float dist = std::hypot(v[1] - static_cast<float>(r0[1]), v[2] - static_cast<float>(r0[2])) +
                           0.18f * std::abs(v[0] - static_cast<float>(r0[0]));
        run = dist > 14.f ? run + 1 : 0;
        if (run >= 3) {
            const float rowY = static_cast<float>(strip.y + y + run - 1);
            return std::clamp(eyeMid.y - rowY, 0.65f * d, 1.35f * d);
        }
    }
    // Skin all the way up (bald / high forehead) or strip clipped by the frame.
    return std::clamp(eyeMid.y - static_cast<float>(strip.y), 0.65f * d, 1.35f * d);
}

void FaceTracker::computeRois(const cv::Mat& frame, const cv::Size& frameSize) {
    // Image-side ordering of the eyes (robust to mirrored input).
    cv::Point2f eL = lmState_[0], eR = lmState_[1];
    if (eL.x > eR.x) std::swap(eL, eR);

    const cv::Point2f axis = eR - eL;
    const float d = std::max(4.f, iodState_);
    const float roll = std::atan2(axis.y, axis.x);
    const cv::Point2f mid = 0.5f * (eL + eR);
    const cv::Point2f up(std::sin(roll), -std::cos(roll));

    // --- Forehead -----------------------------------------------------------
    // Lower bound: eyebrows (≈0.3·IOD above the eye centres, plus margin).
    // Upper bound: hairline, measured dynamically from skin chromaticity when a
    // frame is available, otherwise the anthropometric prior (≈1.0·IOD).
    const float browDist = 0.42f * d;
    float hairMeasure = 1.0f * d;
    if (!frame.empty()) hairMeasure = estimateHairline(frame, mid, d);
    hairState_ = hairState_ <= 0.f ? hairMeasure : hairState_ + 0.15f * (hairMeasure - hairState_);
    const float hairDist = std::max(hairState_, browDist + 0.25f * d);  // guarantee a minimal band
    const float fhHeight = hairDist - browDist;
    const cv::Point2f fhCenter = mid + up * (0.5f * (browDist + hairDist));
    const float fhWidth = 0.95f * d;
    const cv::Rect2f fhMeasure = centeredRect(fhCenter, fhWidth, fhHeight);

    // --- Eyes -----------------------------------------------------------------
    const float ew = 0.58f * d, eh = 0.34f * d;
    const cv::Rect2f lMeasure = centeredRect(eL - up * (0.02f * d), ew, eh);
    const cv::Rect2f rMeasure = centeredRect(eR - up * (0.02f * d), ew, eh);

    last_.forehead = clipRect(foreheadEma_.update(fhMeasure), frameSize);
    last_.leftEye = clipRect(leftEyeEma_.update(lMeasure), frameSize);
    last_.rightEye = clipRect(rightEyeEma_.update(rMeasure), frameSize);

    // --- Cheeks ---------------------------------------------------------------
    // Centred 0.62·IOD below each eye and shifted 0.06·IOD outward (away from the
    // nose); 0.38 × 0.30 IOD keeps them clear of the lower eyelid, nose and mouth.
    const cv::Point2f down = -up;
    const cv::Point2f outward = axis * (1.f / std::max(1.f, norm2(axis)));
    const float cw = 0.38f * d, ch = 0.30f * d;
    const cv::Rect2f lcMeasure = centeredRect(eL + down * (0.62f * d) - outward * (0.06f * d), cw, ch);
    const cv::Rect2f rcMeasure = centeredRect(eR + down * (0.62f * d) + outward * (0.06f * d), cw, ch);
    last_.leftCheek = clipRect(leftCheekEma_.update(lcMeasure), frameSize);
    last_.rightCheek = clipRect(rightCheekEma_.update(rcMeasure), frameSize);
    last_.interOcular = d;
    last_.rollDeg = roll * 180.f / kPi;
}

FaceObservation FaceTracker::update(const cv::Mat& bgr) {
    if (bgr.empty()) {
        reset();
        return last_;
    }
    if (detector_.empty()) throw std::logic_error("FaceTracker::update called without a model; use ingest()");
    RawFace raw;
    const bool found = detect(bgr, raw);
    return integrate(found, raw, bgr, bgr.size());
}

FaceObservation FaceTracker::ingest(const cv::Mat& faces, const cv::Mat& bgrFrame,
                                    const cv::Size& frameSize) {
    RawFace raw;
    const bool found = selectFace(faces, 1.f, raw);
    return integrate(found, raw, bgrFrame, frameSize);
}

FaceObservation FaceTracker::integrate(bool found, const RawFace& raw, const cv::Mat& frame,
                                       const cv::Size& frameSize) {
    if (!found) {
        ++missed_;
        if (last_.valid && missed_ <= cfg_.maxMissedFrames) {
            last_.detectedThisFrame = false;  // coast on the stabilised ROIs
            last_.motion = 0.f;
            return last_;
        }
        reset();
        return last_;
    }
    missed_ = 0;

    // Stabilise the face box and landmarks.
    const cv::Rect2f& face = faceEma_.update(raw.box);
    const float rawIod = norm2(raw.lm[1] - raw.lm[0]);
    if (!lmInitialised_ || std::abs(rawIod - iodState_) > 0.4f * std::max(1.f, iodState_)) {
        lmState_ = raw.lm;
        iodState_ = rawIod;
        lmInitialised_ = true;
    } else {
        const float a = cfg_.landmarkAlpha;
        for (int k = 0; k < 5; ++k) lmState_[k] += a * (raw.lm[k] - lmState_[k]);
        iodState_ += a * (rawIod - iodState_);
    }

    const cv::Point2f center(raw.box.x + 0.5f * raw.box.width, raw.box.y + 0.5f * raw.box.height);
    float motion = 0.f;
    if (prevCenter_.x >= 0.f) motion = norm2(center - prevCenter_) / std::max(1.f, rawIod);
    prevCenter_ = center;

    last_.valid = true;
    last_.detectedThisFrame = true;
    last_.score = raw.score;
    last_.face = face;
    last_.landmarks = lmState_;
    last_.motion = motion;
    computeRois(frame, frameSize);

    // Reject degenerate geometry (face partially outside the frame, etc.).
    if (last_.forehead.area() < 64 || last_.leftEye.area() < 36 || last_.rightEye.area() < 36) {
        last_.valid = false;
    }
    return last_;
}

}  // namespace vitals

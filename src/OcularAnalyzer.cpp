#include "OcularAnalyzer.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace vitals {

OcularAnalyzer::OcularAnalyzer(const OcularConfig& cfg) : cfg_(cfg) {}

void OcularAnalyzer::reset() {
    metrics_ = OcularMetrics{};
    openHistory_.clear();
    openSum_ = 0.0;
    frames_.clear();
    trackedSec_ = closedSec_ = 0.0;
    blinks_.clear();
    smoothedEar_ = -1.0;
    startT_ = lastT_ = -1.0;
    fpsEma_ = 0.0;
    inClosure_ = false;
    closureFrames_ = 0;
}

void OcularAnalyzer::pause(double /*t*/) {
    inClosure_ = false;      // an unobserved closure cannot be classified
    closureFrames_ = 0;
    smoothedEar_ = -1.0;     // restart the EMA on re-acquisition
    lastT_ = -1.0;           // the gap must not count as tracked time
    metrics_.paused = true;
}

// ---------------------------------------------------------------------------
// EAR proxy of a single eye patch
// ---------------------------------------------------------------------------
double OcularAnalyzer::patchEar(const cv::Mat& gray, const cv::Rect& roi) {
    const cv::Rect r = roi & cv::Rect(0, 0, gray.cols, gray.rows);
    if (r.width < 6 || r.height < 4) return -1.0;

    cv::resize(gray(r), patch_, cfg_.patchSize, 0, 0, cv::INTER_AREA);
    cv::GaussianBlur(patch_, patch_, cv::Size(3, 3), 0.0);

    // Median intensity ≈ skin tone (sclera is brighter, iris / lashes darker):
    // thresholding relative to it makes the cue independent of exposure.
    histogram_.assign(256, 0);
    for (int y = 0; y < patch_.rows; ++y) {
        const uchar* row = patch_.ptr<uchar>(y);
        for (int x = 0; x < patch_.cols; ++x) ++histogram_[row[x]];
    }
    const int half = patch_.rows * patch_.cols / 2;
    int acc = 0, median = 0;
    for (; median < 256; ++median) {
        acc += histogram_[static_cast<std::size_t>(median)];
        if (acc >= half) break;
    }
    const int thr = static_cast<int>(cfg_.darkRatio * median);
    if (median < 20) return -1.0;  // under-exposed patch: no usable signal

    // Longest vertical run of dark pixels in every column of the central band.
    const int x0 = static_cast<int>(cfg_.centralBand * patch_.cols);
    const int x1 = patch_.cols - x0;
    colRun_.assign(static_cast<std::size_t>(x1 - x0), 0);
    for (int x = x0; x < x1; ++x) {
        int run = 0, best = 0;
        for (int y = 0; y < patch_.rows; ++y) {
            if (patch_.at<uchar>(y, x) < thr) {
                best = std::max(best, ++run);
            } else {
                run = 0;
            }
        }
        colRun_[static_cast<std::size_t>(x - x0)] = best;
    }
    // Average of the tallest half of the columns (the iris columns; robust to
    // gaze direction and to the sclera columns at the eye corners).
    std::sort(colRun_.begin(), colRun_.end(), std::greater<int>());
    const std::size_t n = std::max<std::size_t>(1, colRun_.size() / 2);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += colRun_[i];
    const double aperture = sum / static_cast<double>(n);

    return aperture / (cfg_.eyeWidthFraction * patch_.cols);
}

// ---------------------------------------------------------------------------
// Bookkeeping
// ---------------------------------------------------------------------------
void OcularAnalyzer::prune(double t) {
    while (!openHistory_.empty() && t - openHistory_.front().t > cfg_.meanWindowSec) {
        openSum_ -= openHistory_.front().v;
        openHistory_.pop_front();
    }
    const double window = std::max(cfg_.perclosWindowSec, cfg_.rateWindowSec);
    while (!frames_.empty() && t - frames_.front().t > window) {
        trackedSec_ -= frames_.front().dt;
        if (frames_.front().closed) closedSec_ -= frames_.front().dt;
        frames_.pop_front();
    }
    while (!blinks_.empty() && t - blinks_.front().t > cfg_.rateWindowSec) blinks_.pop_front();
}

void OcularAnalyzer::pushOpenSample(double ear, double t) {
    openHistory_.push_back({t, ear});
    openSum_ += ear;
}

void OcularAnalyzer::updateBlinkState(double ear, double t) {
    const double mean = metrics_.meanEar;
    // Frame limits scale with the real frame rate (a 150 ms blink is ≈ 5 frames
    // at 30 FPS but ≈ 9 at 60 FPS).
    const double scale = std::clamp(fpsEma_ / cfg_.referenceFps, 0.5, 4.0);
    const int minF = std::max(2, static_cast<int>(std::lround(cfg_.minBlinkFrames * std::max(1.0, scale))));
    const int maxF = std::max(4, static_cast<int>(std::lround(cfg_.maxBlinkFrames * scale)));

    if (!inClosure_) {
        if (ear < cfg_.blinkRatio * mean) {
            inClosure_ = true;
            closureFrames_ = 1;
            closureStartT_ = t;
            closureMinEar_ = ear;
        }
        return;
    }

    if (ear < cfg_.recoverRatio * mean) {  // still closed (hysteresis band included)
        ++closureFrames_;
        closureMinEar_ = std::min(closureMinEar_, ear);
        return;
    }

    // Eye re-opened: classify the closure that just ended.
    inClosure_ = false;
    const double durMs = (t - closureStartT_) * 1000.0;
    if (closureFrames_ >= minF && closureFrames_ <= maxF) {
        const bool complete = closureMinEar_ < cfg_.completeRatio * mean;
        blinks_.push_back({t, complete});
        ++metrics_.totalBlinks;
        if (!complete) ++metrics_.incompleteBlinks;
        metrics_.lastBlinkDurationMs = durMs;
        metrics_.lastBlinkMinRatio = closureMinEar_ / std::max(mean, 1e-9);
    } else if (closureFrames_ > maxF) {
        ++metrics_.longClosures;
        metrics_.lastBlinkDurationMs = durMs;
    }
    closureFrames_ = 0;
}

// ---------------------------------------------------------------------------
// Per-frame entry point
// ---------------------------------------------------------------------------
const OcularMetrics& OcularAnalyzer::update(const cv::Mat& gray, const cv::Rect& leftEye,
                                            const cv::Rect& rightEye, double t) {
    CV_Assert(gray.empty() || gray.type() == CV_8UC1);
    if (gray.empty()) return metrics_;

    const double el = patchEar(gray, leftEye);
    const double er = patchEar(gray, rightEye);
    double ear;
    if (el >= 0.0 && er >= 0.0) ear = 0.5 * (el + er);
    else if (el >= 0.0) ear = el;
    else if (er >= 0.0) ear = er;
    else {
        pause(t);
        return metrics_;
    }

    metrics_.paused = false;
    ++metrics_.framesProcessed;
    if (startT_ < 0.0) startT_ = t;
    double dt = 0.0;
    if (lastT_ >= 0.0 && t > lastT_) {
        dt = std::min(t - lastT_, 0.2);  // never count gaps > 200 ms as tracked time
        const double inst = 1.0 / (t - lastT_);
        if (inst > 2.0 && inst < 240.0) fpsEma_ = fpsEma_ <= 0.0 ? inst : 0.95 * fpsEma_ + 0.05 * inst;
    }
    lastT_ = t;

    smoothedEar_ = smoothedEar_ < 0.0 ? ear : smoothedEar_ + cfg_.emaWeight * (ear - smoothedEar_);
    ear = smoothedEar_;
    prune(t);

    // Rolling mean of *open-eye* EAR: during warm-up every frame contributes;
    // afterwards only frames above the blink threshold, so blinks and long
    // closures cannot drag the reference (and thus the threshold) downwards.
    const bool warm = (t - startT_) >= cfg_.warmupSec;
    const double mean = openHistory_.empty() ? ear : openSum_ / static_cast<double>(openHistory_.size());
    if (!warm || (!inClosure_ && ear >= cfg_.blinkRatio * mean)) pushOpenSample(ear, t);

    const double meanNow = openHistory_.empty() ? ear : openSum_ / static_cast<double>(openHistory_.size());
    metrics_.ear = ear;
    metrics_.meanEar = meanNow;
    metrics_.threshold = cfg_.blinkRatio * meanNow;
    metrics_.openness = meanNow > 1e-6 ? ear / meanNow : 0.0;
    metrics_.valid = warm && openHistory_.size() >= 15 && meanNow > 0.05;
    if (!metrics_.valid) {
        frames_.push_back({t, dt, false});
        trackedSec_ += dt;
        return metrics_;
    }

    updateBlinkState(ear, t);
    metrics_.eyesClosed = inClosure_;

    // PERCLOS: fraction of tracked frames with EAR below closedRatio · mean.
    const bool closed = ear < cfg_.closedRatio * meanNow;
    frames_.push_back({t, dt, closed});
    trackedSec_ += dt;
    if (closed) closedSec_ += dt;
    metrics_.sessionTrackedSec += dt;
    if (closed) metrics_.sessionClosedSec += dt;
    metrics_.perclos = trackedSec_ > 0.0 ? std::clamp(closedSec_ / trackedSec_, 0.0, 1.0) : 0.0;

    // Blink rate over the tracked part of the last 60 s (≥ 10 s avoids start-up spikes).
    int incomplete = 0;
    for (const auto& b : blinks_)
        if (!b.complete) ++incomplete;
    const double window = std::clamp(trackedSec_, 10.0, cfg_.rateWindowSec);
    metrics_.blinkRate = static_cast<double>(blinks_.size()) * 60.0 / window;
    metrics_.incompleteRatio =
        blinks_.empty() ? 0.0 : static_cast<double>(incomplete) / static_cast<double>(blinks_.size());
    return metrics_;
}

}  // namespace vitals

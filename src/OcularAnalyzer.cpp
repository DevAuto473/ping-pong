#include "OcularAnalyzer.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace vitals {

OcularAnalyzer::OcularAnalyzer(const OcularConfig& cfg) : cfg_(cfg) {}

void OcularAnalyzer::reset() {
    metrics_ = OcularMetrics{};
    cueHistory_.clear();
    closureHistory_.clear();
    blinks_.clear();
    smoothedCue_ = -1.0;
    openBaseline_ = floor_ = 0.0;
    startT_ = lastT_ = -1.0;
    paused_ = false;
    inClosure_ = false;
}

void OcularAnalyzer::pause(double /*t*/) {
    paused_ = true;
    inClosure_ = false;     // an unobserved closure cannot be classified
    smoothedCue_ = -1.0;    // restart the cue EMA on re-acquisition
    metrics_.paused = true;
}

double OcularAnalyzer::patchCue(const cv::Mat& gray, const cv::Rect& roi) {
    const cv::Rect r = roi & cv::Rect(0, 0, gray.cols, gray.rows);
    if (r.width < 6 || r.height < 4) return -1.0;

    cv::resize(gray(r), patch_, cfg_.patchSize, 0, 0, cv::INTER_AREA);
    cv::GaussianBlur(patch_, patch_, cv::Size(3, 3), 0.0);
    patch_.convertTo(patchF_, CV_32F);

    // Contrast cue: open eyes expose dark iris against bright sclera.
    cv::Scalar mu, sigma;
    cv::meanStdDev(patchF_, mu, sigma);
    const double contrast = sigma[0] / (mu[0] + 1.0);

    // Aperture cue: vertical spread of horizontal-edge energy (upper vs lower lid).
    cv::Sobel(patchF_, dy_, CV_32F, 0, 1, 3);
    const int h = dy_.rows;
    double sumE = 0.0, sumR = 0.0, sumR2 = 0.0;
    for (int y = 0; y < h; ++y) {
        const float* row = dy_.ptr<float>(y);
        double e = 0.0;
        for (int x = 0; x < dy_.cols; ++x) e += static_cast<double>(row[x]) * row[x];
        sumE += e;
        sumR += e * y;
        sumR2 += e * y * y;
    }
    if (sumE <= 1e-9) return -1.0;
    const double meanR = sumR / sumE;
    const double spread = std::sqrt(std::max(0.0, sumR2 / sumE - meanR * meanR)) / h;

    // Geometric mean keeps the cue ~linear in lid aperture (a product would be quadratic).
    return std::sqrt(contrast * spread);
}

void OcularAnalyzer::prune(double t) {
    while (!cueHistory_.empty() && t - cueHistory_.front().t > cfg_.baselineWindowSec)
        cueHistory_.pop_front();
    while (!closureHistory_.empty() && t - closureHistory_.front().t > cfg_.perclosWindowSec)
        closureHistory_.pop_front();
    while (!blinks_.empty() && t - blinks_.front().t > 60.0) blinks_.pop_front();
}

void OcularAnalyzer::updateBaseline(double cue, double t) {
    // Freeze the baseline during a closure so long closures don't drag it down.
    if (!inClosure_) cueHistory_.push_back({t, cue});
    if (cueHistory_.size() < 10) return;

    sortScratch_.clear();
    for (const auto& s : cueHistory_) sortScratch_.push_back(s.v);
    const auto pct = [&](double q) {
        const std::size_t k = static_cast<std::size_t>(q * (sortScratch_.size() - 1));
        std::nth_element(sortScratch_.begin(), sortScratch_.begin() + static_cast<std::ptrdiff_t>(k),
                         sortScratch_.end());
        return sortScratch_[k];
    };
    openBaseline_ = pct(0.85);
    const double low = pct(0.02);
    floor_ = std::clamp(low, 0.15 * openBaseline_, cfg_.floorRatio * openBaseline_);
}

void OcularAnalyzer::updateBlinkState(double r, double t) {
    if (!inClosure_) {
        if (r < cfg_.blinkThreshold) {
            inClosure_ = true;
            closureStart_ = t;
            closureMin_ = r;
        }
        return;
    }
    closureMin_ = std::min(closureMin_, r);
    if (r > cfg_.recoverThreshold) {
        inClosure_ = false;
        const double dur = t - closureStart_;
        if (dur >= cfg_.minBlinkSec && dur <= cfg_.maxBlinkSec) {
            const bool complete = closureMin_ < cfg_.completeThreshold;
            blinks_.push_back({t, complete});
            ++metrics_.totalBlinks;
            if (!complete) ++metrics_.incompleteBlinks;
            metrics_.lastBlinkDurationMs = dur * 1000.0;
        } else if (dur > cfg_.maxBlinkSec) {
            ++metrics_.longClosures;
            metrics_.lastBlinkDurationMs = dur * 1000.0;
        }
    }
}

const OcularMetrics& OcularAnalyzer::update(const cv::Mat& gray, const cv::Rect& leftEye,
                                            const cv::Rect& rightEye, double t) {
    CV_Assert(gray.empty() || gray.type() == CV_8UC1);
    if (gray.empty()) return metrics_;

    const double cl = patchCue(gray, leftEye);
    const double cr = patchCue(gray, rightEye);
    double cue;
    if (cl >= 0.0 && cr >= 0.0) cue = 0.5 * (cl + cr);
    else if (cl >= 0.0) cue = cl;
    else if (cr >= 0.0) cue = cr;
    else {
        pause(t);
        return metrics_;
    }

    if (startT_ < 0.0) startT_ = t;
    const double dt = lastT_ < 0.0 ? 0.0 : std::clamp(t - lastT_, 0.0, 0.2);  // gaps not counted
    lastT_ = t;
    paused_ = false;
    metrics_.paused = false;

    smoothedCue_ = smoothedCue_ < 0.0 ? cue : smoothedCue_ + cfg_.smoothing * (cue - smoothedCue_);
    prune(t);
    updateBaseline(smoothedCue_, t);

    if (openBaseline_ <= floor_ + 1e-9) return metrics_;
    const double r = std::clamp((smoothedCue_ - floor_) / (openBaseline_ - floor_), 0.0, 1.5);
    metrics_.openness = r;
    metrics_.ear = std::min(r, 1.2) * cfg_.nominalOpenEar;
    metrics_.valid = (t - startT_) >= cfg_.warmupSec && cueHistory_.size() >= 30;
    if (!metrics_.valid) return metrics_;

    updateBlinkState(r, t);
    metrics_.eyesClosed = inClosure_;

    // PERCLOS (P80 by default): fraction of tracked time with r ≤ 1 − closure.
    const bool closed = r <= (1.0 - cfg_.perclosClosure);
    closureHistory_.push_back({t, dt, closed});
    double tracked = 0.0, closedTime = 0.0;
    for (const auto& s : closureHistory_) {
        tracked += s.dt;
        if (s.closed) closedTime += s.dt;
    }
    metrics_.perclos = tracked > 0.0 ? closedTime / tracked : 0.0;

    // Blink rate over the last 60 s of *tracked* time (min. 10 s to avoid spikes).
    int incomplete = 0;
    for (const auto& b : blinks_)
        if (!b.complete) ++incomplete;
    const double window = std::clamp(tracked, 10.0, 60.0);
    metrics_.blinkRate = static_cast<double>(blinks_.size()) * 60.0 / window;
    metrics_.incompleteRatio = blinks_.empty() ? 0.0 : static_cast<double>(incomplete) / blinks_.size();
    return metrics_;
}

}  // namespace vitals

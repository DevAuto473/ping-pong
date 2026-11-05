#include "QualityAdvisor.hpp"

#include <cmath>

namespace vitals {

const char* adviceText(Advice a) {
    switch (a) {
        case Advice::HoldStill: return "HOLD STILL";
        case Advice::MoveCloser: return "MOVE CLOSER TO THE CAMERA";
        case Advice::FaceCamera: return "FACE THE CAMERA";
        case Advice::TooDark: return "TOO DARK - ADD FRONT LIGHT";
        case Advice::TooBright: return "GLARE - REDUCE DIRECT LIGHT";
        case Advice::UncoverForehead: return "UNCOVER FOREHEAD (HAIR / SHADOW)";
        case Advice::Flicker: return "FLICKERING LIGHT / SCREEN DETECTED";
        case Advice::LowFps: return "LOW FRAME RATE - IMPROVE LIGHTING";
        default: return "";
    }
}

const char* adviceId(Advice a) {
    switch (a) {
        case Advice::HoldStill: return "hold_still";
        case Advice::MoveCloser: return "move_closer";
        case Advice::FaceCamera: return "face_camera";
        case Advice::TooDark: return "too_dark";
        case Advice::TooBright: return "too_bright";
        case Advice::UncoverForehead: return "uncover_forehead";
        case Advice::Flicker: return "flicker";
        case Advice::LowFps: return "low_fps";
        default: return "";
    }
}

// ---------------------------------------------------------------------------
QualityAdvisor::QualityAdvisor(const AdviceConfig& cfg) : cfg_(cfg) { reset(); }

void QualityAdvisor::reset() {
    since_.fill(-1.0);
    lastSeen_.fill(-1e9);
    current_ = Advice::None;
}

std::vector<Advice> QualityAdvisor::assess(const AdviceInput& in, const AdviceConfig& cfg) {
    std::vector<Advice> out;
    const FaceObservation* f = in.face;
    if (!f || !f->valid) return out;  // "no face" already has its own banner

    if (in.motionRejected) out.push_back(Advice::HoldStill);
    if (f->interOcular > 0.f && f->interOcular < cfg.minInterOcularPx) out.push_back(Advice::MoveCloser);
    if (std::abs(f->yawDeg) > cfg.maxYawDeg) out.push_back(Advice::FaceCamera);

    if (in.sample) {
        // Exposure from the skin-masked colour of the best available region.
        const RoiSample* r = nullptr;
        for (int k : {kForehead, kLeftCheek, kRightCheek})
            if (in.sample->roi[k].valid) { r = &in.sample->roi[k]; break; }
        if (r) {
            const double luma = 0.114 * r->bgr[0] + 0.587 * r->bgr[1] + 0.299 * r->bgr[2];
            if (luma < cfg.minLuma) out.push_back(Advice::TooDark);
            else if (luma > cfg.maxLuma) out.push_back(Advice::TooBright);
        }
        const RoiSample& fh = in.sample->roi[kForehead];
        if (fh.valid && fh.skinFraction < cfg.minForeheadSkin) out.push_back(Advice::UncoverForehead);
    }
    if (in.signal && in.signal->ambientSuspect) out.push_back(Advice::Flicker);
    if (!in.offline && in.fps > 0.0 && in.fps < cfg.minFps) out.push_back(Advice::LowFps);
    return out;
}

Advice QualityAdvisor::update(const AdviceInput& in, double t) {
    std::array<bool, static_cast<int>(Advice::Count)> now{};
    for (Advice a : assess(in, cfg_)) now[static_cast<std::size_t>(a)] = true;

    current_ = Advice::None;
    for (int i = 0; i < static_cast<int>(Advice::Count); ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        if (now[k]) {
            if (since_[k] < 0.0) since_[k] = t;
            lastSeen_[k] = t;
        } else if (t - lastSeen_[k] > cfg_.holdSec) {
            since_[k] = -1.0;
        }
        const bool shown = since_[k] >= 0.0 && lastSeen_[k] - since_[k] >= cfg_.onsetSec;
        if (shown && current_ == Advice::None) current_ = static_cast<Advice>(i);
    }
    return current_;
}

// ---------------------------------------------------------------------------
DrowsinessMonitor::DrowsinessMonitor(const DrowsinessConfig& cfg) : cfg_(cfg) {}

void DrowsinessMonitor::reset() {
    closedSince_ = -1.0;
    lastLongClosures_ = 0;
    longClosureT_ = -1e9;
    active_ = false;
    episodes_ = 0;
    reason_ = "";
}

bool DrowsinessMonitor::update(const OcularMetrics& om, double t) {
    const bool was = active_;
    if (!om.valid || om.paused) {
        closedSince_ = -1.0;
        active_ = false;
        reason_ = "";
        return active_;
    }
    if (om.eyesClosed) {
        if (closedSince_ < 0.0) closedSince_ = t;
    } else {
        closedSince_ = -1.0;
    }
    if (om.longClosures > lastLongClosures_) longClosureT_ = t;
    lastLongClosures_ = om.longClosures;

    reason_ = "";
    if (closedSince_ >= 0.0 && t - closedSince_ >= cfg_.closureAlertSec) reason_ = "EYES CLOSED";
    else if (t - longClosureT_ <= cfg_.longClosureHoldSec) reason_ = "LONG EYE CLOSURE";
    else if (om.perclos >= cfg_.perclosAlert) reason_ = "HIGH PERCLOS";
    active_ = reason_[0] != '\0';
    if (active_ && !was) ++episodes_;
    return active_;
}

}  // namespace vitals

#include "TelemetryUI.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace vitals {

namespace {

// Palette (BGR) — dark clinical monitor theme.
const cv::Scalar kBg(20, 18, 16);
const cv::Scalar kCard(34, 30, 28);
const cv::Scalar kGrid(60, 55, 50);
const cv::Scalar kText(235, 235, 235);
const cv::Scalar kMuted(150, 145, 140);
const cv::Scalar kPulse(110, 255, 120);
const cv::Scalar kCyan(255, 210, 70);
const cv::Scalar kAmber(0, 190, 255);
const cv::Scalar kRed(80, 80, 255);
const cv::Scalar kViolet(255, 140, 190);

constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;
constexpr int kFontBold = cv::FONT_HERSHEY_DUPLEX;

void blendRect(cv::Mat& img, const cv::Rect& r, const cv::Scalar& color, double alpha) {
    const cv::Rect rr = r & cv::Rect(0, 0, img.cols, img.rows);
    if (rr.area() <= 0) return;
    cv::Mat roi = img(rr);
    cv::Mat fill(roi.size(), roi.type(), color);
    cv::addWeighted(fill, alpha, roi, 1.0 - alpha, 0.0, roi);
}

cv::Size text(cv::Mat& img, const std::string& s, cv::Point org, double scale,
              const cv::Scalar& color, int thickness = 1, int font = kFont) {
    cv::putText(img, s, org, font, scale, color, thickness, cv::LINE_AA);
    int base = 0;
    return cv::getTextSize(s, font, scale, thickness, &base);
}

std::string fmt(const char* f, double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), f, v);
    return buf;
}

void cornerBox(cv::Mat& img, const cv::Rect& r, const cv::Scalar& c, int len, int th) {
    const cv::Point tl = r.tl(), br = r.br();
    const cv::Point tr(br.x, tl.y), bl(tl.x, br.y);
    len = std::min({len, r.width / 3, r.height / 3});
    cv::line(img, tl, tl + cv::Point(len, 0), c, th, cv::LINE_AA);
    cv::line(img, tl, tl + cv::Point(0, len), c, th, cv::LINE_AA);
    cv::line(img, tr, tr + cv::Point(-len, 0), c, th, cv::LINE_AA);
    cv::line(img, tr, tr + cv::Point(0, len), c, th, cv::LINE_AA);
    cv::line(img, bl, bl + cv::Point(len, 0), c, th, cv::LINE_AA);
    cv::line(img, bl, bl + cv::Point(0, -len), c, th, cv::LINE_AA);
    cv::line(img, br, br + cv::Point(-len, 0), c, th, cv::LINE_AA);
    cv::line(img, br, br + cv::Point(0, -len), c, th, cv::LINE_AA);
}

void bar(cv::Mat& img, const cv::Rect& r, double frac, const cv::Scalar& c) {
    cv::rectangle(img, r, kGrid, cv::FILLED);
    const int w = static_cast<int>(std::round(std::clamp(frac, 0.0, 1.0) * r.width));
    if (w > 0) cv::rectangle(img, cv::Rect(r.x, r.y, w, r.height), c, cv::FILLED);
}

}  // namespace

// ---------------------------------------------------------------------------
void TelemetryUI::drawRois(cv::Mat& frame, const FaceObservation& f) const {
    if (!f.valid) return;
    const double s = frame.rows / 720.0;
    const int th = std::max(1, static_cast<int>(std::round(1.5 * s)));
    const cv::Scalar faceCol = f.detectedThisFrame ? kCyan : kAmber;

    cornerBox(frame, cv::Rect(f.face) & cv::Rect(0, 0, frame.cols, frame.rows), faceCol,
              static_cast<int>(22 * s), th + 1);

    blendRect(frame, f.forehead, kPulse, 0.16);
    cv::rectangle(frame, f.forehead, kPulse, 1, cv::LINE_AA);
    text(frame, "rPPG", f.forehead.tl() + cv::Point(3, -5), 0.4 * s + 0.1, kPulse, 1);

    for (const cv::Rect& e : {f.leftEye, f.rightEye}) {
        blendRect(frame, e, kViolet, 0.14);
        cv::rectangle(frame, e, kViolet, 1, cv::LINE_AA);
    }
    for (const auto& p : f.landmarks)
        cv::circle(frame, p, std::max(1, th), kText, cv::FILLED, cv::LINE_AA);
}

void TelemetryUI::drawHeader(cv::Mat& frame, const UiFrameState& st) const {
    const double s = frame.rows / 720.0;
    const int h = static_cast<int>(40 * s);
    blendRect(frame, cv::Rect(0, 0, frame.cols, h), kBg, 0.78);
    cv::line(frame, {0, h}, {frame.cols, h}, kGrid, 1);

    const int base = static_cast<int>(27 * s);
    int x = static_cast<int>(14 * s);
    x += text(frame, "VITALS", {x, base}, 0.7 * s, kPulse, std::max(1, (int)(2 * s)), kFontBold).width;
    x += static_cast<int>(8 * s);
    x += text(frame, "rPPG / OCULAR TELEMETRY", {x, base}, 0.5 * s, kMuted, 1).width + (int)(24 * s);

    std::string badge;
    cv::Scalar col;
    switch (st.state) {
        case SystemState::Running: badge = "TRACKING"; col = kPulse; break;
        case SystemState::Acquiring: {
            const double fill = st.signal ? st.signal->bufferFill : 0.0;
            badge = "ACQUIRING " + std::to_string(static_cast<int>(fill * 100.0)) + "%";
            col = kAmber;
            break;
        }
        case SystemState::FaceLost: badge = "FACE LOST - BUFFERS PAUSED"; col = kRed; break;
        case SystemState::CameraLost: badge = "CAMERA OFFLINE"; col = kRed; break;
    }
    int bl = 0;
    const cv::Size bs = cv::getTextSize(badge, kFont, 0.5 * s, 1, &bl);
    const cv::Rect br(x, static_cast<int>(9 * s), bs.width + (int)(28 * s), static_cast<int>(22 * s));
    blendRect(frame, br, col, 0.22);
    cv::rectangle(frame, br, col, 1, cv::LINE_AA);
    cv::circle(frame, {br.x + (int)(10 * s), br.y + br.height / 2}, std::max(2, (int)(4 * s)), col,
               cv::FILLED, cv::LINE_AA);
    text(frame, badge, {br.x + (int)(20 * s), br.y + br.height - (int)(6 * s)}, 0.5 * s, col, 1);

    // Right side: session clock + FPS.
    const int secs = static_cast<int>(st.sessionSeconds);
    char clock[32];
    std::snprintf(clock, sizeof(clock), "%02d:%02d", secs / 60, secs % 60);
    const std::string fps = fmt("%.1f FPS", st.fps);
    const cv::Size fs = cv::getTextSize(fps, kFontBold, 0.6 * s, 1, &bl);
    const cv::Scalar fpsCol = st.fps >= 24.0 ? kPulse : (st.fps >= 15.0 ? kAmber : kRed);
    const int fx = frame.cols - fs.width - static_cast<int>(14 * s);
    text(frame, fps, {fx, base}, 0.6 * s, fpsCol, 1, kFontBold);
    const cv::Size cs = cv::getTextSize(clock, kFont, 0.55 * s, 1, &bl);
    text(frame, clock, {fx - cs.width - (int)(18 * s), base}, 0.55 * s, kMuted, 1);
}

void TelemetryUI::drawCards(cv::Mat& frame, const cv::Rect& panel, const UiFrameState& st) const {
    const double s = frame.rows / 720.0;
    const int gap = static_cast<int>(10 * s);
    const int cardH = (panel.height - 3 * gap) / 4;
    const bool paused = st.state == SystemState::FaceLost;
    const SignalMetrics* sig = st.signal;
    const OcularMetrics* oc = st.ocular;

    auto card = [&](int idx, const std::string& title, const cv::Scalar& accent) {
        const cv::Rect r(panel.x, panel.y + idx * (cardH + gap), panel.width, cardH);
        blendRect(frame, r, kCard, 0.82);
        cv::rectangle(frame, r, kGrid, 1);
        cv::rectangle(frame, cv::Rect(r.x, r.y, std::max(3, (int)(4 * s)), r.height), accent, cv::FILLED);
        text(frame, title, {r.x + (int)(14 * s), r.y + (int)(20 * s)}, 0.42 * s, kMuted, 1);
        return r;
    };
    const double big = 1.25 * s, unit = 0.5 * s, small = 0.4 * s;
    const int bigTh = std::max(1, (int)(2 * s));
    const cv::Scalar valueCol = paused ? kMuted : kText;

    // --- Heart rate -------------------------------------------------------
    {
        const bool ok = sig && sig->valid;
        const double conf = ok ? sig->confidence : 0.0;
        const cv::Scalar accent = !ok ? kMuted : (conf > 0.6 ? kPulse : (conf > 0.35 ? kAmber : kRed));
        const cv::Rect r = card(0, "HEART RATE", accent);
        const cv::Point vp(r.x + (int)(14 * s), r.y + (int)(64 * s));
        const std::string v = ok ? fmt("%.0f", sig->bpm) : "--";
        const cv::Size vs = text(frame, v, vp, big, ok ? valueCol : kMuted, bigTh, kFontBold);
        text(frame, "BPM", vp + cv::Point(vs.width + (int)(6 * s), 0), unit, kMuted, 1);

        // Beating heart indicator driven by the live waveform.
        if (sig && !sig->waveform.empty() && !paused) {
            const double a = std::max(0.0, static_cast<double>(sig->waveform.back()));
            const cv::Point c(r.x + r.width - (int)(30 * s), r.y + (int)(48 * s));
            cv::circle(frame, c, (int)((7 + 6 * a) * s), kRed, cv::FILLED, cv::LINE_AA);
        }
        const cv::Rect cb(r.x + (int)(14 * s), r.y + r.height - (int)(22 * s), r.width - (int)(80 * s),
                          std::max(3, (int)(5 * s)));
        bar(frame, cb, conf, accent);
        text(frame, fmt("Q %.0f%%", conf * 100.0), {cb.x + cb.width + (int)(8 * s), cb.y + (int)(6 * s)},
             small, kMuted, 1);
    }
    // --- HRV ----------------------------------------------------------------
    {
        const bool ok = sig && sig->valid && sig->rmssdMs > 0.0;
        const cv::Rect r = card(1, "HRV  (RMSSD)", kCyan);
        const cv::Point vp(r.x + (int)(14 * s), r.y + (int)(64 * s));
        const cv::Size vs = text(frame, ok ? fmt("%.0f", sig->rmssdMs) : "--", vp, big,
                                 ok ? valueCol : kMuted, bigTh, kFontBold);
        text(frame, "ms", vp + cv::Point(vs.width + (int)(6 * s), 0), unit, kMuted, 1);
        const std::string sub = sig && sig->meanIbiMs > 0.0
                                    ? fmt("IBI %.0f ms", sig->meanIbiMs) + "   beats " +
                                          std::to_string(sig->beatCount)
                                    : "collecting beats...";
        text(frame, sub, {r.x + (int)(14 * s), r.y + r.height - (int)(12 * s)}, small, kMuted, 1);
    }
    // --- Blink rate ---------------------------------------------------------
    {
        const bool ok = oc && oc->valid;
        const cv::Scalar accent = !ok ? kMuted : ((oc->blinkRate < 8.0 || oc->blinkRate > 30.0) ? kAmber : kViolet);
        const cv::Rect r = card(2, "BLINK RATE", accent);
        const cv::Point vp(r.x + (int)(14 * s), r.y + (int)(64 * s));
        const cv::Size vs = text(frame, ok ? fmt("%.0f", oc->blinkRate) : "--", vp, big,
                                 ok ? valueCol : kMuted, bigTh, kFontBold);
        text(frame, "/min", vp + cv::Point(vs.width + (int)(6 * s), 0), unit, kMuted, 1);
        if (ok) {
            // Live EAR gauge.
            const cv::Rect eb(r.x + r.width - (int)(70 * s), r.y + (int)(40 * s), (int)(56 * s),
                              std::max(3, (int)(5 * s)));
            bar(frame, eb, oc->ear / 0.36, oc->eyesClosed ? kRed : kViolet);
            text(frame, fmt("EAR %.2f", oc->ear), {eb.x, eb.y + (int)(20 * s)}, small, kMuted, 1);
            const std::string sub = "incomplete " + std::to_string(oc->incompleteBlinks) + " (" +
                                    fmt("%.0f%%", oc->incompleteRatio * 100.0) + ")  total " +
                                    std::to_string(oc->totalBlinks);
            text(frame, sub, {r.x + (int)(14 * s), r.y + r.height - (int)(12 * s)}, small, kMuted, 1);
        } else {
            text(frame, "calibrating eye baseline...", {r.x + (int)(14 * s), r.y + r.height - (int)(12 * s)},
                 small, kMuted, 1);
        }
    }
    // --- PERCLOS --------------------------------------------------------------
    {
        const bool ok = oc && oc->valid;
        const double p = ok ? oc->perclos : 0.0;
        // Common drowsiness bands: <7.5 % alert, 7.5–15 % questionable, >15 % drowsy.
        const cv::Scalar accent = !ok ? kMuted : (p < 0.075 ? kPulse : (p < 0.15 ? kAmber : kRed));
        const std::string label = !ok ? "" : (p < 0.075 ? "ALERT" : (p < 0.15 ? "FATIGUE?" : "DROWSY"));
        const cv::Rect r = card(3, "PERCLOS  (P80, 60 s)", accent);
        const cv::Point vp(r.x + (int)(14 * s), r.y + (int)(64 * s));
        const cv::Size vs = text(frame, ok ? fmt("%.1f", p * 100.0) : "--", vp, big,
                                 ok ? valueCol : kMuted, bigTh, kFontBold);
        text(frame, "%", vp + cv::Point(vs.width + (int)(6 * s), 0), unit, kMuted, 1);
        if (ok) {
            int bl = 0;
            const cv::Size ls = cv::getTextSize(label, kFontBold, 0.5 * s, 1, &bl);
            text(frame, label, {r.x + r.width - ls.width - (int)(12 * s), vp.y - (int)(10 * s)}, 0.5 * s,
                 accent, 1, kFontBold);
        }
        const cv::Rect pb(r.x + (int)(14 * s), r.y + r.height - (int)(22 * s), r.width - (int)(28 * s),
                          std::max(3, (int)(5 * s)));
        bar(frame, pb, p / 0.30, accent);  // full scale = 30 %
        const int t1 = pb.x + static_cast<int>(pb.width * 0.075 / 0.30);
        const int t2 = pb.x + static_cast<int>(pb.width * 0.15 / 0.30);
        cv::line(frame, {t1, pb.y - 2}, {t1, pb.y + pb.height + 2}, kText, 1);
        cv::line(frame, {t2, pb.y - 2}, {t2, pb.y + pb.height + 2}, kText, 1);
        if (ok && oc->longClosures > 0)
            text(frame, "long closures " + std::to_string(oc->longClosures),
                 {pb.x, pb.y + (int)(16 * s)}, small, kRed, 1);
    }
}

void TelemetryUI::drawWaveform(cv::Mat& frame, const cv::Rect& area, const SignalMetrics& m,
                               bool paused) const {
    const double s = frame.rows / 720.0;
    blendRect(frame, area, kBg, 0.80);
    cv::rectangle(frame, area, kGrid, 1);

    const cv::Rect plot(area.x + (int)(10 * s), area.y + (int)(26 * s), area.width - (int)(20 * s),
                        area.height - (int)(36 * s));
    text(frame, "PULSE WAVEFORM  (BANDPASS 0.75-2.5 Hz)", {area.x + (int)(12 * s), area.y + (int)(18 * s)},
         0.42 * s, kMuted, 1);
    if (m.sampleRate > 0.0) {
        const std::string fsTxt = fmt("fs %.1f Hz", m.sampleRate) + fmt("   SNR %.1f dB", m.snrDb);
        int bl = 0;
        const cv::Size ts = cv::getTextSize(fsTxt, kFont, 0.42 * s, 1, &bl);
        text(frame, fsTxt, {area.x + area.width - ts.width - (int)(12 * s), area.y + (int)(18 * s)},
             0.42 * s, kMuted, 1);
    }

    // Grid: horizontal quarters + 1 s vertical divisions.
    for (int q = 1; q < 4; ++q) {
        const int y = plot.y + q * plot.height / 4;
        cv::line(frame, {plot.x, y}, {plot.x + plot.width, y}, kGrid, 1);
    }
    const std::size_t n = m.waveform.size();
    if (m.sampleRate > 0.0 && n > 1) {
        const double secs = n / m.sampleRate;
        for (int k = 1; k < secs; ++k) {
            const int x = plot.x + plot.width - static_cast<int>(plot.width * k / secs);
            cv::line(frame, {x, plot.y}, {x, plot.y + plot.height}, kGrid, 1);
        }
    }
    if (n < 2) {
        text(frame, paused ? "SIGNAL PAUSED" : "ACQUIRING SIGNAL...",
             {plot.x + plot.width / 2 - (int)(80 * s), plot.y + plot.height / 2}, 0.55 * s, kMuted, 1);
        return;
    }

    std::vector<cv::Point> pts(n);
    const double half = 0.45 * plot.height;
    const int cy = plot.y + plot.height / 2;
    for (std::size_t i = 0; i < n; ++i) {
        const int x = plot.x + static_cast<int>(static_cast<double>(i) * plot.width / (n - 1));
        const int y = cy - static_cast<int>(std::round(m.waveform[i] * half));
        pts[i] = {x, y};
    }
    const cv::Scalar line = paused ? kMuted : kPulse;
    const cv::Scalar glow = paused ? kGrid : cv::Scalar(40, 110, 45);
    cv::polylines(frame, pts, false, glow, std::max(3, (int)(5 * s)), cv::LINE_AA);
    cv::polylines(frame, pts, false, line, std::max(1, (int)(2 * s)), cv::LINE_AA);

    for (float f : m.beatMarkers) {
        const int x = plot.x + static_cast<int>(f * plot.width);
        cv::line(frame, {x, plot.y}, {x, plot.y + (int)(8 * s)}, kRed, std::max(1, (int)(2 * s)));
    }
    cv::circle(frame, pts.back(), std::max(3, (int)(4 * s)), line, cv::FILLED, cv::LINE_AA);
    if (paused)
        text(frame, "PAUSED", {plot.x + (int)(6 * s), plot.y + (int)(18 * s)}, 0.5 * s, kRed, 1, kFontBold);
}

void TelemetryUI::drawBanner(cv::Mat& frame, const std::string& msg, const cv::Scalar& color) const {
    const double s = frame.rows / 720.0;
    int bl = 0;
    const cv::Size ts = cv::getTextSize(msg, kFontBold, 0.7 * s, 1, &bl);
    const cv::Rect r((frame.cols - ts.width) / 2 - (int)(20 * s), frame.rows / 2 - (int)(30 * s),
                     ts.width + (int)(40 * s), (int)(46 * s));
    blendRect(frame, r, kBg, 0.8);
    cv::rectangle(frame, r, color, 1, cv::LINE_AA);
    text(frame, msg, {r.x + (int)(20 * s), r.y + (int)(31 * s)}, 0.7 * s, color, 1, kFontBold);
}

void TelemetryUI::render(cv::Mat& frame, const UiFrameState& st) const {
    if (frame.empty()) return;
    const double s = frame.rows / 720.0;
    const int margin = static_cast<int>(12 * s);
    const int headerH = static_cast<int>(40 * s);
    const int waveH = static_cast<int>(150 * s);
    const int panelW = std::min(frame.cols / 3, static_cast<int>(300 * s));

    if (st.face) drawRois(frame, *st.face);
    drawHeader(frame, st);

    const cv::Rect wave(margin, frame.rows - waveH - margin, frame.cols - 2 * margin, waveH);
    const cv::Rect panel(frame.cols - panelW - margin, headerH + margin, panelW,
                         wave.y - headerH - 2 * margin);
    drawCards(frame, panel, st);
    if (st.signal) drawWaveform(frame, wave, *st.signal, st.state == SystemState::FaceLost);

    if (st.state == SystemState::FaceLost)
        drawBanner(frame, st.statusMessage.empty() ? "NO FACE DETECTED" : st.statusMessage, kRed);

    text(frame, "Research prototype - not a medical device",
         {margin, wave.y - (int)(8 * s)}, 0.38 * s, kMuted, 1);
}

cv::Mat TelemetryUI::renderNoSignal(const cv::Size& size, const std::string& message, double fps) const {
    cv::Mat img(size, CV_8UC3, kBg);
    UiFrameState st;
    st.state = SystemState::CameraLost;
    st.fps = fps;
    drawHeader(img, st);
    drawBanner(img, "CAMERA DISCONNECTED", kRed);
    const double s = size.height / 720.0;
    int bl = 0;
    const cv::Size ts = cv::getTextSize(message, kFont, 0.55 * s, 1, &bl);
    text(img, message, {(size.width - ts.width) / 2, size.height / 2 + (int)(50 * s)}, 0.55 * s, kMuted, 1);
    return img;
}

}  // namespace vitals

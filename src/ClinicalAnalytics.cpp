#include "ClinicalAnalytics.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <sstream>

namespace vitals {

namespace {
constexpr double kPi = 3.14159265358979323846;

std::string fmt(const char* f, double v) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), f, v);
    return buf;
}

double stddev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}

/// Linear interpolation of x at fractional index p (clamped).
double sampleAt(const std::vector<double>& x, double p) {
    if (x.empty()) return 0.0;
    p = std::clamp(p, 0.0, static_cast<double>(x.size() - 1));
    const std::size_t i = static_cast<std::size_t>(p);
    if (i + 1 >= x.size()) return x.back();
    const double u = p - static_cast<double>(i);
    return x[i] + u * (x[i + 1] - x[i]);
}

void smooth3(std::vector<double>& v, int passes) {
    if (v.size() < 3) return;
    std::vector<double> t(v.size());
    for (int p = 0; p < passes; ++p) {
        t.front() = v.front();
        t.back() = v.back();
        for (std::size_t i = 1; i + 1 < v.size(); ++i) t[i] = (v[i - 1] + v[i] + v[i + 1]) / 3.0;
        v.swap(t);
    }
}
}  // namespace

ClinicalAnalytics::ClinicalAnalytics(const ClinicalConfig& cfg) : cfg_(cfg) {}

double ClinicalAnalytics::median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const std::size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m), v.end());
    return v[m];
}

// ---------------------------------------------------------------------------
// Per-frame aggregation
// ---------------------------------------------------------------------------
void ClinicalAnalytics::update(double t, const SignalMetrics& sm, const OcularMetrics& om, bool faceTracked,
                               bool motionRejected) {
    if (tFirst_ < 0.0) tFirst_ = t;
    tLast_ = t;
    ++frames_;
    if (faceTracked) ++trackedFrames_;
    if (motionRejected) ++motionFrames_;
    ocular_ = om;

    for (const BeatEvent& b : sm.newBeats) beats_.push_back(b);

    if (faceTracked && !sm.paused && sm.bufferFill >= 1.0) {
        sqiSum_ += sm.sqi;
        ++sqiCount_;
    }
    if (!sm.paused && sm.valid && sm.confidence >= cfg_.minHrConfidence) {
        hrSum_ += sm.bpm;
        ++hrCount_;
        hrMin_ = std::min(hrMin_, sm.bpm);
        hrMax_ = std::max(hrMax_, sm.bpm);
    }
    if (!sm.paused && sm.backgroundChecked) {
        ++ambientChecked_;
        if (sm.ambientSuspect) ++ambientSuspect_;
    }
    if (t - lastAnalysis_ >= cfg_.analysisPeriodSec) {
        lastAnalysis_ = t;
        periodicAnalysis(sm);
    }
}

void ClinicalAnalytics::periodicAnalysis(const SignalMetrics& sm) {
    hrv_ = analyseHrv(beats_, cfg_);
    const bool good = !sm.paused && sm.valid && sm.confidence >= cfg_.minHrConfidence;
    if (good && sm.ratioValid) ratioSamples_.push_back(sm.ratioRB);
    if (!sm.paused && sm.respValid) respSamples_.push_back(sm.respRate);
    if (good && sm.sampleRate > 0.0) {
        const ApgSummary a = analyseApg(sm.morphWave, sm.sampleRate, sm.instantBpm / 60.0, cfg_);
        if (a.valid) {
            apgLatest_ = a;
            baSamples_.push_back(a.ba);
            riseSamples_.push_back(a.riseTimeMs);
            crestSamples_.push_back(a.crestRatio);
            if (a.agiValid) agiSamples_.push_back(a.agi);
            if (a.riValid) riSamples_.push_back(a.reflectionIndex);
        }
    }
}

bool ClinicalAnalytics::spo2(double& percent, double& ratio) const {
    if (ratioSamples_.empty()) return false;
    ratio = median(ratioSamples_);
    percent = cfg_.spo2A - cfg_.spo2B * ratio;
    return true;
}

// ---------------------------------------------------------------------------
// HRV: time domain, Poincaré, ectopic screening, Welch LF/HF
// ---------------------------------------------------------------------------
HrvSummary ClinicalAnalytics::analyseHrv(const std::vector<BeatEvent>& beats, const ClinicalConfig& cfg) {
    HrvSummary h;
    h.beats = static_cast<int>(beats.size());

    // 1. Plausible intervals between consecutive beats of the same segment.
    struct Ibi { double t; double ms; int seg; };
    std::vector<Ibi> ibi;
    for (std::size_t i = 1; i < beats.size(); ++i) {
        if (beats[i].segment != beats[i - 1].segment) continue;
        const double ms = (beats[i].t - beats[i - 1].t) * 1000.0;
        if (ms >= cfg.ibiMinMs && ms <= cfg.ibiMaxMs) ibi.push_back({beats[i].t, ms, beats[i].segment});
    }
    h.intervals = static_cast<int>(ibi.size());
    if (ibi.size() < 5) return h;

    // 2. Local median → NN classification and ectopic pattern search.
    std::vector<double> med(ibi.size());
    for (std::size_t i = 0; i < ibi.size(); ++i) {
        std::vector<double> win;
        const std::size_t a = i >= static_cast<std::size_t>(cfg.localMedianHalfWidth) ? i - cfg.localMedianHalfWidth : 0;
        const std::size_t b = std::min(ibi.size(), i + cfg.localMedianHalfWidth + 1);
        for (std::size_t j = a; j < b; ++j)
            if (ibi[j].seg == ibi[i].seg) win.push_back(ibi[j].ms);
        med[i] = median(win);
    }
    std::vector<char> nn(ibi.size());
    for (std::size_t i = 0; i < ibi.size(); ++i) nn[i] = std::abs(ibi[i].ms - med[i]) <= cfg.nnTolerance * med[i];
    for (std::size_t i = 0; i + 1 < ibi.size(); ++i) {
        if (ibi[i + 1].seg != ibi[i].seg) continue;
        // Successive intervals must be contiguous (no dropped beat between them).
        if (std::abs((ibi[i + 1].t - ibi[i].t) * 1000.0 - ibi[i + 1].ms) > 1.0) continue;
        const double m = med[i];
        const double sum = ibi[i].ms + ibi[i + 1].ms;
        if (ibi[i].ms < cfg.prematureRatio * m && ibi[i + 1].ms > cfg.compensatoryRatio * m &&
            sum > 1.7 * m && sum < 2.3 * m) {
            ++h.ectopicBeats;
            ++i;  // the compensatory interval belongs to the same event
        }
    }

    // 3. Time domain + Poincaré on NN intervals.
    std::vector<double> nnVals, diffs;
    for (std::size_t i = 0; i < ibi.size(); ++i) {
        if (!nn[i]) continue;
        nnVals.push_back(ibi[i].ms);
        const bool adjacent = i > 0 && nn[i - 1] && ibi[i - 1].seg == ibi[i].seg &&
                              std::abs((ibi[i].t - ibi[i - 1].t) * 1000.0 - ibi[i].ms) <= 1.0;
        if (adjacent) diffs.push_back(ibi[i].ms - ibi[i - 1].ms);
    }
    h.nnCount = static_cast<int>(nnVals.size());
    if (nnVals.size() >= 5 && diffs.size() >= 4) {
        h.timeValid = true;
        h.meanNnMs = std::accumulate(nnVals.begin(), nnVals.end(), 0.0) / static_cast<double>(nnVals.size());
        h.sdnnMs = stddev(nnVals);
        double ss = 0.0;
        for (double d : diffs) ss += d * d;
        h.rmssdMs = std::sqrt(ss / static_cast<double>(diffs.size()));
        // SD1² = ½·Var(ΔRR);  SD2² = 2·SDNN² − SD1²
        const double sdDiff = stddev(diffs);
        h.sd1Ms = std::sqrt(0.5) * sdDiff;
        h.sd2Ms = std::sqrt(std::max(0.0, 2.0 * h.sdnnMs * h.sdnnMs - h.sd1Ms * h.sd1Ms));
        h.poincareValid = h.sd2Ms > 0.0;
    }
    h.irregular = h.ectopicBeats >= cfg.minEctopicForFlag;

    // 4. Frequency domain: longest contiguous NN run → 4 Hz tachogram → Welch.
    std::vector<std::pair<double, double>> best, run;
    for (std::size_t i = 0; i < ibi.size(); ++i) {
        if (!nn[i]) continue;
        if (!run.empty() && (ibi[i].t - run.back().first > cfg.maxNnGapSec || ibi[i].seg != ibi[i - 1].seg)) {
            if (run.size() > best.size()) best = run;
            run.clear();
        }
        run.emplace_back(ibi[i].t, ibi[i].ms);
    }
    if (run.size() > best.size()) best = run;
    if (best.size() < 8) return h;
    const double span = best.back().first - best.front().first;
    h.analysedSec = span;
    if (span < cfg.hrvMinSeconds) return h;

    const double fsT = cfg.tachogramHz;
    const int n = static_cast<int>(span * fsT);
    std::vector<double> tach(static_cast<std::size_t>(n));
    std::size_t j = 0;
    for (int i = 0; i < n; ++i) {
        const double t = best.front().first + i / fsT;
        while (j + 2 < best.size() && best[j + 1].first < t) ++j;
        const auto& a = best[j];
        const auto& b = best[std::min(j + 1, best.size() - 1)];
        const double u = b.first > a.first ? std::clamp((t - a.first) / (b.first - a.first), 0.0, 1.0) : 0.0;
        tach[static_cast<std::size_t>(i)] = a.second + u * (b.second - a.second);
    }
    const int seg = std::min(n, static_cast<int>(cfg.welchSegmentSec * fsT));
    const int step = std::max(1, seg / 2);
    const int nfft = cv::getOptimalDFTSize(std::max(1024, seg));
    std::vector<double> psd(static_cast<std::size_t>(nfft / 2), 0.0);
    double wss = 0.0;
    for (int i = 0; i < seg; ++i) {
        const double w = 0.5 - 0.5 * std::cos(2.0 * kPi * i / (seg - 1));  // Hann
        wss += w * w;
    }
    int segments = 0;
    for (int start = 0; start + seg <= n; start += step) {
        // Detrend (least squares line) then window.
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < seg; ++i) {
            const double y = tach[static_cast<std::size_t>(start + i)];
            sx += i; sy += y; sxx += static_cast<double>(i) * i; sxy += i * y;
        }
        const double den = seg * sxx - sx * sx;
        const double slope = std::abs(den) > 1e-12 ? (seg * sxy - sx * sy) / den : 0.0;
        const double icpt = (sy - slope * sx) / seg;
        cv::Mat src = cv::Mat::zeros(1, nfft, CV_64F), spec;
        double* s = src.ptr<double>();
        for (int i = 0; i < seg; ++i) {
            const double w = 0.5 - 0.5 * std::cos(2.0 * kPi * i / (seg - 1));
            s[i] = (tach[static_cast<std::size_t>(start + i)] - (icpt + slope * i)) * w;
        }
        cv::dft(src, spec, cv::DFT_COMPLEX_OUTPUT);
        const cv::Vec2d* c = spec.ptr<cv::Vec2d>();
        for (int k = 0; k < nfft / 2; ++k) psd[static_cast<std::size_t>(k)] += c[k][0] * c[k][0] + c[k][1] * c[k][1];
        ++segments;
    }
    if (segments == 0) return h;
    const double df = fsT / nfft;
    double lf = 0.0, hf = 0.0, hfPeak = 0.0, hfPeakHz = 0.0;
    for (int k = 1; k < nfft / 2; ++k) {
        // One-sided PSD in ms²/Hz, averaged over segments.
        const double p = 2.0 * psd[static_cast<std::size_t>(k)] / (segments * fsT * wss);
        const double f = k * df;
        if (f >= 0.04 && f < 0.15) lf += p * df;
        if (f >= 0.15 && f < 0.40) {
            hf += p * df;
            if (p > hfPeak) { hfPeak = p; hfPeakHz = f; }
        }
    }
    if (hf <= 1e-12) return h;
    h.freqValid = true;
    h.lfMs2 = lf;
    h.hfMs2 = hf;
    h.lfHf = lf / hf;
    h.rsaBrpm = hfPeakHz * 60.0;
    return h;
}

// ---------------------------------------------------------------------------
// APG (second derivative) and pulse morphology on an ensemble-averaged beat
// ---------------------------------------------------------------------------
ApgSummary ClinicalAnalytics::analyseApg(const std::vector<double>& x, double fs, double hrHz,
                                         const ClinicalConfig& cfg) {
    ApgSummary out;
    const int n = static_cast<int>(x.size());
    if (fs <= 0.0 || hrHz <= 0.0 || n < static_cast<int>(cfg.apgMinSeconds * fs)) return out;
    const double period = fs / hrHz;
    const int minDist = std::max(2, static_cast<int>(0.6 * period));
    const int edge = static_cast<int>(std::ceil(fs));  // skip 1 s at both ends (filter transients)

    // Systolic maxima (non-maximum suppression), then sub-sample feet between them.
    std::vector<int> cand;
    for (int i = edge; i < n - edge; ++i)
        if (x[i] > 0.0 && x[i] > x[i - 1] && x[i] >= x[i + 1]) cand.push_back(i);
    std::sort(cand.begin(), cand.end(), [&](int a, int b) { return x[a] > x[b]; });
    std::vector<int> peaks;
    for (int i : cand) {
        bool ok = true;
        for (int k : peaks)
            if (std::abs(k - i) < minDist) { ok = false; break; }
        if (ok) peaks.push_back(i);
    }
    std::sort(peaks.begin(), peaks.end());
    std::vector<double> feet;
    for (std::size_t i = 1; i < peaks.size(); ++i) {
        int m = peaks[i - 1] + 1;
        for (int k = peaks[i - 1] + 1; k < peaks[i]; ++k)
            if (x[k] < x[m]) m = k;
        double d = 0.0;
        if (m > 0 && m + 1 < n) {
            const double den = x[m - 1] - 2.0 * x[m] + x[m + 1];
            if (std::abs(den) > 1e-15) d = std::clamp(0.5 * (x[m - 1] - x[m + 1]) / den, -0.5, 0.5);
        }
        feet.push_back(m + d);
    }

    // Resample every plausible foot-to-foot cycle to a fixed grid; median template.
    const int N = cfg.apgTemplatePoints;
    std::vector<std::vector<double>> cycles;
    std::vector<double> lengths;
    for (std::size_t i = 1; i < feet.size(); ++i) {
        const double L = feet[i] - feet[i - 1];
        if (L < 0.7 * period || L > 1.3 * period) continue;
        const double y0 = sampleAt(x, feet[i - 1]), y1 = sampleAt(x, feet[i]);
        std::vector<double> c(static_cast<std::size_t>(N));
        double mx = -1e18;
        for (int k = 0; k < N; ++k) {
            const double u = static_cast<double>(k) / N;
            c[static_cast<std::size_t>(k)] = sampleAt(x, feet[i - 1] + u * L) - (y0 + u * (y1 - y0));
            mx = std::max(mx, c[static_cast<std::size_t>(k)]);
        }
        if (mx <= 1e-12) continue;
        for (double& v : c) v /= mx;
        cycles.push_back(std::move(c));
        lengths.push_back(L);
    }
    out.beatsUsed = static_cast<int>(cycles.size());
    if (out.beatsUsed < cfg.apgMinBeats) return out;

    std::vector<double> tpl(static_cast<std::size_t>(N));
    for (int k = 0; k < N; ++k) {
        std::vector<double> col;
        for (const auto& c : cycles) col.push_back(c[static_cast<std::size_t>(k)]);
        tpl[static_cast<std::size_t>(k)] = median(col);
    }
    smooth3(tpl, 2);
    std::vector<double> acc(static_cast<std::size_t>(N), 0.0);
    for (int k = 1; k + 1 < N; ++k)
        acc[static_cast<std::size_t>(k)] = tpl[static_cast<std::size_t>(k + 1)] - 2.0 * tpl[static_cast<std::size_t>(k)] +
                                           tpl[static_cast<std::size_t>(k - 1)];
    smooth3(acc, 2);

    const auto at = [&](int k) { return acc[static_cast<std::size_t>(k)]; };
    const int sp = static_cast<int>(std::max_element(tpl.begin(), tpl.end()) - tpl.begin());
    const double cycleMs = median(lengths) / fs * 1000.0;
    out.riseTimeMs = sp * cycleMs / N;
    out.crestRatio = static_cast<double>(sp) / N;

    // a: acceleration maximum in early systole (before the systolic peak).
    auto argExt = [&](int lo, int hi, bool wantMax) {
        lo = std::max(1, lo);
        hi = std::min(N - 2, hi);
        int best = -1;
        for (int k = lo; k <= hi; ++k) {
            const bool ext = wantMax ? (at(k) >= at(k - 1) && at(k) >= at(k + 1))
                                     : (at(k) <= at(k - 1) && at(k) <= at(k + 1));
            if (!ext) continue;
            if (best < 0 || (wantMax ? at(k) > at(best) : at(k) < at(best))) best = k;
        }
        return best;
    };
    auto firstExt = [&](int lo, int hi, bool wantMax) {
        lo = std::max(1, lo);
        hi = std::min(N - 2, hi);
        for (int k = lo; k <= hi; ++k) {
            if (wantMax ? (at(k) > at(k - 1) && at(k) >= at(k + 1)) : (at(k) < at(k - 1) && at(k) <= at(k + 1)))
                return k;
        }
        return -1;
    };
    const int ia = argExt(1, std::max(2, sp), true);
    if (ia < 0 || at(ia) <= 0.0) return out;
    const int ib = argExt(ia + 1, std::min(N - 2, sp + N / 10), false);
    if (ib < 0 || at(ib) >= 0.0) return out;
    out.ba = at(ib) / at(ia);
    out.valid = true;

    const int ic = firstExt(ib + 1, ib + 3 * N / 10, true);
    const int id = ic > 0 ? firstExt(ic + 1, ic + N / 5, false) : -1;
    const int ie = id > 0 ? firstExt(id + 1, std::min(N - 2, 7 * N / 10), true) : -1;
    if (ic > 0 && id > 0 && ie > 0) {
        out.agi = (at(ib) - at(ic) - at(id) - at(ie)) / at(ia);
        out.agiValid = true;
    }

    // Reflection index: diastolic peak (or, if absent, the e-wave inflection) / systolic peak.
    int dia = -1;
    for (int k = sp + N / 10; k < 85 * N / 100; ++k)
        if (tpl[static_cast<std::size_t>(k)] > tpl[static_cast<std::size_t>(k - 1)] &&
            tpl[static_cast<std::size_t>(k)] >= tpl[static_cast<std::size_t>(k + 1)]) { dia = k; break; }
    if (dia < 0 && ie > sp) dia = ie;
    if (dia > 0) {
        out.reflectionIndex = tpl[static_cast<std::size_t>(dia)] / std::max(1e-9, tpl[static_cast<std::size_t>(sp)]);
        out.riValid = true;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Runtime log line
// ---------------------------------------------------------------------------
std::string ClinicalAnalytics::statusLine(double t, const SignalMetrics& sm, const OcularMetrics& om,
                                          bool motionRejected) const {
    std::ostringstream o;
    o << fmt("t=%6.1fs", t);
    if (sm.valid)
        o << fmt(" | HR %5.1f bpm", sm.bpm) << fmt(" conf %3.0f%%", sm.confidence * 100.0);
    else
        o << " | HR   --- bpm conf   -%";
    o << fmt(" SQI %3.0f%%", sm.sqi * 100.0) << fmt(" [F %3.0f", sm.roiSqi[kForehead] * 100.0)
      << fmt(" L %3.0f", sm.roiSqi[kLeftCheek] * 100.0) << fmt(" R %3.0f]", sm.roiSqi[kRightCheek] * 100.0);
    o << (motionRejected ? " | MOTION" : " | still ");
    if (hrv_.timeValid)
        o << fmt(" | RMSSD %5.1f", hrv_.rmssdMs) << fmt(" SDNN %5.1f", hrv_.sdnnMs);
    else
        o << " | RMSSD  ---  SDNN  ---";
    o << (hrv_.freqValid ? fmt(" LF/HF %4.2f", hrv_.lfHf) : std::string(" LF/HF  ---"));
    double sp = 0, r = 0;
    o << (spo2(sp, r) ? fmt(" | SpO2* %5.1f", sp) + fmt(" (R %.2f)", r) : std::string(" | SpO2*   ---"));
    o << (sm.respValid ? fmt(" | RR %4.1f", sm.respRate) : std::string(" | RR  ---"));
    o << (apgLatest_.valid ? fmt(" | b/a %5.2f", apgLatest_.ba) : std::string(" | b/a   ---"));
    o << " | blinks " << om.totalBlinks << " (inc " << om.incompleteBlinks << ")"
      << fmt(" PERCLOS %4.1f%%", om.perclos * 100.0);
    o << (sm.ambientSuspect ? " | AMBIENT!" : "");
    return o.str();
}

// ---------------------------------------------------------------------------
// End-of-session report
// ---------------------------------------------------------------------------
std::string ClinicalAnalytics::report() const {
    const std::string rule(80, '=');
    std::ostringstream o;
    auto line = [&](const std::string& label, const std::string& value) {
        std::string l = label;
        if (l.size() < 26) l.append(26 - l.size(), ' ');
        o << l << value << '\n';
    };

    const double dur = tFirst_ >= 0.0 ? std::max(0.0, tLast_ - tFirst_) : 0.0;
    const int secs = static_cast<int>(std::lround(dur));
    char durBuf[32];
    std::snprintf(durBuf, sizeof(durBuf), "%02d:%02d", secs / 60, secs % 60);

    o << '\n' << rule << '\n' << "                    CLINICAL VITALS & TELEMETRY REPORT\n" << rule << '\n';
    line("Session Duration:", durBuf);
    const double motionPct = frames_ > 0 ? 100.0 * static_cast<double>(motionFrames_) / frames_ : 0.0;
    line("Overall Signal Integrity:",
         (sqiCount_ > 0 ? fmt("%.1f%% mean SQI", 100.0 * sqiSum_ / sqiCount_) : std::string("N/A")) +
             " | Motion Rejected Frames: " + std::to_string(motionFrames_) + fmt(" / %.1f%%", motionPct));
    if (ambientChecked_ > 0) {
        const double pct = 100.0 * static_cast<double>(ambientSuspect_) / ambientChecked_;
        line("Ambient Lock Check:", pct < 5.0 ? fmt("PASS (background pulse match in %.1f%% of windows)", pct)
                                               : fmt("WARNING - background matches pulse in %.1f%% of windows", pct));
    } else {
        line("Ambient Lock Check:", "N/A (no background visible)");
    }

    // ---- Cardiovascular ------------------------------------------------------
    o << "\n[ CARDIOVASCULAR & HEMODYNAMIC PROFILE ]\n";
    if (hrCount_ > 0)
        line("- Heart Rate (BPM):", fmt("Mean: %.1f", hrSum_ / hrCount_) + fmt(" | Min: %.1f", hrMin_) +
                                        fmt(" | Max: %.1f", hrMax_));
    else
        line("- Heart Rate (BPM):", "N/A (no frames above the confidence threshold)");

    if (!riseSamples_.empty()) {
        std::string m = fmt("rise time %.0f ms", median(riseSamples_)) + fmt(" | crest ratio %.2f", median(crestSamples_));
        m += riSamples_.empty() ? std::string(" | RI N/A") : fmt(" | reflection index %.2f", median(riSamples_));
        line("  Pulse Morphology:", m);
    } else {
        line("  Pulse Morphology:", "N/A (insufficient clean beats)");
    }

    double spo2v = 0, ratio = 0;
    if (spo2(spo2v, ratio)) {
        if (spo2v >= cfg_.spo2PlausibleMin && spo2v <= 100.0)
            line("- Blood Oxygen (SpO2):", fmt("%.1f%% UNCALIBRATED", spo2v) + fmt(" (red/blue ratio-of-ratios R = %.2f)", ratio));
        else
            line("- Blood Oxygen (SpO2):", fmt("out of calibrated range (R = %.2f); requires reference calibration", ratio));
    } else {
        line("- Blood Oxygen (SpO2):", "N/A (insufficient pulsatile red/blue signal)");
    }

    if (!baSamples_.empty()) {
        const double ba = median(baSamples_);
        const char* cls = ba <= -0.70 ? "Optimal" : (ba <= -0.50 ? "Elevated" : "Rigid");
        std::string v = fmt("b/a %.2f", ba) + " (Classification: " + cls + ", indicative)";
        if (!agiSamples_.empty()) v += fmt(" | AGI (b-c-d-e)/a %.2f", median(agiSamples_));
        line("- Vascular Aging Index:", v);
    } else {
        line("- Vascular Aging Index:", "N/A (APG a/b waves not resolved)");
    }

    // ---- Autonomic -------------------------------------------------------------
    o << "\n[ AUTONOMIC & HRV ANALYSIS ]\n";
    if (hrv_.timeValid) {
        line("- RMSSD:", fmt("%.1f ms", hrv_.rmssdMs) + " (" + std::to_string(hrv_.nnCount) + " NN intervals)");
        line("- SDNN:", fmt("%.1f ms", hrv_.sdnnMs));
    } else {
        line("- RMSSD:", "N/A (too few clean beats)");
        line("- SDNN:", "N/A");
    }
    if (hrv_.freqValid) {
        const char* st = hrv_.lfHf > 2.0 ? "Sympathetic Dominant" : (hrv_.lfHf < 0.5 ? "Parasympathetic Dominant" : "Balanced");
        std::string v = fmt("%.2f", hrv_.lfHf) + " (Autonomic Stress: " + st + ")";
        if (hrv_.analysedSec < cfg_.hrvStandardSeconds) v += fmt(" [short-term, %.0f s]", hrv_.analysedSec);
        line("- LF / HF Ratio:", v);
    } else {
        line("- LF / HF Ratio:", fmt("N/A (needs >= %.0f s of contiguous clean beats", cfg_.hrvMinSeconds) +
                                     fmt("; longest run %.0f s)", hrv_.analysedSec));
    }
    if (hrv_.timeValid) {
        line("- Arrhythmia Screening:", hrv_.irregular
                                            ? "Suspected Ectopic Beats Detected (" + std::to_string(hrv_.ectopicBeats) + ")"
                                            : std::string("No Interval Irregularity Detected (not an ECG rhythm diagnosis)"));
        if (hrv_.poincareValid)
            line("", fmt("(Poincare Dispersion SD1/SD2: %.2f", hrv_.sd1Ms / hrv_.sd2Ms) +
                         fmt("; SD1 %.1f ms", hrv_.sd1Ms) + fmt(", SD2 %.1f ms)", hrv_.sd2Ms));
    } else {
        line("- Arrhythmia Screening:", "N/A (too few clean beats)");
    }

    // ---- Respiratory & ocular --------------------------------------------------
    o << "\n[ RESPIRATORY & OCULAR FATIGUE ]\n";
    if (hrv_.freqValid && hrv_.rsaBrpm > 0.0)
        line("- Respiration Rate:", fmt("%.1f BrPM (Derived from Respiratory Sinus Arrhythmia)", hrv_.rsaBrpm));
    else if (!respSamples_.empty())
        line("- Respiration Rate:", fmt("%.1f BrPM (Derived from rPPG baseline modulation)", median(respSamples_)));
    else
        line("- Respiration Rate:", "N/A (no stable respiratory component)");

    const double ocMin = ocular_.sessionTrackedSec / 60.0;
    if (ocMin > 0.1) {
        line("- Blink Rate:", fmt("%.1f blinks/min", ocular_.totalBlinks / ocMin) +
                                  " (Total Blinks: " + std::to_string(ocular_.totalBlinks) + ")");
        const double inc = ocular_.totalBlinks > 0 ? 100.0 * ocular_.incompleteBlinks / ocular_.totalBlinks : 0.0;
        line("- Incomplete Blinks:", std::to_string(ocular_.incompleteBlinks) + fmt(" (%.1f%%)", inc));
        const double p = ocular_.sessionTrackedSec > 0.0 ? 100.0 * ocular_.sessionClosedSec / ocular_.sessionTrackedSec : 0.0;
        const char* st = p < 7.5 ? "Alert" : (p < 15.0 ? "Fatigued" : "Drowsy");
        line("- PERCLOS Index:", fmt("%.1f%%", p) + " (Alertness State: " + st + ")");
    } else {
        line("- Blink Rate:", "N/A (eyes not tracked)");
        line("- Incomplete Blinks:", "N/A");
        line("- PERCLOS Index:", "N/A");
    }

    o << '\n' << rule << '\n'
      << "             NOTICE: FOR RESEARCH & MONITORING PURPOSES ONLY.\n"
      << "   Not a medical device. SpO2 is uncalibrated.\n"
      << rule << '\n';
    return o.str();
}

}  // namespace vitals

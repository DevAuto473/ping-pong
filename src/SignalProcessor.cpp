#include "SignalProcessor.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace vitals {

namespace {
constexpr double kPi = 3.14159265358979323846;

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const std::size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m), v.end());
    return v[m];
}
}  // namespace

// ---------------------------------------------------------------------------
// Biquad (RBJ cookbook, bilinear transform)
// ---------------------------------------------------------------------------
Biquad Biquad::butterworthLowpass(double fc, double fs, double q) {
    const double w0 = 2.0 * kPi * fc / fs;
    const double c = std::cos(w0), alpha = std::sin(w0) / (2.0 * q);
    const double a0 = 1.0 + alpha;
    Biquad b;
    b.b0 = (1.0 - c) / 2.0 / a0;
    b.b1 = (1.0 - c) / a0;
    b.b2 = b.b0;
    b.a1 = -2.0 * c / a0;
    b.a2 = (1.0 - alpha) / a0;
    return b;
}

Biquad Biquad::butterworthHighpass(double fc, double fs, double q) {
    const double w0 = 2.0 * kPi * fc / fs;
    const double c = std::cos(w0), alpha = std::sin(w0) / (2.0 * q);
    const double a0 = 1.0 + alpha;
    Biquad b;
    b.b0 = (1.0 + c) / 2.0 / a0;
    b.b1 = -(1.0 + c) / a0;
    b.b2 = b.b0;
    b.a1 = -2.0 * c / a0;
    b.a2 = (1.0 - alpha) / a0;
    return b;
}

void Biquad::process(std::vector<double>& x) const {
    if (x.empty()) return;
    // Steady-state initial conditions for a constant input equal to x[0].
    const double den = 1.0 + a1 + a2;
    const double dc = std::abs(den) > 1e-12 ? (b0 + b1 + b2) / den : 0.0;
    const double x0 = x[0], y0 = dc * x0;
    double z2 = b2 * x0 - a2 * y0;
    double z1 = b1 * x0 - a1 * y0 + z2;
    for (double& v : x) {
        const double in = v;
        const double out = b0 * in + z1;
        z1 = b1 * in - a1 * out + z2;
        z2 = b2 * in - a2 * out;
        v = out;
    }
}

// ---------------------------------------------------------------------------
// SignalProcessor — input side
// ---------------------------------------------------------------------------
SignalProcessor::SignalProcessor(const SignalConfig& cfg)
    : cfg_(cfg),
      buffer_(std::max<std::size_t>(cfg.bufferSize, 64)),
      respBuffer_(std::max<std::size_t>(cfg.respBufferSize, cfg.bufferSize)) {
    cfg_.minSeconds = std::max(cfg_.minSeconds, 3.0);
}

void SignalProcessor::reset() {
    buffer_.clear();
    respBuffer_.clear();
    smoothedResp_ = 0.0;
    respHistory_.clear();
    metrics_ = SignalMetrics{};
    paused_ = false;
    timeOffset_ = 0.0;
    levelOffset_.fill(cv::Vec3d(0, 0, 0));
    lastRaw_.fill(cv::Vec3d(0, 0, 0));
    haveRaw_.fill(false);
    bgOffset_ = bgLast_ = cv::Vec3d(0, 0, 0);
    haveBg_ = false;
    scalarInput_ = false;
    lastT_ = -1.0;
    smoothedBpm_ = 0.0;
    disagreeSince_ = -1.0;
    ++segment_;           // segment ids stay monotonic across resets
    lastBeatT_ = -1e9;
    segmentStartT_ = -1e9;
}

void SignalProcessor::pause(double t) {
    if (!paused_) {
        paused_ = true;
        pauseStart_ = t;
        ++segment_;       // beats after the pause must never pair with beats before it
    }
    metrics_.paused = true;
}

void SignalProcessor::push(double value, double t) {
    push(cv::Vec3d(value, value, value), t);
    scalarInput_ = true;
}

void SignalProcessor::push(const cv::Vec3d& bgr, double t) {
    FrameSample s;
    s.roi[kForehead] = RoiSample{bgr, 1.0, true};
    push(s, t);
}

void SignalProcessor::push(const FrameSample& s, double t) {
    if (!std::isfinite(t)) return;

    bool realigned = false;  // the splice below already re-aligned the DC level
    if (paused_) {
        realigned = true;
        paused_ = false;
        const double gap = t - pauseStart_;
        if (buffer_.empty() || gap > cfg_.maxGapSec) {
            reset();  // long dropout: old history is no longer phase-coherent
        } else {
            // Short dropout: splice the timeline (remove the gap) and re-align each
            // region's DC level so the detrender does not see a step.
            double dt = 1.0 / 30.0;
            if (buffer_.size() > 1)
                dt = (buffer_.back().t - buffer_[0].t) / static_cast<double>(buffer_.size() - 1);
            timeOffset_ = t - (buffer_.back().t + dt);
            for (int k = 0; k < kRoiCount; ++k)
                if (s.roi[k].valid && haveRaw_[k]) levelOffset_[k] += lastRaw_[k] - s.roi[k].bgr;
            if (s.hasBackground && haveBg_) bgOffset_ += bgLast_ - s.background;
        }
        segmentStartT_ = t - timeOffset_;
    }

    const double te = t - timeOffset_;
    if (!buffer_.empty() && te <= buffer_.back().t) return;  // non-monotonic clock

    Sample smp;
    smp.t = te;
    const Sample* prev = buffer_.empty() ? nullptr : &buffer_.back();
    for (int k = 0; k < kRoiCount; ++k) {
        const RoiSample& r = s.roi[k];
        const bool ok = r.valid && std::isfinite(r.bgr[0]) && std::isfinite(r.bgr[1]) && std::isfinite(r.bgr[2]);
        if (ok) {
            if (r.moved && !realigned && haveRaw_[k]) levelOffset_[k] += lastRaw_[k] - r.bgr;
            smp.c[k] = r.bgr + levelOffset_[k];
            lastRaw_[k] = r.bgr;
            haveRaw_[k] = true;
        } else {
            smp.c[k] = prev ? prev->c[k] : cv::Vec3d(0, 0, 0);  // hold; region marked invalid
        }
        smp.ok[k] = ok;
        smp.skin[k] = ok ? static_cast<float>(r.skinFraction) : 0.f;
    }
    if (s.hasBackground) {
        smp.bg = s.background + bgOffset_;
        smp.bgOk = true;
        bgLast_ = s.background;
        haveBg_ = true;
    } else if (prev) {
        smp.bg = prev->bg;
    }
    buffer_.push(smp);
    respBuffer_.push(smp);
    lastT_ = t;
    metrics_.paused = false;
}

bool SignalProcessor::skinMean(const cv::Mat& bgr, const std::vector<cv::Rect>& rois, cv::Vec3d& out) {
    cv::Vec3d sumMasked(0, 0, 0), sumAll(0, 0, 0);
    double nMasked = 0.0, nAll = 0.0;
    cv::Mat ycc, mask;
    for (const cv::Rect& r0 : rois) {
        const cv::Rect r = r0 & cv::Rect(0, 0, bgr.cols, bgr.rows);
        if (r.area() <= 0) continue;
        const cv::Mat roi = bgr(r);
        cv::cvtColor(roi, ycc, cv::COLOR_BGR2YCrCb);
        cv::inRange(ycc, cv::Scalar(40, 133, 77), cv::Scalar(240, 180, 130), mask);
        const double nm = cv::countNonZero(mask);
        const cv::Scalar mAll = cv::mean(roi);
        sumAll += cv::Vec3d(mAll[0], mAll[1], mAll[2]) * static_cast<double>(r.area());
        nAll += r.area();
        if (nm > 0) {
            const cv::Scalar mm = cv::mean(roi, mask);
            sumMasked += cv::Vec3d(mm[0], mm[1], mm[2]) * nm;
            nMasked += nm;
        }
    }
    if (nAll <= 0.0) return false;
    out = nMasked >= 0.25 * nAll ? sumMasked / nMasked : sumAll / nAll;
    return true;
}

FrameSample SignalProcessor::sampleFrame(const cv::Mat& bgr, const std::array<cv::Rect, kRoiCount>& rois,
                                         const std::vector<cv::Rect>& background) {
    FrameSample out;
    const cv::Rect frame(0, 0, bgr.cols, bgr.rows);
    cv::Mat ycc, mask;
    for (int k = 0; k < kRoiCount; ++k) {
        const cv::Rect r = rois[k] & frame;
        if (r.area() < 16) continue;
        const cv::Mat roi = bgr(r);
        // Broad skin chroma gate (light to dark skin), excluding deep shadow (Y < 40)
        // and clipped highlights (Y > 240) whose pulsatile modulation is lost.
        cv::cvtColor(roi, ycc, cv::COLOR_BGR2YCrCb);
        cv::inRange(ycc, cv::Scalar(40, 133, 77), cv::Scalar(240, 180, 130), mask);
        const double frac = static_cast<double>(cv::countNonZero(mask)) / r.area();
        const cv::Scalar m = frac >= 0.25 ? cv::mean(roi, mask) : cv::mean(roi);
        out.roi[k] = RoiSample{cv::Vec3d(m[0], m[1], m[2]), frac, true};
    }
    cv::Vec3d sum(0, 0, 0);
    double area = 0.0;
    for (const cv::Rect& b0 : background) {
        const cv::Rect b = b0 & frame;
        if (b.area() < 64) continue;
        const cv::Scalar m = cv::mean(bgr(b));
        sum += cv::Vec3d(m[0], m[1], m[2]) * static_cast<double>(b.area());
        area += b.area();
    }
    if (area > 0.0) {
        out.background = sum / area;
        out.hasBackground = true;
    }
    return out;
}

template <typename Getter>
bool SignalProcessor::resample(const CircularBuffer<Sample>& buf, Getter get, std::vector<double>& out,
                               double& fs) {
    const std::size_t n = buf.size();
    if (n < 2) return false;
    const double t0 = buf[0].t, t1 = buf.back().t;
    const double span = t1 - t0;
    if (span <= 0.0) return false;
    fs = static_cast<double>(n - 1) / span;  // measured, never assumed
    out.resize(n);
    const double step = span / static_cast<double>(n - 1);
    std::size_t j = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double t = t0 + step * static_cast<double>(i);
        while (j + 2 < n && buf[j + 1].t < t) ++j;
        const Sample& a = buf[j];
        const Sample& b = buf[std::min(j + 1, n - 1)];
        const double dt = b.t - a.t;
        const double u = dt > 0.0 ? std::clamp((t - a.t) / dt, 0.0, 1.0) : 0.0;
        const double va = get(a), vb = get(b);
        out[i] = va + u * (vb - va);
    }
    return true;
}

double SignalProcessor::rms(const std::vector<double>& x) {
    if (x.empty()) return 0.0;
    double s = 0.0;
    for (double v : x) s += v * v;
    return std::sqrt(s / static_cast<double>(x.size()));
}

void SignalProcessor::normalise(std::vector<double>& x) {
    if (x.empty()) return;
    const double m = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
    for (double& v : x) v -= m;
    const double r = rms(x);
    if (r > 1e-15)
        for (double& v : x) v /= r;
}

std::vector<double> SignalProcessor::pos(const std::vector<double>& b, const std::vector<double>& g,
                                         const std::vector<double>& r, int window) {
    const int n = static_cast<int>(g.size());
    std::vector<double> h(static_cast<std::size_t>(n), 0.0);
    const int l = std::clamp(window, 4, n);
    std::vector<double> s1(static_cast<std::size_t>(l)), s2(static_cast<std::size_t>(l));
    for (int start = 0; start + l <= n; ++start) {
        // Temporal normalisation: divide each channel by its mean over the window.
        double mb = 0, mg = 0, mr = 0;
        for (int i = start; i < start + l; ++i) { mb += b[i]; mg += g[i]; mr += r[i]; }
        mb /= l; mg /= l; mr /= l;
        if (mb <= 1e-6 || mg <= 1e-6 || mr <= 1e-6) continue;
        // Projection onto the plane orthogonal to the skin tone:
        //   S1 = Gn − Bn,  S2 = Gn + Bn − 2·Rn   (P = [[0, 1, −1], [−2, 1, 1]] on [R, G, B])
        double m1 = 0, m2 = 0;
        for (int i = 0; i < l; ++i) {
            const double rn = r[start + i] / mr, gn = g[start + i] / mg, bn = b[start + i] / mb;
            s1[i] = gn - bn;
            s2[i] = gn + bn - 2.0 * rn;
            m1 += s1[i]; m2 += s2[i];
        }
        m1 /= l; m2 /= l;
        double v1 = 0, v2 = 0;
        for (int i = 0; i < l; ++i) { v1 += (s1[i] - m1) * (s1[i] - m1); v2 += (s2[i] - m2) * (s2[i] - m2); }
        // Alpha tuning: combine the two projections so the residual distortion cancels.
        const double alpha = v2 > 1e-18 ? std::sqrt(v1 / v2) : 0.0;
        double mh = 0;
        for (int i = 0; i < l; ++i) { s1[i] += alpha * s2[i]; mh += s1[i]; }
        mh /= l;
        for (int i = 0; i < l; ++i) h[static_cast<std::size_t>(start + i)] += s1[i] - mh;  // overlap-add
    }
    return h;
}

void SignalProcessor::detrend(std::vector<double>& x, int window) {
    const int n = static_cast<int>(x.size());
    if (n == 0) return;
    window = std::max(3, window | 1);
    const int half = window / 2;
    // Prefix sums → O(n) centred moving average with shrinking edges.
    std::vector<double> ps(static_cast<std::size_t>(n) + 1, 0.0);
    for (int i = 0; i < n; ++i) ps[i + 1] = ps[i] + x[i];
    std::vector<double> y(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const int a = std::max(0, i - half), b = std::min(n, i + half + 1);
        y[i] = x[i] - (ps[b] - ps[a]) / static_cast<double>(b - a);
    }
    const double mean = std::accumulate(y.begin(), y.end(), 0.0) / n;
    for (int i = 0; i < n; ++i) x[i] = y[i] - mean;
}

void SignalProcessor::removeLinearTrend(std::vector<double>& x) {
    const double n = static_cast<double>(x.size());
    if (x.size() < 3) return;
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double xi = static_cast<double>(i);
        sx += xi; sy += x[i]; sxx += xi * xi; sxy += xi * x[i];
    }
    const double den = n * sxx - sx * sx;
    const double slope = std::abs(den) > 1e-12 ? (n * sxy - sx * sy) / den : 0.0;
    const double icpt = (sy - slope * sx) / n;
    for (std::size_t i = 0; i < x.size(); ++i) x[i] -= icpt + slope * static_cast<double>(i);
}

void SignalProcessor::bandpass(std::vector<double>& x, double fs, double loHz, double hiHz,
                               int nSections) {
    const int n = static_cast<int>(x.size());
    if (n < 8) return;
    const double nyq = 0.5 * fs;
    const double lo = std::min(loHz, 0.4 * nyq);
    const double hi = std::min(hiHz, 0.9 * nyq);

    // Butterworth order 2N realised as N cascaded biquads with pole-pair Qs.
    const int sections = std::max(1, nSections);
    const int order = 2 * sections;
    std::vector<Biquad> chain;
    for (int k = 0; k < sections; ++k) {
        const double q = 1.0 / (2.0 * std::sin((2.0 * k + 1.0) * kPi / (2.0 * order)));
        chain.push_back(Biquad::butterworthHighpass(lo, fs, q));
        chain.push_back(Biquad::butterworthLowpass(hi, fs, q));
    }

    // Zero-phase filtering (filtfilt) with odd-reflection padding.
    const int pad = std::min(n - 1, static_cast<int>(3.0 * fs / std::max(0.2, lo / 0.75)));
    std::vector<double> ext(static_cast<std::size_t>(n + 2 * pad));
    for (int i = 0; i < pad; ++i) ext[i] = 2.0 * x[0] - x[pad - i];
    for (int i = 0; i < n; ++i) ext[pad + i] = x[i];
    for (int i = 0; i < pad; ++i) ext[pad + n + i] = 2.0 * x[n - 1] - x[n - 2 - i];

    for (const Biquad& b : chain) b.process(ext);
    std::reverse(ext.begin(), ext.end());
    for (const Biquad& b : chain) b.process(ext);
    std::reverse(ext.begin(), ext.end());

    for (int i = 0; i < n; ++i) x[i] = ext[pad + i];
}

bool SignalProcessor::spectrum(const std::vector<double>& x, double fs, double lo, double hi,
                               double lobeHz, int nfftMin, Spectrum& out, double refLo, double refHi,
                               double prevHz, double trackSigmaHz, double trackFloor) {
    const int n = static_cast<int>(x.size());
    if (n < 16) return false;
    const int nfft = cv::getOptimalDFTSize(std::max(nfftMin, n));
    cv::Mat src = cv::Mat::zeros(1, nfft, CV_64F);
    double* s = src.ptr<double>();
    for (int i = 0; i < n; ++i) {
        const double w = 0.54 - 0.46 * std::cos(2.0 * kPi * i / (n - 1));  // Hamming
        s[i] = x[i] * w;
    }
    cv::Mat spec;
    cv::dft(src, spec, cv::DFT_COMPLEX_OUTPUT);
    const cv::Vec2d* c = spec.ptr<cv::Vec2d>();

    const int half = nfft / 2;
    std::vector<double> p(static_cast<std::size_t>(half));
    for (int k = 0; k < half; ++k) p[k] = c[k][0] * c[k][0] + c[k][1] * c[k][1];

    const double df = fs / nfft;
    const int kLo = std::max(1, static_cast<int>(std::ceil(lo / df)));
    const int kHi = std::min(half - 2, static_cast<int>(std::floor(hi / df)));
    if (kHi <= kLo + 2) return false;

    // Candidates: true local maxima of the *full* spectrum inside [lo, hi]. A bin at the
    // lower edge whose left neighbour (outside the band) is larger is leakage from the
    // baseline, not a pulse peak, and is therefore never selected.
    int kMax = -1, kRaw = -1;
    double best = -1.0;
    for (int k = kLo; k <= kHi; ++k) {
        if (!(p[k] > p[k - 1] && p[k] >= p[k + 1])) continue;
        if (kRaw < 0 || p[k] > p[kRaw]) kRaw = k;
        double w = 1.0;
        if (prevHz > 0.0 && trackFloor < 1.0) {
            const double d = (k * df - prevHz) / std::max(1e-6, trackSigmaHz);
            w = trackFloor + (1.0 - trackFloor) * std::exp(-d * d);
        }
        if (p[k] * w > best) { best = p[k] * w; kMax = k; }
    }
    out.edgeOnly = kMax < 0;
    if (kMax < 0) return false;
    out.rawPeakHz = kRaw * df;

    // Parabolic interpolation on the magnitude for sub-bin accuracy.
    double delta = 0.0;
    {
        const double a = std::sqrt(p[kMax - 1]), b = std::sqrt(p[kMax]), cc = std::sqrt(p[kMax + 1]);
        const double den = a - 2.0 * b + cc;
        if (std::abs(den) > 1e-18) delta = std::clamp(0.5 * (a - cc) / den, -0.5, 0.5);
    }
    out.peakHz = (kMax + delta) * df;

    // Power ratio: energy of the peak lobe vs. the total energy of the band.
    double lobe = 0.0, total = 0.0;
    for (int k = kLo; k <= kHi; ++k) {
        total += p[k];
        if (std::abs(k * df - out.peakHz) <= lobeHz) lobe += p[k];
    }
    if (total <= 1e-18) return false;
    out.ratio = std::clamp(lobe / total, 0.0, 1.0);
    // Normalised spectral entropy of the band: 0 = single line, 1 = flat (noise).
    double h = 0.0;
    for (int k = kLo; k <= kHi; ++k) {
        const double q = p[k] / total;
        if (q > 1e-15) h -= q * std::log(q);
    }
    out.entropy = std::clamp(h / std::log(static_cast<double>(kHi - kLo + 1)), 0.0, 1.0);
    const double rest = std::max(total - lobe, 1e-18);
    out.snrDb = 10.0 * std::log10(std::max(lobe, 1e-18) / rest);

    out.peakToRefDb = 0.0;
    if (refHi > refLo) {
        const int rLo = std::max(1, static_cast<int>(std::ceil(refLo / df)));
        const int rHi = std::min(half - 1, static_cast<int>(std::floor(refHi / df)));
        if (rHi > rLo) {
            std::vector<double> ref(p.begin() + rLo, p.begin() + rHi + 1);
            const double med = std::max(median(std::move(ref)), 1e-18);
            out.peakToRefDb = 10.0 * std::log10(std::max(p[kMax], 1e-18) / med);
        }
    }
    return true;
}

void SignalProcessor::computeHrv(const std::vector<double>& x, double fs, double hrHz,
                                 std::vector<double>& peaks) {
    peaks.clear();
    const int n = static_cast<int>(x.size());
    if (n < 16 || hrHz <= 0.0) return;

    double sd = 0.0;
    for (double v : x) sd += v * v;
    sd = std::sqrt(sd / n);
    const double thr = 0.3 * sd;
    const int minDist = std::max(2, static_cast<int>(0.6 * fs / hrHz));

    // Candidate local maxima, then non-maximum suppression by amplitude.
    std::vector<int> cand;
    for (int i = 2; i < n - 2; ++i)
        if (x[i] > thr && x[i] > x[i - 1] && x[i] >= x[i + 1]) cand.push_back(i);
    std::sort(cand.begin(), cand.end(), [&](int a, int b) { return x[a] > x[b]; });
    std::vector<int> kept;
    for (int i : cand) {
        bool ok = true;
        for (int k : kept)
            if (std::abs(k - i) < minDist) { ok = false; break; }
        if (ok) kept.push_back(i);
    }
    std::sort(kept.begin(), kept.end());

    for (int i : kept) {
        const double a = x[i - 1], b = x[i], c = x[i + 1];
        const double den = a - 2.0 * b + c;
        const double d = std::abs(den) > 1e-18 ? std::clamp(0.5 * (a - c) / den, -0.5, 0.5) : 0.0;
        peaks.push_back(i + d);
    }
    if (peaks.size() < 3) {
        metrics_.rmssdMs = 0.0;
        metrics_.meanIbiMs = 0.0;
        metrics_.beatCount = static_cast<int>(peaks.size());
        return;
    }

    // Inter-beat intervals with physiological + consistency gating.
    std::vector<double> ibi;
    for (std::size_t i = 1; i < peaks.size(); ++i) ibi.push_back((peaks[i] - peaks[i - 1]) * 1000.0 / fs);
    const double ref = 0.5 * (median(ibi) + 60000.0 / (hrHz * 60.0));
    const double ibiMin = 1000.0 / cfg_.highHz, ibiMax = 1000.0 / cfg_.lowHz;
    std::vector<char> good(ibi.size());
    std::vector<double> accepted;
    for (std::size_t i = 0; i < ibi.size(); ++i) {
        good[i] = ibi[i] >= ibiMin && ibi[i] <= ibiMax && std::abs(ibi[i] - ref) <= 0.25 * ref;
        if (good[i]) accepted.push_back(ibi[i]);
    }

    double ss = 0.0;
    int nd = 0;
    for (std::size_t i = 1; i < ibi.size(); ++i) {
        const double d = ibi[i] - ibi[i - 1];
        // Malik criterion: successive IBIs differing > 20 % are treated as artefacts.
        if (good[i] && good[i - 1] && std::abs(d) <= 0.2 * ibi[i - 1]) {
            ss += d * d;
            ++nd;
        }
    }
    metrics_.beatCount = static_cast<int>(peaks.size());
    metrics_.meanIbiMs = accepted.empty()
                             ? 0.0
                             : std::accumulate(accepted.begin(), accepted.end(), 0.0) / accepted.size();
    if (nd >= 3) {
        const double rmssd = std::sqrt(ss / nd);
        metrics_.rmssdMs = metrics_.rmssdMs > 0.0 ? metrics_.rmssdMs + 0.2 * (rmssd - metrics_.rmssdMs)
                                                  : rmssd;
    }
}

void SignalProcessor::processRespiration() {
    const std::size_t n = respBuffer_.size();
    if (n < 64) {
        metrics_.respValid = false;
        metrics_.respProgress = 0.0;
        return;
    }
    const double span = respBuffer_.back().t - respBuffer_[0].t;
    metrics_.respProgress = std::clamp(span / cfg_.respMinSeconds, 0.0, 1.0);
    if (span < cfg_.respMinSeconds) {
        metrics_.respValid = false;
        return;
    }

    std::vector<double> x;
    double fs = 0.0;
    if (!resample(respBuffer_, [](const Sample& s) { return s.c[kForehead][1]; }, x, fs) ||
        fs < 2.0 * cfg_.respHighHz + 1.0) {
        metrics_.respValid = false;
        return;
    }
    // Relative baseline modulation. No short-window detrending here: that would
    // remove exactly the 0.15–0.4 Hz content we want. Only the linear trend
    // goes (reduces filter edge transients); slower drift is rejected by the
    // 0.15 Hz high-pass edge of the band-pass.
    const double mean = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
    if (mean > 1e-6)
        for (double& v : x) v = v / mean - 1.0;
    removeLinearTrend(x);

    // Peak-vs-floor test on the *unfiltered* spectrum: genuine breathing stands well above the
    // neighbouring 0.45–0.70 Hz floor, whereas the peak of band-limited noise does not.
    Spectrum raw;
    const bool rawOk = spectrum(x, fs, cfg_.respLowHz, cfg_.respHighHz, cfg_.respLobeHz, 8192, raw,
                                cfg_.respRefLowHz, cfg_.respRefHighHz);
    bandpass(x, fs, cfg_.respLowHz, cfg_.respHighHz, 2);

    Spectrum sp;
    if (!spectrum(x, fs, cfg_.respLowHz, cfg_.respHighHz, cfg_.respLobeHz, 8192, sp)) {
        metrics_.respValid = false;
        return;
    }
    const double fftBrpm = sp.peakHz * 60.0;

    // Zero-crossing cross-check: median interval between rising crossings of a
    // hysteresis-gated version of the band-passed signal.
    double rms = 0.0;
    for (double v : x) rms += v * v;
    rms = std::sqrt(rms / static_cast<double>(x.size()));
    std::vector<double> crossings;
    bool armed = false;
    for (std::size_t i = 1; i < x.size(); ++i) {
        if (x[i] < -0.25 * rms) armed = true;
        if (armed && x[i - 1] < 0.0 && x[i] >= 0.0) {
            const double f = x[i - 1] / (x[i - 1] - x[i]);  // sub-sample interpolation
            crossings.push_back((static_cast<double>(i - 1) + f) / fs);
            armed = false;
        }
    }
    double zcBrpm = 0.0;
    if (crossings.size() >= 3) {
        std::vector<double> iv;
        for (std::size_t i = 1; i < crossings.size(); ++i) iv.push_back(crossings[i] - crossings[i - 1]);
        zcBrpm = 60.0 / median(iv);
    }

    // The breathing band is narrow (0.25 Hz): a lobe of ±respLobeHz covers `chance` of it even for
    // pure noise, so the ratio is chance-corrected before normalisation.
    const double chance = std::min(0.95, 2.0 * cfg_.respLobeHz / (cfg_.respHighHz - cfg_.respLowHz));
    double conf = std::clamp((sp.ratio - chance) / std::max(1e-6, cfg_.respConfFull - chance), 0.0, 1.0);
    const bool agree = zcBrpm > 0.0 && std::abs(zcBrpm - fftBrpm) <= 3.0;
    if (!agree) conf *= 0.5;  // two independent methods disagree → less trustworthy
    metrics_.respConfidence = std::clamp(conf, 0.0, 1.0);

    // Temporal stability: a genuine breathing rhythm does not jump between estimates,
    // whereas the peak of a noise-only spectrum wanders. Require the last ~5 s of FFT
    // estimates to stay within ±1.5 BrPM of their median.
    const double tNow = respBuffer_.back().t;
    respHistory_.emplace_back(tNow, fftBrpm);
    while (!respHistory_.empty() && tNow - respHistory_.front().first > cfg_.respStabilitySec)
        respHistory_.pop_front();
    bool stable = tNow - respHistory_.front().first >= 0.9 * cfg_.respStabilitySec;
    if (stable) {
        std::vector<double> vals;
        for (const auto& e : respHistory_) vals.push_back(e.second);
        const double med = median(vals);
        for (double v : vals)
            if (std::abs(v - med) > 1.5) { stable = false; break; }
    }

    const bool standsOut = rawOk && raw.peakToRefDb >= cfg_.respMinPeakDb;
    if (agree && stable && standsOut && metrics_.respConfidence >= cfg_.respMinConfidence) {
        const double a = std::clamp(cfg_.respSmoothing * (0.3 + 0.7 * metrics_.respConfidence), 0.02, 1.0);
        smoothedResp_ = smoothedResp_ <= 0.0 ? fftBrpm : smoothedResp_ + a * (fftBrpm - smoothedResp_);
        metrics_.respRate = smoothedResp_;
        metrics_.respValid = true;
    } else {
        metrics_.respValid = false;
    }
}


// ---------------------------------------------------------------------------
// SignalProcessor — multi-ROI processing
// ---------------------------------------------------------------------------
std::vector<double> SignalProcessor::regionPulse(int k, double fs) const {
    std::vector<double> b, g, r;
    double f = 0.0;
    resample(buffer_, [k](const Sample& s) { return s.c[k][1]; }, g, f);
    if (cfg_.method == PulseMethod::POS && !scalarInput_) {
        resample(buffer_, [k](const Sample& s) { return s.c[k][0]; }, b, f);
        resample(buffer_, [k](const Sample& s) { return s.c[k][2]; }, r, f);
        std::vector<double> h = pos(b, g, r, static_cast<int>(std::lround(cfg_.posWindowSec * fs)));
        // Skin reflectance *drops* when blood volume rises, so POS output is inverted
        // blood volume; flip it so systolic peaks are maxima (needed for morphology).
        for (double& v : h) v = -v;
        return h;
    }
    const double mean = std::accumulate(g.begin(), g.end(), 0.0) / static_cast<double>(g.size());
    if (mean > 1e-6)
        for (double& v : g) v = 1.0 - v / mean;  // same blood-volume orientation
    return g;
}

double SignalProcessor::sqiOf(const Spectrum& sp) const {
    const double snrTerm =
        std::clamp((sp.ratio - cfg_.sqiRatioNoise) / (cfg_.sqiRatioClean - cfg_.sqiRatioNoise), 0.0, 1.0);
    const double entTerm =
        std::clamp((cfg_.sqiEntropyNoise - sp.entropy) / (cfg_.sqiEntropyNoise - cfg_.sqiEntropyClean), 0.0, 1.0);
    return std::clamp(0.5 * snrTerm + 0.5 * entTerm, 0.0, 1.0);
}

void SignalProcessor::processRatio(const std::array<double, kRoiCount>& w, double f0) {
    metrics_.ratioValid = false;
    double acR = 0.0, acB = 0.0, wsum = 0.0, fs = 0.0;
    const double lo = std::max(cfg_.lowHz, f0 - 0.25), hi = std::min(cfg_.highHz, f0 + 0.25);
    for (int k = 0; k < kRoiCount; ++k) {
        if (w[k] <= 0.0) continue;
        std::vector<double> r, b;
        resample(buffer_, [k](const Sample& s) { return s.c[k][2]; }, r, fs);
        resample(buffer_, [k](const Sample& s) { return s.c[k][0]; }, b, fs);
        auto acdc = [&](std::vector<double>& x) {
            const double dc = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
            if (dc <= 1e-6) return 0.0;
            for (double& v : x) v = v / dc - 1.0;  // AC normalised by DC
            removeLinearTrend(x);
            bandpass(x, fs, lo, hi, 2);            // pulsatile component around the HR only
            return rms(x);
        };
        acR += w[k] * acdc(r);
        acB += w[k] * acdc(b);
        wsum += w[k];
    }
    if (wsum <= 0.0 || acB <= 1e-12 || acR <= 1e-12) return;
    metrics_.acDcRed = acR / wsum;
    metrics_.acDcBlue = acB / wsum;
    metrics_.ratioRB = metrics_.acDcRed / metrics_.acDcBlue;
    metrics_.ratioValid = true;
}

void SignalProcessor::processBackground(double f0, const std::vector<double>& faceGreenRel, double fs) {
    metrics_.backgroundChecked = false;
    metrics_.ambientSuspect = false;
    const std::size_t n = buffer_.size();
    std::size_t okCount = 0;
    for (std::size_t i = 0; i < n; ++i) okCount += buffer_[i].bgOk ? 1 : 0;
    if (okCount < static_cast<std::size_t>(0.9 * static_cast<double>(n))) return;

    std::vector<double> g;
    double f = 0.0;
    resample(buffer_, [](const Sample& s) { return s.bg[1]; }, g, f);
    const double mean = std::accumulate(g.begin(), g.end(), 0.0) / static_cast<double>(g.size());
    if (mean <= 1e-6) return;
    for (double& v : g) v = v / mean - 1.0;
    detrend(g, static_cast<int>(std::lround(cfg_.detrendWindowSec * fs)));
    bandpass(g, fs, cfg_.lowHz, cfg_.highHz, cfg_.filterOrderSections);

    // Look for background energy at the *face* pulse frequency (strong prior on f0).
    Spectrum sp;
    metrics_.backgroundChecked = true;
    if (!spectrum(g, fs, cfg_.lowHz, cfg_.highHz, cfg_.confLobeHz, cfg_.dftMinSize, sp, 0.0, 0.0, f0, 0.05, 0.0)) {
        metrics_.backgroundRatio = 0.0;
        return;
    }
    metrics_.backgroundRatio = std::abs(sp.peakHz - f0) <= 0.1 ? sp.ratio : 0.0;
    const double relBg = rms(g), relFace = rms(faceGreenRel);
    metrics_.ambientSuspect = metrics_.backgroundRatio >= cfg_.ambientMinRatio &&
                              relBg >= cfg_.ambientMinRelAmp * std::max(relFace, 1e-12);
}

void SignalProcessor::emitBeats(const std::vector<double>& peaks, double fs) {
    const double t0 = buffer_[0].t, tEnd = buffer_.back().t;
    const double newest = tEnd - cfg_.beatConfirmSec;      // filter edge settled before this
    const double oldest = newest - 1.0;                    // only the "just confirmed" zone
    for (double p : peaks) {
        const double tb = t0 + p / fs;
        if (tb > newest || tb < oldest) continue;
        if (tb < segmentStartT_ + 0.5) continue;           // too close to a splice
        if (tb <= lastBeatT_ + 0.25) continue;             // already emitted (allows jitter)
        metrics_.newBeats.push_back(BeatEvent{tb, segment_});
        lastBeatT_ = tb;
    }
}

const SignalMetrics& SignalProcessor::process() {
    metrics_.paused = paused_;
    metrics_.newBeats.clear();
    if (paused_) return metrics_;

    const std::size_t n = buffer_.size();
    const double span = n > 1 ? buffer_.back().t - buffer_[0].t : 0.0;
    metrics_.bufferFill = std::clamp(span / cfg_.minSeconds, 0.0, 1.0);
    if (n < 32) {
        metrics_.valid = false;
        metrics_.waveform.clear();
        metrics_.beatMarkers.clear();
        metrics_.morphWave.clear();
        return metrics_;
    }

    // 1. Sampling rate measured from the real timestamps on every call.
    std::vector<double> probe;
    double fs = 0.0;
    if (!resample(buffer_, [](const Sample& s) { return s.c[0][1]; }, probe, fs) || fs < 2.0 * cfg_.highHz + 0.5) {
        metrics_.valid = false;
        return metrics_;
    }
    metrics_.sampleRate = fs;
    const bool usePos = cfg_.method == PulseMethod::POS && !scalarInput_;
    metrics_.method = usePos ? PulseMethod::POS : PulseMethod::Green;
    const int dw = static_cast<int>(std::lround(cfg_.detrendWindowSec * fs));
    const double morphHi = std::min(cfg_.morphHighHz, 0.45 * fs);

    // 2. Per-region pulse, quality and band-limited versions.
    std::array<std::vector<double>, kRoiCount> narrow, wide;
    std::array<double, kRoiCount> sqi{}, w{};
    int bestUsable = -1;
    for (int k = 0; k < kRoiCount; ++k) {
        std::size_t ok = 0;
        double skin = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            if (buffer_[i].ok[k]) { ++ok; skin += buffer_[i].skin[k]; }
        }
        const bool usable = ok >= static_cast<std::size_t>(cfg_.minRoiValidFraction * static_cast<double>(n)) &&
                            ok > 0 && skin / static_cast<double>(ok) >= cfg_.minSkinFraction;
        if (!usable) continue;
        std::vector<double> raw = regionPulse(k, fs);
        detrend(raw, dw);
        wide[k] = raw;
        bandpass(wide[k], fs, cfg_.morphLowHz, morphHi, 2);
        narrow[k] = std::move(raw);
        bandpass(narrow[k], fs, cfg_.lowHz, cfg_.highHz, cfg_.filterOrderSections);
        Spectrum sp;
        sqi[k] = spectrum(narrow[k], fs, cfg_.lowHz, cfg_.highHz, cfg_.confLobeHz, cfg_.dftMinSize, sp) ? sqiOf(sp) : 0.0;
        if (bestUsable < 0 || sqi[k] > sqi[bestUsable]) bestUsable = k;
    }
    metrics_.roiSqi = sqi;
    if (bestUsable < 0) {
        metrics_.valid = false;
        metrics_.confidence = 0.0;
        metrics_.sqi = 0.0;
        metrics_.roiWeight.fill(0.0);
        return metrics_;
    }

    // 3. SQI-weighted fusion. A weak region is discarded only when a better region passes
    //    the minimum SQI; if all regions are weak, all are averaged (√N noise reduction).
    //    Non-skin / untracked regions were already excluded above.
    // "Clearly good" = at least twice the minimum; below that SQI differences are mostly
    // estimation noise, so no region is singled out.
    const bool anyGood = sqi[bestUsable] >= 2.0 * cfg_.minRoiSqi;
    double wsum = 0.0;
    for (int k = 0; k < kRoiCount; ++k) {
        // Discard only regions that are both below the minimum and much worse than the best
        // (comparable regions are averaged, the SQI² weights still favour the better one).
        if (narrow[k].empty() || (anyGood && sqi[k] < cfg_.minRoiSqi && sqi[k] < 0.5 * sqi[bestUsable])) continue;
        const double q = std::max(sqi[k], 0.12);  // floor flattens weights in the weak regime
        w[k] = q * q;
        wsum += w[k];
    }
    for (double& v : w) v /= wsum;
    metrics_.roiWeight = w;

    std::vector<double> x(n, 0.0), morph(n, 0.0);
    for (int k = 0; k < kRoiCount; ++k) {
        if (w[k] <= 0.0) continue;
        normalise(narrow[k]);
        normalise(wide[k]);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] += w[k] * narrow[k][i];
            morph[i] += w[k] * wide[k][i];
        }
    }

    // 4. Heart rate from the fused spectrum (true local maxima + tracking prior).
    std::vector<double> peaks;
    double peakHz = 0.0;
    if (span >= cfg_.minSeconds) {
        Spectrum sp;
        const double prevHz = smoothedBpm_ > 0.0 ? smoothedBpm_ / 60.0 : 0.0;
        if (spectrum(x, fs, cfg_.lowHz, cfg_.highHz, cfg_.confLobeHz, cfg_.dftMinSize, sp, 0.0, 0.0,
                     prevHz, cfg_.trackSigmaBpm / 60.0, cfg_.trackFloor)) {
            peakHz = sp.peakHz;
            const double tNow = buffer_.back().t;
            if (std::abs(sp.rawPeakHz - peakHz) * 60.0 > cfg_.trackSigmaBpm) {
                if (disagreeSince_ < 0.0) disagreeSince_ = tNow;
                if (tNow - disagreeSince_ >= cfg_.reacquireSec) {
                    Spectrum fresh;
                    if (spectrum(x, fs, cfg_.lowHz, cfg_.highHz, cfg_.confLobeHz, cfg_.dftMinSize, fresh)) {
                        sp = fresh;
                        peakHz = fresh.peakHz;
                        smoothedBpm_ = peakHz * 60.0;
                    }
                    disagreeSince_ = -1.0;
                }
            } else {
                disagreeSince_ = -1.0;
            }
            metrics_.instantBpm = peakHz * 60.0;
            metrics_.snrDb = sp.snrDb;
            metrics_.confidence = std::clamp(sp.ratio * cfg_.confNorm, 0.0, 1.0);
            metrics_.sqi = sqiOf(sp);
            const double a = std::clamp(cfg_.bpmSmoothing * (0.3 + 0.7 * metrics_.confidence), 0.02, 1.0);
            smoothedBpm_ = smoothedBpm_ <= 0.0 ? metrics_.instantBpm
                                                : smoothedBpm_ + a * (metrics_.instantBpm - smoothedBpm_);
            metrics_.bpm = smoothedBpm_;
            metrics_.valid = true;
            // Beat timing on a band centred on the fundamental only (f0 ± 0.5 Hz): the 0.75–2.5 Hz
            // signal also holds the 2nd harmonic, i.e. a second (diastolic) peak per cycle that
            // noise can make the detector pick — a half-cycle timing error. ±0.5 Hz keeps the
            // HRV side-bands (≤ 0.4 Hz) and always excludes 2·f0 for HR ≥ 45 BPM.
            std::vector<double> xb = morph;
            bandpass(xb, fs, std::max(0.5, peakHz - 0.5), peakHz + 0.5, 2);
            computeHrv(xb, fs, peakHz, peaks);
            if (metrics_.confidence >= cfg_.minBeatConfidence && metrics_.sqi >= cfg_.minBeatSqi)
                emitBeats(peaks, fs);
        } else {
            metrics_.valid = false;
            metrics_.confidence = 0.0;
            metrics_.sqi = 0.0;
        }
    } else {
        metrics_.valid = false;
        metrics_.sqi = 0.0;
    }

    // 5. Side products.
    metrics_.morphWave = morph;
    if (metrics_.valid) {
        processRatio(w, peakHz);
        std::vector<double> faceG;
        double f = 0.0;
        const int k = bestUsable;
        resample(buffer_, [k](const Sample& s) { return s.c[k][1]; }, faceG, f);
        const double mg = std::accumulate(faceG.begin(), faceG.end(), 0.0) / static_cast<double>(n);
        if (mg > 1e-6)
            for (double& v : faceG) v = v / mg - 1.0;
        detrend(faceG, dw);
        bandpass(faceG, fs, cfg_.lowHz, cfg_.highHz, cfg_.filterOrderSections);
        processBackground(peakHz, faceG, fs);
    } else {
        metrics_.ratioValid = false;
        metrics_.backgroundChecked = false;
        metrics_.ambientSuspect = false;
    }
    processRespiration();

    // Export the most recent part of the fused narrow-band waveform for display.
    const std::size_t wn = std::min(n, static_cast<std::size_t>(cfg_.waveformSeconds * fs));
    const std::size_t start = n - wn;
    double amp = 1e-12;
    for (std::size_t i = start; i < n; ++i) amp = std::max(amp, std::abs(x[i]));
    metrics_.waveform.resize(wn);
    for (std::size_t i = 0; i < wn; ++i) metrics_.waveform[i] = static_cast<float>(x[start + i] / amp);
    metrics_.beatMarkers.clear();
    for (double p : peaks)
        if (p >= static_cast<double>(start))
            metrics_.beatMarkers.push_back(static_cast<float>((p - start) / std::max<std::size_t>(1, wn - 1)));
    return metrics_;
}

}  // namespace vitals

#include "SignalProcessor.hpp"

#include <opencv2/core.hpp>

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
// SignalProcessor
// ---------------------------------------------------------------------------
SignalProcessor::SignalProcessor(const SignalConfig& cfg)
    : cfg_(cfg),
      buffer_(std::max<std::size_t>(cfg.bufferSize, 64)),
      respBuffer_(std::max<std::size_t>(cfg.respBufferSize, cfg.bufferSize)) {
    cfg_.minSamples = std::min(std::max<std::size_t>(cfg_.minSamples, 64), buffer_.capacity());
}

void SignalProcessor::reset() {
    buffer_.clear();
    respBuffer_.clear();
    smoothedResp_ = 0.0;
    respHistory_.clear();
    metrics_ = SignalMetrics{};
    paused_ = false;
    timeOffset_ = levelOffset_ = 0.0;
    lastRaw_ = 0.0;
    lastT_ = -1.0;
    smoothedBpm_ = 0.0;
}

void SignalProcessor::pause(double t) {
    if (!paused_) {
        paused_ = true;
        pauseStart_ = t;
    }
    metrics_.paused = true;
}

void SignalProcessor::push(double value, double t) {
    if (!std::isfinite(value) || !std::isfinite(t)) return;

    if (paused_) {
        paused_ = false;
        const double gap = t - pauseStart_;
        if (buffer_.empty() || gap > cfg_.maxGapSec) {
            // Long dropout: the old history is no longer phase-coherent.
            reset();
        } else {
            // Short dropout: splice the timeline (remove the gap) and re-align
            // the DC level so the detrender does not see a step.
            double dt = 1.0 / 30.0;
            if (buffer_.size() > 1) {
                dt = (buffer_.back().t - buffer_[0].t) / static_cast<double>(buffer_.size() - 1);
            }
            timeOffset_ = t - (buffer_.back().t + dt);
            levelOffset_ += lastRaw_ - value;
        }
    }

    const double te = t - timeOffset_;
    if (!buffer_.empty() && te <= buffer_.back().t) return;  // non-monotonic clock
    buffer_.push({value + levelOffset_, te});
    respBuffer_.push({value + levelOffset_, te});
    lastRaw_ = value;
    lastT_ = t;
    metrics_.paused = false;
}

bool SignalProcessor::resampleUniform(const CircularBuffer<Sample>& buffer_, std::vector<double>& out,
                                      double& fs) {
    const std::size_t n = buffer_.size();
    if (n < 2) return false;
    const double t0 = buffer_[0].t, t1 = buffer_.back().t;
    const double span = t1 - t0;
    if (span <= 0.0) return false;
    fs = static_cast<double>(n - 1) / span;
    out.resize(n);
    const double step = span / static_cast<double>(n - 1);
    std::size_t j = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double t = t0 + step * static_cast<double>(i);
        while (j + 2 < n && buffer_[j + 1].t < t) ++j;
        const Sample& a = buffer_[j];
        const Sample& b = buffer_[std::min(j + 1, n - 1)];
        const double dt = b.t - a.t;
        const double u = dt > 0.0 ? std::clamp((t - a.t) / dt, 0.0, 1.0) : 0.0;
        out[i] = a.v + u * (b.v - a.v);
    }
    return true;
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
                               double lobeHz, int nfftMin, Spectrum& out, double refLo, double refHi) {
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

    int kMax = kLo;
    for (int k = kLo; k <= kHi; ++k)
        if (p[k] > p[kMax]) kMax = k;

    // Parabolic interpolation on the magnitude for sub-bin accuracy.
    double delta = 0.0;
    if (kMax > kLo && kMax < kHi) {
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
    if (!resampleUniform(respBuffer_, x, fs) || fs < 2.0 * cfg_.respHighHz + 1.0) {
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
    respHistory_.push_back(fftBrpm);
    while (respHistory_.size() > cfg_.respStabilityFrames) respHistory_.pop_front();
    bool stable = respHistory_.size() >= cfg_.respStabilityFrames;
    if (stable) {
        const double med = median(std::vector<double>(respHistory_.begin(), respHistory_.end()));
        for (double v : respHistory_)
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

const SignalMetrics& SignalProcessor::process() {
    metrics_.paused = paused_;
    if (paused_) return metrics_;

    const std::size_t n = buffer_.size();
    metrics_.bufferFill = std::min(1.0, static_cast<double>(n) / cfg_.minSamples);
    if (n < 32) {
        metrics_.valid = false;
        metrics_.waveform.clear();
        metrics_.beatMarkers.clear();
        return metrics_;
    }

    // 1. Uniform resampling at the measured rate.
    std::vector<double> x;
    double fs = 0.0;
    if (!resampleUniform(buffer_, x, fs) || fs < 2.0 * cfg_.highHz + 0.5) {
        metrics_.valid = false;
        return metrics_;
    }
    metrics_.sampleRate = fs;

    // 2. Normalise to relative intensity change, then zero-mean detrend.
    const double mean = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(n);
    if (mean > 1e-6)
        for (double& v : x) v = v / mean - 1.0;
    detrend(x, static_cast<int>(std::lround(cfg_.detrendWindowSec * fs)));

    // 3. Zero-phase Butterworth band-pass.
    bandpass(x, fs, cfg_.lowHz, cfg_.highHz, cfg_.filterOrderSections);

    // 4–5. Spectrum → HR, peaks → IBI / RMSSD.
    std::vector<double> peaks;
    if (n >= cfg_.minSamples) {
        Spectrum sp;
        if (spectrum(x, fs, cfg_.lowHz, cfg_.highHz, cfg_.confLobeHz, cfg_.dftMinSize, sp)) {
            const double peakHz = sp.peakHz;
            metrics_.instantBpm = peakHz * 60.0;
            metrics_.snrDb = sp.snrDb;
            // Confidence = (peak power / total band power) · normalisation, hard-clamped to [0, 1].
            metrics_.confidence = std::clamp(sp.ratio * cfg_.confNorm, 0.0, 1.0);

            const double a = std::clamp(cfg_.bpmSmoothing * (0.3 + 0.7 * metrics_.confidence), 0.02, 1.0);
            smoothedBpm_ = smoothedBpm_ <= 0.0 ? metrics_.instantBpm
                                                : smoothedBpm_ + a * (metrics_.instantBpm - smoothedBpm_);
            metrics_.bpm = smoothedBpm_;
            metrics_.valid = true;
            computeHrv(x, fs, peakHz, peaks);
        }
    } else {
        metrics_.valid = false;
    }

    processRespiration();

    // Export the most recent part of the filtered waveform for display.
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

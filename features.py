import cv2
import numpy as np
from scipy import stats, fft


def fft_peak_energy(gray):
    f = np.abs(fft.fft2(gray.astype(np.float32)))
    f_shift = fft.fftshift(f)

    h, w = f_shift.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8

    mask = np.zeros_like(f_shift, dtype=bool)
    mask[cy - r:cy + r, cx - r:cx + r] = True

    background = f_shift[~mask].mean()
    peak = np.percentile(f_shift[~mask], 99)

    return float(peak / (background + 1e-6))


def laplacian_variance(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def saturation_stats(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0

    skew = stats.skew(s.ravel())
    p95 = np.percentile(s, 95)

    return float(skew), float(p95)


def lbp_entropy(gray):
    from skimage.feature import local_binary_pattern

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")

    hist, _ = np.histogram(lbp.ravel(), bins=10, density=True)
    hist = hist[hist > 0]

    return float(-np.sum(hist * np.log2(hist)))


def dct_block_artifacts(gray):
    row_var = np.var(gray.astype(np.float32), axis=1)
    col_var = np.var(gray.astype(np.float32), axis=0)

    def periodicity(v, p=8):
        # Bounds-safety fix: guard against very small images where the
        # autocorrelation array is shorter than the lag p, and against a
        # flat (zero-variance) signal which would otherwise divide by ~0.
        if len(v) <= p or np.allclose(v, v[0]):
            return 0.0
        v = v - v.mean()
        corr = np.correlate(v, v, mode="full")
        corr = corr[len(corr) // 2:]
        denom = corr[0] if abs(corr[0]) > 1e-6 else 1e-6
        return float(corr[p] / denom)

    return (periodicity(row_var) + periodicity(col_var)) / 2


def highlight_ratio(gray):
    return float(np.mean(gray > 245))


def extract_features(img_path):
    # ---------- Load image ----------
    bgr = cv2.imread(img_path)

    if bgr is None:
        raise ValueError(f"Could not read image: {img_path}")

    # ---------- Full-resolution grayscale (FFT only) ----------
    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ---------- Resize for remaining features ----------
    resized_bgr = cv2.resize(
        bgr,
        (512, 512),
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

    # ---------- Extract features ----------
    fft_s = fft_peak_energy(gray_full)
    lap_v = laplacian_variance(gray)
    sat_sk, sat_p95 = saturation_stats(resized_bgr)
    lbp_e = lbp_entropy(gray)
    dct_s = dct_block_artifacts(gray)
    hi_r = highlight_ratio(gray)

    return [
        fft_s,
        lap_v,
        sat_sk,
        sat_p95,
        lbp_e,
        dct_s,
        hi_r,
    ]
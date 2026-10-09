"""
Exact thin-film optical forward model and a measurement model.

A single transparent film of refractive index n_f and physical thickness d sits on a substrate
of index n_s in air (n_0 = 1). At normal incidence the amplitude reflection coefficient is the
Airy formula from the two interface Fresnel coefficients and the propagation phase, and the
measured quantity is the reflectance R = |r|^2. This is exact (non-absorbing, normal incidence),
so the "data" we invert are physics, not invention; the only randomness is measurement noise we
add explicitly. The instrument may also apply an unknown multiplicative intensity scale (a gain
miscalibration), which is the nuisance the self-calibration experiment marginalizes over.
"""
import numpy as np


def reflectance(lam, n_f, d, n0=1.0, n_s=1.5):
    """Reflectance R(lambda) of a single film (thickness d, index n_f) on a substrate.
    lam and d share units (nanometres here). Vectorised over lam."""
    lam = np.asarray(lam, dtype=float)
    r01 = (n0 - n_f) / (n0 + n_f)
    r12 = (n_f - n_s) / (n_f + n_s)
    beta = 2.0 * np.pi * n_f * d / lam
    e = np.exp(-2j * beta)
    r = (r01 + r12 * e) / (1.0 + r01 * r12 * e)
    return np.abs(r) ** 2


def measure(lam, n_f, d, gain=1.0, sigma=0.01, rng=None, n0=1.0, n_s=1.5):
    """A noisy spectral measurement: true reflectance times an unknown gain, plus Gaussian noise."""
    R = gain * reflectance(lam, n_f, d, n0=n0, n_s=n_s)
    if rng is not None:
        R = R + rng.normal(0.0, sigma, size=R.shape)
    return R


# A standard visible-light spectral grid (nanometres).
def default_grid(n=120, lam_min=420.0, lam_max=750.0):
    return np.linspace(lam_min, lam_max, n)


if __name__ == "__main__":
    lam = default_grid()
    # Validation 1: quarter-wave anti-reflection coating. n_f = sqrt(n0*n_s), R(lam0) ~ 0.
    n0, n_s, lam0 = 1.0, 2.25, 550.0
    n_f = np.sqrt(n0 * n_s)               # 1.5
    d_qw = lam0 / (4.0 * n_f)             # quarter-wave optical thickness
    R0 = reflectance(lam0, n_f, d_qw, n0=n0, n_s=n_s)
    print(f"AR quarter-wave: n_f={n_f:.3f} d={d_qw:.2f}nm  R(550)={float(R0):.5f}  (expect ~0)")
    # Validation 2: half-wave film is 'absent' -> bare-substrate reflectance.
    d_hw = lam0 / (2.0 * n_f)
    R_hw = reflectance(lam0, n_f, d_hw, n0=n0, n_s=n_s)
    R_bare = ((n0 - n_s) / (n0 + n_s)) ** 2
    print(f"half-wave: R(550)={float(R_hw):.5f}  bare-substrate={R_bare:.5f}  (expect equal)")
    # Validation 3: reflectance stays in [0,1].
    Rs = reflectance(lam, 1.8, 300.0, n_s=2.25)
    print(f"range over spectrum: min={Rs.min():.4f} max={Rs.max():.4f}  (expect in [0,1])")

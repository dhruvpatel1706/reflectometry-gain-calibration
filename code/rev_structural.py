"""
Structural analysis (29 Sep 2026): what the single-film Airy spectrum determines exactly, before noise.

1. Exact symmetry. At normal incidence with real, wavelength-independent indices, R depends on the Fresnel
   coefficients only through p = r01 r12 and s = r01^2 + r12^2:
       R = (s + 2 p cos 2beta) / (1 + p^2 + 2 p cos 2beta),   beta = 2 pi n d / lambda.
   Swapping r01 and r12 leaves (p, s) unchanged; the swap is n -> n' = n0 ns / n, and d' = n^2 d / (n0 ns) keeps
   n' d' = n d, so the two films have identical spectra at every gain. Checked numerically here.
2. Identifiability under an unknown gain. g R is a Moebius function of x = cos 2beta; its shape fixes 2p/(1+p^2) and
   2p/s, both functions of n (with ns known), so n stays identified up to the swap. Checked numerically by the
   noiseless profile: for each candidate index, the smallest residual sum of squares over thickness (and over gain in
   the unknown-gain case), expressed in units of sigma^2 at sigma = 0.02. A zero away from the truth would be an exact
   ambiguity; a small positive value is weak sensitivity.
3. Calibration tolerance. Noiseless fits with the gain fixed at one to spectra whose true gain is 1 + delta give the
   index, thickness and optical-thickness bias an uncorrected gain error causes.
4. Departures a scalar gain cannot absorb: an additive offset and a wavelength-dependent gain.
Writes data/results/rev_structural.json. CPU-only, deterministic.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
import optics as O

RESULTS = os.path.join(os.path.dirname(__file__), "..", "data", "results")
LAM = O.default_grid()
SIGMA = 0.02
N0 = 1.0


def mirror(n, d, ns, n0=N0):
    return n0 * ns / n, n * n * d / (n0 * ns)


def symmetry_check():
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(2000):
        ns = rng.uniform(1.3, 2.5); n = rng.uniform(1.2, 2.4); d = rng.uniform(100, 1500); g = rng.uniform(0.5, 1.5)
        n2, d2 = mirror(n, d, ns)
        a = g * O.reflectance(LAM, n, d, n_s=ns); b = g * O.reflectance(LAM, n2, d2, n_s=ns)
        worst = max(worst, float(np.max(np.abs(a - b))))
    n, d = 1.40, 700.0
    n2, d2 = mirror(n, d, 2.30)
    return {"n_random_cases": 2000, "max_abs_reflectance_difference": worst,
            "example_ns_2p30": {"n": n, "d_nm": d, "n_mirror": n2, "d_mirror_nm": d2,
                                "max_abs_difference": float(np.max(np.abs(O.reflectance(LAM, n, d, n_s=2.30) - O.reflectance(LAM, n2, d2, n_s=2.30))))}}


def _rss_known(y, n, d, ns, g=1.0):
    r = y - g * O.reflectance(LAM, n, d, n_s=ns)
    return float(r @ r)


def _rss_profile(y, n, d, ns):
    M = O.reflectance(LAM, n, d, n_s=ns)
    SyM, SMM = float(y @ M), float(M @ M)
    return float(y @ y - SyM * SyM / SMM), SyM / SMM


def index_profile(n_true, d_true, ns, gain=1.0, n_grid=None, d_lo=250.0, d_hi=1300.0):
    """Noiseless minimum RSS over d (known gain = truth) and over (d, g) (unknown gain), per candidate index."""
    y = gain * O.reflectance(LAM, n_true, d_true, n_s=ns)
    n_grid = np.linspace(1.20, 2.30, 441) if n_grid is None else n_grid
    d_fine = np.linspace(d_lo, d_hi, 4201)
    out_k, out_p, g_at = [], [], []
    for n in n_grid:
        Mrow = O.reflectance(LAM[None, :], n, d_fine[:, None], n_s=ns)          # (nd, L)
        rk = np.sum((y[None, :] - gain * Mrow) ** 2, axis=1)
        SyM = Mrow @ y; SMM = np.sum(Mrow ** 2, axis=1)
        rp = y @ y - SyM ** 2 / SMM
        res = []
        for arr, f in ((rk, lambda dd: _rss_known(y, n, dd, ns, gain)), (rp, lambda dd: _rss_profile(y, n, dd, ns)[0])):
            i = int(np.argmin(arr)); a, b = d_fine[max(i - 2, 0)], d_fine[min(i + 2, d_fine.size - 1)]
            m = minimize_scalar(f, bounds=(a, b), method="bounded", options={"xatol": 1e-7})
            res.append((min(float(m.fun), float(arr[i])), float(m.x)))
        out_k.append(res[0][0]); out_p.append(res[1][0])
        g_at.append(_rss_profile(y, n, res[1][1], ns)[1])
    return n_grid, np.array(out_k), np.array(out_p), np.array(g_at)


def profile_summary(n_true, d_true, ns, gain=1.0):
    ng, rk, rp, gat = index_profile(n_true, d_true, ns, gain)
    chi = 3.84 * SIGMA ** 2            # profile-likelihood 95% threshold for one parameter, at sigma = 0.02
    nm, dm = mirror(n_true, d_true, ns)
    far = np.abs(ng - n_true) > 0.05
    if 1.20 <= nm <= 2.30:
        far &= np.abs(ng - nm) > 0.05
    return {"n_true": n_true, "d_true_nm": d_true, "ns": ns, "gain": gain, "n_mirror": nm, "d_mirror_nm": dm,
            "mirror_in_grid": bool(1.20 <= nm <= 2.30),
            "known_gain_set_width": float(np.ptp(ng[rk <= chi])) if np.any(rk <= chi) else 0.0,
            "unknown_gain_set_width": float(np.ptp(ng[rp <= chi])) if np.any(rp <= chi) else 0.0,
            "known_gain_set": [float(ng[rk <= chi].min()), float(ng[rk <= chi].max())],
            "unknown_gain_set": [float(ng[rp <= chi].min()), float(ng[rp <= chi].max())],
            "min_far_rss_over_sigma2_known": float(rk[far].min() / SIGMA ** 2),
            "min_far_rss_over_sigma2_unknown": float(rp[far].min() / SIGMA ** 2),
            "curve": {"n": ng.tolist(), "known_over_sigma2": (rk / SIGMA ** 2).tolist(),
                      "unknown_over_sigma2": (rp / SIGMA ** 2).tolist(), "profiled_gain": gat.tolist()}}


def fit(y, ns, start, gain_mode, g_fixed=1.0, lo=(1.20, 150.0), hi=(2.6, 1500.0)):
    def res(p):
        M = O.reflectance(LAM, p[0], p[1], n_s=ns)
        if gain_mode == "fixed":
            return y - g_fixed * M
        g = float(y @ M) / float(M @ M)
        return y - g * M
    sol = least_squares(res, start, bounds=(lo, hi), x_scale=(0.01, 5.0), xtol=1e-13, ftol=1e-15, gtol=1e-15, max_nfev=4000)
    return sol.x, float(sol.fun @ sol.fun)


def _grid_start(y, ns, gain_mode, g_fixed=1.0):
    ng = np.linspace(1.20, 2.50, 66); dg = np.linspace(200.0, 1400.0, 601)  # 0.02 in n, 2 nm in d; least squares refines
    best = (np.inf, None)
    for n in ng:
        Mrow = O.reflectance(LAM[None, :], n, dg[:, None], n_s=ns)
        if gain_mode == "fixed":
            r = np.sum((y[None, :] - g_fixed * Mrow) ** 2, axis=1)
        else:
            SyM = Mrow @ y; SMM = np.sum(Mrow ** 2, axis=1); r = y @ y - SyM ** 2 / SMM
        i = int(np.argmin(r))
        if r[i] < best[0]:
            best = (float(r[i]), (n, dg[i]))
    return np.array(best[1])


def best_fit(y, ns, truth, gain_mode, g_fixed=1.0):
    """Global noiseless fit: least squares from the best coarse-grid cell and from the truth; the lower RSS wins.
    Near index matching (n close to ns) the fringes vanish and a single start can stop in the wrong thickness basin."""
    cands = [fit(y, ns, _grid_start(y, ns, gain_mode, g_fixed), gain_mode, g_fixed), fit(y, ns, np.array(truth, float), gain_mode, g_fixed)]
    return min(cands, key=lambda c: c[1])[0]


def _bias_stats(films, bn, bd, bt, ns):
    far = np.array([abs(n - ns) >= 0.03 for n, _ in films])
    return {"median_abs_bias_n": float(np.median(np.abs(bn))), "mean_bias_n": float(np.mean(bn)),
            "median_abs_bias_d_nm": float(np.median(np.abs(bd))), "mean_bias_d_nm": float(np.mean(bd)),
            "median_abs_bias_tau_nm": float(np.median(np.abs(bt))), "max_abs_bias_tau_nm": float(np.max(np.abs(bt))),
            "n_index_matched_films": int((~far).sum()),
            "max_abs_bias_tau_nm_excluding_index_matched": float(np.max(np.abs(bt[far]))),
            "max_abs_bias_n_excluding_index_matched": float(np.max(np.abs(bn[far]))),
            "frac_abs_bias_tau_gt_10nm": float(np.mean(np.abs(bt) > 10.0)),
            "frac_abs_bias_tau_gt_10nm_excluding_index_matched": float(np.mean(np.abs(bt[far]) > 10.0))}


def gain_error_bias(n_films=150, seed=21):
    """Noiseless: true gain 1 + delta, fitted with the gain fixed at one. Films as in the main audit (ns = 1.5)."""
    rng = np.random.default_rng(seed)
    films = [(rng.uniform(1.35, 2.15), rng.uniform(400, 900)) for _ in range(n_films)]
    out = {}
    for delta in (-0.10, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.10, 0.20):
        bn, bd, bt = [], [], []
        for n, d in films:
            y = (1 + delta) * O.reflectance(LAM, n, d, n_s=1.5)
            x = best_fit(y, 1.5, (n, d), "fixed")
            bn.append(x[0] - n); bd.append(x[1] - d); bt.append(x[0] * x[1] - n * d)
        bn, bd, bt = map(np.array, (bn, bd, bt))
        out[f"{delta:+.2f}"] = _bias_stats(films, bn, bd, bt, 1.5)
    return {"n_films": n_films, "ns": 1.5, "by_gain_error": out}


def departures(n_films=150, seed=22):
    """Noiseless departures from a scalar gain, fitted with the gain fixed at one and with a profiled scalar gain."""
    rng = np.random.default_rng(seed)
    films = [(rng.uniform(1.35, 2.15), rng.uniform(400, 900)) for _ in range(n_films)]
    tilt = np.interp(LAM, [450.0, 700.0], [1.10, 0.95])           # +10% at 450 nm, -5% at 700 nm, linear in lambda
    cases = {"scalar_gain_1p10": lambda R: 1.10 * R, "additive_offset_0p02": lambda R: R + 0.02,
             "linear_spectral_gain_+10pct_450nm_-5pct_700nm": lambda R: tilt * R}
    out = {}
    for name, f in cases.items():
        rec = {}
        for mode in ("fixed", "profile"):
            bn, bd, bt = [], [], []
            for n, d in films:
                y = f(O.reflectance(LAM, n, d, n_s=1.5))
                x = best_fit(y, 1.5, (n, d), mode)
                bn.append(x[0] - n); bd.append(x[1] - d); bt.append(x[0] * x[1] - n * d)
            bn, bd, bt = map(np.array, (bn, bd, bt))
            rec["gain_fixed_at_1" if mode == "fixed" else "scalar_gain_profiled"] = _bias_stats(films, bn, bd, bt, 1.5)
        out[name] = rec
    return {"n_films": n_films, "ns": 1.5, "cases": out}


def main():
    res = {"symmetry": symmetry_check(),
           "index_profiles": {"main_regime_ns1p5": profile_summary(1.80, 650.0, 1.5),
                              "low_index_ns1p5": profile_summary(1.45, 650.0, 1.5),
                              "reviewer_example_ns2p30": profile_summary(1.40, 700.0, 2.30),
                              "high_index_ns2p30": profile_summary(2.00, 650.0, 2.30)},
           "gain_error_bias": gain_error_bias(), "departures": departures(),
           "config": {"sigma_for_units": SIGMA, "n_wavelengths": len(LAM), "lam_nm": [float(LAM[0]), float(LAM[-1])],
                      "profile_threshold": "RSS <= 3.84 sigma^2 (one-parameter 95% profile-likelihood set)"}}
    with open(os.path.join(RESULTS, "rev_structural.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    s = res["symmetry"]; print("symmetry max |dR|", s["max_abs_reflectance_difference"], s["example_ns_2p30"])
    for k, v in res["index_profiles"].items():
        print(k, "mirror", round(v["n_mirror"], 4), v["mirror_in_grid"], "known set", v["known_gain_set"], "unknown set", v["unknown_gain_set"],
              "min far rss/s2 known %.3g unknown %.3g" % (v["min_far_rss_over_sigma2_known"], v["min_far_rss_over_sigma2_unknown"]))
    for k, v in res["gain_error_bias"]["by_gain_error"].items():
        print("gain error", k, {a: round(b, 4) for a, b in v.items()})
    for k, v in res["departures"]["cases"].items():
        print(k, v)


if __name__ == "__main__":
    main()

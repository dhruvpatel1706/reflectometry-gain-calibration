"""
Dispersion experiments (29 Sep 2026).

Films are non-absorbing with Cauchy dispersion, parameterized at the band centre lam_c = 585 nm:
    n(lam) = n_ref + B (1/lam^2 - 1/lam_c^2),    n_ref = n(lam_c),
so the group index at lam_c is n_g = n_ref + 2 B / lam_c^2 (= A + 3B/lam_c^2 in the A, B form). Three optical
thicknesses are distinguished: the phase optical thickness n_ref d, the group optical thickness n_g d, and the
product n_hat d_hat returned by a constant-index fit.

D1  Noiseless misspecified fits. Dispersive spectra (gain one, no noise) fitted by least squares with the
    constant-index Airy model, gain known and gain profiled. For each film: the fitted n, d and n d; the wavelength
    lam* at which the phase thickness n(lam*) d equals the fitted product; and the fit residual in units of the noise
    used elsewhere (sigma = 0.02), to show whether the misfit would be visible in data. Two thickness bands.
D2  A correctly specified dispersive inversion: a grid over (n_ref, B, d) with the gain known, profiled, or
    marginalized under U[0.5, 1.5], for the same questions as E1-E3 (RMSE ratios with paired bootstrap intervals,
    coverage and widths, a true gain from U[0.85, 1.20] analysed at the true gain, at one, and marginalized).
    The model matrix is stored in float32; a float64 check on a coarse grid bounds the effect. A second, coarser
    grid (spacing doubled in all three directions) is run on the same spectra as a convergence check.
The reflectance with dispersion is the exact single-layer Airy formula evaluated with n(lam); it is checked against
an independent characteristic-matrix computation.
Writes data/results/rev_dispersion.json and data/results/rev_dispersion_trials.npz. CPU-only, deterministic.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.optimize import least_squares
import optics as O
import fastgrid as F
from rev_matched import boot_ratio, rmse

RESULTS = os.path.join(os.path.dirname(__file__), "..", "data", "results")
LAM = O.default_grid()
LAM_C = 0.5 * (LAM[0] + LAM[-1])
W = 1.0 / LAM ** 2 - 1.0 / LAM_C ** 2
N_S, SIGMA = 1.5, 0.02
N_LO, N_HI, D_LO, D_HI, B_LO, B_HI = 1.35, 2.15, 400.0, 900.0, 1.2e4, 2.4e4
GAIN_LO, GAIN_HI, PRIOR = 0.85, 1.20, (0.5, 1.5)
MEAN_N, MEAN_D = 0.5 * (N_LO + N_HI), 0.5 * (D_LO + D_HI)


def n_of(n_ref, B, lam=LAM):
    return n_ref + B * (1.0 / lam ** 2 - 1.0 / LAM_C ** 2)


def refl(n_ref, B, d, lam=LAM, ns=N_S):
    return O.reflectance(lam, n_of(n_ref, B, lam), d, n_s=ns)


def refl_charmatrix(n_ref, B, d, lam=LAM, ns=N_S, n0=1.0):
    """Independent check: characteristic (Abeles) matrix of one layer, normal incidence."""
    n = n_of(n_ref, B, lam)
    delta = 2 * np.pi * n * d / lam
    m11, m12, m21, m22 = np.cos(delta), 1j * np.sin(delta) / n, 1j * n * np.sin(delta), np.cos(delta)
    Bv, Cv = m11 + m12 * ns, m21 + m22 * ns
    r = (n0 * Bv - Cv) / (n0 * Bv + Cv)
    return np.abs(r) ** 2


# ---------------------------------------------------------------- D1: noiseless constant-index fits
def _const_fit(y, gain_mode, G, P):
    ll, SyM = G.loglik(y, SIGMA, ("fixed", 1.0) if gain_mode == "known" else ("profile",))
    order = np.argsort(-ll)
    starts = [order[0]]
    for i in order[1:2000]:
        if abs(P["n"][i] - P["n"][starts[0]]) > 0.05:
            starts.append(i); break

    def res(p):
        R = O.reflectance(LAM, p[0], p[1], n_s=N_S)
        if gain_mode == "known":
            return R - y
        g = (R @ y) / (R @ R)
        return g * R - y
    best = None
    for i in starts:
        f = least_squares(res, [P["n"][i], P["d"][i]], bounds=([1.10, 100.0], [2.80, 3500.0]), x_scale=[0.01, 5.0], xtol=1e-12, ftol=1e-14, gtol=1e-14)
        if best is None or f.cost < best.cost:
            best = f
    return best.x[0], best.x[1], 2.0 * best.cost


def d1(n_films=200, seed=41):
    out = {}
    for tag, (dlo, dhi), dgrid in (("band_400_900", (400.0, 900.0), np.linspace(250.0, 1300.0, 2100)),
                                   ("band_1600_2400", (1600.0, 2400.0), np.linspace(1300.0, 3000.0, 3400))):
        M, P = F.airy_matrix(LAM, np.linspace(1.10, 2.80, 341), dgrid, n_s=N_S)
        G = F.Grid(M, P); del M
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n_films):
            nr, B, d = rng.uniform(N_LO, N_HI), rng.uniform(B_LO, B_HI), rng.uniform(dlo, dhi)
            y = refl(nr, B, d)
            ng = nr + 2.0 * B / LAM_C ** 2
            row = {"n_ref": nr, "B": B, "d": d, "n_g": ng}
            for gm in ("known", "profile"):
                nh, dh, rss = _const_fit(y, gm, G, P)
                t = nh * dh
                inv = (t / d - nr) / B + 1.0 / LAM_C ** 2
                row[gm] = {"n": nh, "d": dh, "tau": t, "rss_over_L_sigma2": rss / (LAM.size * SIGMA ** 2),
                           "lam_star": float(1.0 / np.sqrt(inv)) if inv > 0 else float("nan")}
            rows.append(row)
        del G
        s = {"n_films": n_films, "d_band": [dlo, dhi]}
        for gm in ("known", "profile"):
            tau = np.array([r[gm]["tau"] for r in rows]); ph = np.array([r["n_ref"] * r["d"] for r in rows])
            gr = np.array([r["n_g"] * r["d"] for r in rows]); nh = np.array([r[gm]["n"] for r in rows])
            nr = np.array([r["n_ref"] for r in rows]); ngv = np.array([r["n_g"] for r in rows])
            dh = np.array([r[gm]["d"] for r in rows]); dd = np.array([r["d"] for r in rows])
            ls = np.array([r[gm]["lam_star"] for r in rows]); chi = np.array([r[gm]["rss_over_L_sigma2"] for r in rows])
            q = lambda v: [float(np.percentile(v, 25)), float(np.median(v)), float(np.percentile(v, 75))]
            s[gm] = {"rel_diff_tau_vs_phase_OT_q25_med_q75": q((tau - ph) / ph), "rel_diff_tau_vs_group_OT_q25_med_q75": q((tau - gr) / gr),
                     "frac_tau_closer_to_phase_OT": float(np.mean(np.abs(tau - ph) < np.abs(tau - gr))),
                     "n_hat_minus_n_ref_q25_med_q75": q(nh - nr), "n_hat_minus_n_g_q25_med_q75": q(nh - ngv),
                     "rel_diff_d_hat_q25_med_q75": q((dh - dd) / dd),
                     "lam_star_nm_q25_med_q75": q(ls[np.isfinite(ls)]), "n_lam_star_undefined": int(np.sum(~np.isfinite(ls))),
                     "rss_over_L_sigma2_q25_med_q75": q(chi), "frac_rss_over_L_sigma2_gt_0p1": float(np.mean(chi > 0.1))}
        s["dispersion_span_over_band_pct_median"] = float(np.median([100 * np.ptp(n_of(r["n_ref"], r["B"])) / np.mean(n_of(r["n_ref"], r["B"])) for r in rows]))
        s["group_minus_phase_index_median"] = float(np.median([r["n_g"] - r["n_ref"] for r in rows]))
        out[tag] = s
        out[tag + "_films"] = rows
    return out


# ---------------------------------------------------------------- D2: correctly specified dispersive inversion
class Grid32(F.Grid):
    """fastgrid.Grid with the model matrix held in float32 (SyM accumulated by float32 BLAS, returned as float64)."""

    def __init__(self, M32, params):
        self.M = M32
        self.L = M32.shape[1]
        self.SMM = np.zeros(M32.shape[0])
        for i in range(0, M32.shape[0], 200000):
            blk = M32[i:i + 200000].astype(np.float64)
            self.SMM[i:i + 200000] = np.einsum("ij,ij->i", blk, blk)
        self.params = {k: np.asarray(v, float) for k, v in params.items()}
        self.order = {k: np.argsort(v, kind="stable") for k, v in self.params.items()}
        self.sorted_vals = {k: v[self.order[k]] for k, v in self.params.items()}

    def project(self, y):
        return (self.M @ np.asarray(y, np.float32)).astype(np.float64)


def disp_matrix(n_grid, B_grid, d_grid, dtype=np.float32):
    nn, nb, nd = n_grid.size, B_grid.size, d_grid.size
    M = np.empty((nn, nb, nd, LAM.size), dtype=dtype)
    d = d_grid[None, None, :, None]; lam = LAM[None, None, None, :]
    for i in range(0, nn, 4):
        n = n_grid[i:i + 4, None, None, None] + B_grid[None, :, None, None] * W[None, None, None, :]
        r01, r12 = (1.0 - n) / (1.0 + n), (n - N_S) / (n + N_S)
        e = np.exp(-4j * np.pi * n * d / lam)
        r = (r01 + r12 * e) / (1.0 + r01 * r12 * e)
        M[i:i + 4] = (np.abs(r) ** 2).astype(dtype)
    NN, BB, DD = np.meshgrid(n_grid, B_grid, d_grid, indexing="ij")
    NN, BB, DD = NN.ravel(), BB.ravel(), DD.ravel()
    return M.reshape(-1, LAM.size), {"n": NN, "B": BB, "d": DD, "tau": NN * DD, "tau_g": (NN + 2.0 * BB / LAM_C ** 2) * DD}


PARS = ("n", "B", "d", "tau", "tau_g")


def run_d2(G, seed_films=51, n_films=300):
    """E1-type (gain one) and E3-type (true gain in U[0.85, 1.20]) analyses on the same grid."""
    rng = np.random.default_rng(seed_films)
    films = [(rng.uniform(N_LO, N_HI), rng.uniform(B_LO, B_HI), rng.uniform(D_LO, D_HI)) for _ in range(n_films)]
    noise1 = rng.normal(0.0, SIGMA, size=(n_films, LAM.size))
    gains = rng.uniform(GAIN_LO, GAIN_HI, size=n_films)
    noise3 = rng.normal(0.0, SIGMA, size=(n_films, LAM.size))
    rec = {}
    def add(key, s, truth):
        for p in PARS:
            rec.setdefault(f"{key}_{p}_err", []).append(s[f"{p}_mean"] - truth[p])
            lo, hi = s[f"{p}_ci90"]
            rec.setdefault(f"{key}_{p}_cov90", []).append(float(lo <= truth[p] <= hi))
            rec.setdefault(f"{key}_{p}_width90", []).append(hi - lo)
    for j, (nr, B, d) in enumerate(films):
        truth = {"n": nr, "B": B, "d": d, "tau": nr * d, "tau_g": (nr + 2 * B / LAM_C ** 2) * d}
        R = refl(nr, B, d)
        y1 = R + noise1[j]
        for key, gain in (("e1_known", ("fixed", 1.0)), ("e1_profile", ("profile",)), ("e1_marginal", ("marginal",) + PRIOR)):
            ll, _ = G.loglik(y1, SIGMA, gain)
            add(key, G.summarize(G.weights(ll), levels=(0.90,), names=PARS), truth)
        y3 = gains[j] * R + noise3[j]
        for key, gain in (("e3_true_gain", ("fixed", gains[j])), ("e3_gain_one", ("fixed", 1.0)), ("e3_marginal", ("marginal",) + PRIOR)):
            ll, _ = G.loglik(y3, SIGMA, gain)
            add(key, G.summarize(G.weights(ll), levels=(0.90,), names=PARS), truth)
    A = {k: np.asarray(v) for k, v in rec.items()}
    A["films"] = np.asarray(films); A["gains"] = gains
    return A


def summarize_d2(A):
    means = {"n": MEAN_N, "d": MEAN_D, "tau": MEAN_N * MEAN_D, "B": 0.5 * (B_LO + B_HI), "tau_g": None}
    out = {}
    for key in ("e1_known", "e1_profile", "e1_marginal", "e3_true_gain", "e3_gain_one", "e3_marginal"):
        out[key] = {}
        for p in PARS:
            e = A[f"{key}_{p}_err"]
            out[key][p] = {"rmse": rmse(e), "bias": float(np.mean(e)), "cov90": float(np.mean(A[f"{key}_{p}_cov90"])),
                           "mean_width90": float(np.mean(A[f"{key}_{p}_width90"]))}
            if means[p]:
                out[key][p]["rel_rmse"] = rmse(e) / means[p]
    out["ratios"] = {"e1_profile_over_known_n": boot_ratio(A["e1_profile_n_err"], A["e1_known_n_err"]),
                     "e1_marginal_over_known_n": boot_ratio(A["e1_marginal_n_err"], A["e1_known_n_err"]),
                     "e1_profile_over_known_d": boot_ratio(A["e1_profile_d_err"], A["e1_known_d_err"]),
                     "e1_profile_over_known_tau_g": boot_ratio(A["e1_profile_tau_g_err"], A["e1_known_tau_g_err"]),
                     "e3_marginal_over_true_gain_n": boot_ratio(A["e3_marginal_n_err"], A["e3_true_gain_n_err"]),
                     "e3_gain_one_over_true_gain_n": boot_ratio(A["e3_gain_one_n_err"], A["e3_true_gain_n_err"]),
                     "e3_marginal_over_true_gain_B": boot_ratio(A["e3_marginal_B_err"], A["e3_true_gain_B_err"])}
    return out


def float_check():
    """float32 against float64 model matrices on a small grid: largest change in any posterior mean."""
    ng, Bg, dg = np.linspace(1.30, 2.20, 46), np.linspace(0.0, 3.6e4, 7), np.linspace(380.0, 920.0, 271)
    M32, P = disp_matrix(ng, Bg, dg, np.float32)
    M64, _ = disp_matrix(ng, Bg, dg, np.float64)
    G32, G64 = Grid32(M32, P), F.Grid(M64, P)
    rng = np.random.default_rng(61)
    worst = {p: 0.0 for p in PARS}
    for _ in range(20):
        nr, B, d = rng.uniform(1.4, 2.1), rng.uniform(B_LO, B_HI), rng.uniform(420, 880)
        y = refl(nr, B, d) + rng.normal(0, SIGMA, LAM.size)
        for gain in (("fixed", 1.0), ("profile",), ("marginal",) + PRIOR):
            s32 = G32.summarize(G32.weights(G32.loglik(y, SIGMA, gain)[0]), levels=(), names=PARS)
            s64 = G64.summarize(G64.weights(G64.loglik(y, SIGMA, gain)[0]), levels=(), names=PARS)
            for p in PARS:
                worst[p] = max(worst[p], abs(s32[f"{p}_mean"] - s64[f"{p}_mean"]))
    return worst


def main():
    t0 = time.time()
    rng = np.random.default_rng(71)
    chk = max(float(np.max(np.abs(refl(a, b, c) - refl_charmatrix(a, b, c)))) for a, b, c in
              [(rng.uniform(1.3, 2.3), rng.uniform(0, 4e4), rng.uniform(100, 2500)) for _ in range(200)])
    print("Airy-with-dispersion vs characteristic matrix: max |dR|", chk, flush=True)
    res = {"model_check_max_abs_dR": chk}
    res["float32_check_max_abs_change_in_posterior_mean"] = float_check()
    print("float32 check", res["float32_check_max_abs_change_in_posterior_mean"], flush=True)

    D1 = d1()
    res["D1"] = {k: v for k, v in D1.items() if not k.endswith("_films")}
    for k, v in res["D1"].items():
        print("D1", k, json.dumps({g: v[g] for g in ("known", "profile")}), flush=True)

    grids = {"main": (np.linspace(1.20, 2.30, 221), np.linspace(0.0, 3.6e4, 25), np.linspace(350.0, 950.0, 701)),
             "coarse": (np.linspace(1.20, 2.30, 111), np.linspace(0.0, 3.6e4, 13), np.linspace(350.0, 950.0, 351))}
    trials = {}
    res["D2"] = {}
    for gname, (ng, Bg, dg) in grids.items():
        M, P = disp_matrix(ng, Bg, dg)
        G = Grid32(M, P); del M
        A = run_d2(G)
        del G
        res["D2"][gname] = summarize_d2(A)
        res["D2"][gname]["grid"] = {"n": [float(ng[0]), float(ng[-1]), int(ng.size)], "B_nm2": [float(Bg[0]), float(Bg[-1]), int(Bg.size)],
                                     "d_nm": [float(dg[0]), float(dg[-1]), int(dg.size)]}
        for k, v in A.items():
            trials[f"{gname}_{k}"] = v
        S = res["D2"][gname]
        print("D2", gname, "E1 rel n known %.4f profile %.4f marginal %.4f" % (S["e1_known"]["n"]["rel_rmse"], S["e1_profile"]["n"]["rel_rmse"], S["e1_marginal"]["n"]["rel_rmse"]), flush=True)
        print("   ratios", {a: (round(b["ratio"], 2), [round(x, 2) for x in b["ci95"]]) for a, b in S["ratios"].items()}, flush=True)
        for key in ("e1_known", "e3_true_gain", "e3_gain_one", "e3_marginal"):
            print("  ", key, {p: (round(S[key][p]["cov90"], 3), round(S[key][p]["rmse"], 4), round(S[key][p]["mean_width90"], 4)) for p in PARS}, flush=True)
    res["config"] = {"lam_nm": [float(LAM[0]), float(LAM[-1]), int(LAM.size)], "lam_c_nm": float(LAM_C), "n_s": N_S, "sigma": SIGMA,
                     "n_ref_range": [N_LO, N_HI], "B_range_nm2": [B_LO, B_HI], "d_range_nm": [D_LO, D_HI],
                     "true_gain_range": [GAIN_LO, GAIN_HI], "gain_prior_uniform": list(PRIOR),
                     "seeds": {"D1": 41, "D2_films_and_noise": 51, "model_check": 71, "float_check": 61}, "D2_films": 300, "D1_films": 200}
    with open(os.path.join(RESULTS, "rev_dispersion.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    d1rows = {f"d1_{band}_{g}_{q}": np.array([r[g][q] for r in D1[band + "_films"]]) for band in ("band_400_900", "band_1600_2400")
              for g in ("known", "profile") for q in ("n", "d", "tau", "lam_star", "rss_over_L_sigma2")}
    d1rows.update({f"d1_{band}_truth": np.array([[r["n_ref"], r["B"], r["d"], r["n_g"]] for r in D1[band + "_films"]]) for band in ("band_400_900", "band_1600_2400")})
    np.savez_compressed(os.path.join(RESULTS, "rev_dispersion_trials.npz"), **trials, **d1rows)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

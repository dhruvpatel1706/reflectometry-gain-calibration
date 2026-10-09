"""
Local sensitivity analysis (29 Sep 2026): analytic derivatives, the (n, tau, g) decomposition, and a
linearized calibration tolerance.

For real indices the Airy reflectance depends on the film only through the two Fresnel coefficients and the phase,
    R = (s + 2 p c) / (1 + p^2 + 2 p c),   s = r01^2 + r12^2,  p = r01 r12,  c = cos(4 pi tau / lambda),  tau = n d,
so in the coordinates (n, tau) the index acts on the spectrum in two separate ways:
    dR/dn |_tau   the Fresnel channel: amplitudes change, fringe positions do not;
    dR/dtau |_n   the phase channel: fringes move.
The familiar derivative at fixed physical thickness is dR/dn|_d = dR/dn|_tau + d dR/dtau|_n. A gain g multiplies the
spectrum, so its sensitivity is R itself. With unit noise the information about n left after the nuisance
directions are removed is the squared length of the Fresnel vector after projection:
    known gain     I_n = |P_tau^perp  dR/dn|_tau|^2
    unknown gain   I_n = |P_(tau,g)^perp dR/dn|_tau|^2 = (1 - rho^2) |P_tau^perp dR/dn|_tau|^2,
where rho is the correlation between the Fresnel vector and R after both are projected orthogonal to dR/dtau (the
index-gain correlation "after accounting for thickness"). This is the Schur complement of the Fisher matrix written
out. The standard-deviation-bound ratio (square root of the Cramer-Rao variance-bound ratio) is 1/sqrt(1 - rho^2).
The bound for n does not depend on whether the nuisance is d or tau: both span the same direction.

The swap n -> n0 ns / n exchanges r01 and r12 and preserves tau, so R(n, tau) = R(n0 ns / n, tau) exactly and the
Fresnel vector vanishes at n* = sqrt(n0 ns): there the local information about n is zero with or without the gain.
At index matching (n = ns) r12 = 0, the phase vector vanishes and tau (hence d) carries no information; those points
are singular and are excluded from band summaries (grids are placed so that no point lies within 0.01 of n_s).

Also computed: derivative checks (analytic against central differences over a range of steps, including fixed
steps h_n = 1e-6, h_d = 1e-4 nm; the JSON key fisher_theory_steps_* names them after an earlier script), condition
numbers of the correlation-normalized Fisher matrices, the band summaries on the sweep axes, per-film bounds for the 600 E1 films of rev_matched.py compared with their Monte Carlo errors, and the
first-order bias from a gain error delta analysed with the gain fixed at one,
    b = delta (S^T S)^{-1} S^T R,   S = [dR/dn|_d, dR/dd|_n],
compared with the nonlinear fits of rev_structural.py on the same 150 films, and turned into a tolerance: the gain
error at which the index bias equals the calibrated Cramer-Rao standard deviation at sigma = 0.02.
Writes data/results/rev_fisher.json. CPU-only, deterministic, no Monte Carlo.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import optics as O

RESULTS = os.path.join(os.path.dirname(__file__), "..", "data", "results")
LAM = O.default_grid()
N0, SIGMA = 1.0, 0.02
N_LO, N_HI = 1.35, 2.15


def airy(lam, n, tau, ns, n0=N0):
    """R, the Fresnel vector dR/dn|_tau and the phase vector dR/dtau|_n (analytic)."""
    r01, r12 = (n0 - n) / (n0 + n), (n - ns) / (n + ns)
    dr01, dr12 = -2.0 * n0 / (n0 + n) ** 2, 2.0 * ns / (n + ns) ** 2
    ph = 4.0 * np.pi * tau / lam
    c, dc_dtau = np.cos(ph), -np.sin(ph) * 4.0 * np.pi / lam
    s, p = r01 ** 2 + r12 ** 2, r01 * r12
    N, D = s + 2.0 * p * c, 1.0 + p * p + 2.0 * p * c
    R = N / D
    dR_dr01 = ((2 * r01 + 2 * r12 * c) * D - N * (2 * p * r12 + 2 * r12 * c)) / D ** 2
    dR_dr12 = ((2 * r12 + 2 * r01 * c) * D - N * (2 * p * r01 + 2 * r01 * c)) / D ** 2
    dR_dc = 2.0 * p * (D - N) / D ** 2
    return R, dR_dr01 * dr01 + dR_dr12 * dr12, dR_dc * dc_dtau


def _perp(v, basis):
    """Component of v orthogonal to span(basis) (least squares)."""
    B = np.column_stack(basis)
    coef, *_ = np.linalg.lstsq(B, v, rcond=None)
    return v - B @ coef


def decompose(n, d, ns, lam=LAM):
    tau = n * d
    R, Fn, Pt = airy(lam, n, tau, ns)
    a = _perp(Fn, [Pt])                       # Fresnel vector after removing thickness
    b = _perp(R, [Pt])                        # gain vector after removing thickness
    I_known = float(a @ a)
    rho = float(a @ b / np.sqrt((a @ a) * (b @ b)))
    I_unknown = float(_perp(Fn, [Pt, R]) @ _perp(Fn, [Pt, R]))
    # the same quantities from the (n, d, g) Fisher matrix, for the check
    S = np.column_stack([Fn + d * Pt, n * Pt, R])
    J3 = S.T @ S
    ratio_nd = float(np.sqrt(np.linalg.inv(J3)[0, 0] / np.linalg.inv(J3[:2, :2])[0, 0]))
    St = np.column_stack([Fn, Pt, R])
    def cond_corr(J):
        dsq = np.sqrt(np.diag(J))
        return float(np.linalg.cond(J / np.outer(dsq, dsq)))
    Sn = Fn + d * Pt
    rho_nd = float(Sn @ (n * Pt) / np.sqrt((Sn @ Sn) * ((n * Pt) @ (n * Pt))))
    return {"I_known": I_known, "I_unknown": I_unknown, "rho_ng_given_tau": rho,
            "sd_ratio": float(np.sqrt(I_known / I_unknown)), "sd_ratio_from_rho": float(1.0 / np.sqrt(1.0 - rho ** 2)),
            "sd_ratio_from_nd_fisher": ratio_nd, "corr_n_d": rho_nd,
            "cond_corr_nd": cond_corr(J3[:2, :2]), "cond_corr_ndg": cond_corr(J3),
            "cond_corr_ntaug": cond_corr(St.T @ St),
            "crb_sd_n_known": float(SIGMA / np.sqrt(I_known)), "crb_sd_n_unknown": float(SIGMA / np.sqrt(I_unknown))}


def linear_gain_bias(n, d, ns, lam=LAM):
    """First-order (n, d) bias per unit gain error when the gain is fixed at one."""
    R, Fn, Pt = airy(lam, n, n * d, ns)
    S = np.column_stack([Fn + d * Pt, n * Pt])
    b = np.linalg.solve(S.T @ S, S.T @ R)
    return float(b[0]), float(b[1])


def derivative_checks():
    rng = np.random.default_rng(31)
    pts = [(rng.uniform(1.25, 2.25), rng.uniform(150, 2400), rng.choice([1.35, 1.5, 1.75, 2.0, 2.3])) for _ in range(200)]
    R_err = 0.0
    for n, d, ns in pts:
        R_err = max(R_err, float(np.max(np.abs(airy(LAM, n, n * d, ns)[0] - O.reflectance(LAM, n, d, n_s=ns)))))
    steps = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    table = []
    for h in steps:
        en, ed = 0.0, 0.0
        for n, d, ns in pts:
            _, Fn, Pt = airy(LAM, n, n * d, ns)
            an, ad = Fn + d * Pt, n * Pt
            fn = (O.reflectance(LAM, n + h, d, n_s=ns) - O.reflectance(LAM, n - h, d, n_s=ns)) / (2 * h)
            hd = h * 100.0                                   # the same relative step in d (d ~ 100 x n)
            fd = (O.reflectance(LAM, n, d + hd, n_s=ns) - O.reflectance(LAM, n, d - hd, n_s=ns)) / (2 * hd)
            en = max(en, float(np.max(np.abs(fn - an)) / np.max(np.abs(an))))
            ed = max(ed, float(np.max(np.abs(fd - ad)) / np.max(np.abs(ad))))
        table.append({"step_n": h, "step_d_nm": h * 100.0, "max_rel_err_dRdn": en, "max_rel_err_dRdd": ed})
    # fixed steps h_n = 1e-6, h_d = 1e-4 nm and their effect on the reported ratio
    old = []
    for n, d, ns in pts[:50]:
        R = lambda nn, dd: O.reflectance(LAM, nn, dd, n_s=ns)
        S = np.column_stack([(R(n + 1e-6, d) - R(n - 1e-6, d)) / 2e-6, (R(n, d + 1e-4) - R(n, d - 1e-4)) / 2e-4, R(n, d)])
        J3 = S.T @ S
        r_fd = np.sqrt(np.linalg.inv(J3)[0, 0] / np.linalg.inv(J3[:2, :2])[0, 0])
        old.append(abs(r_fd / decompose(n, d, ns)["sd_ratio"] - 1.0))
    return {"max_abs_R_difference_vs_optics_reflectance": R_err, "n_points": len(pts), "central_difference_table": table,
            "fisher_theory_steps_max_rel_change_in_sd_ratio": float(max(old))}


def band(d_lo, d_hi, ns, k=20):
    ngrid = N_LO + (np.arange(k) + 0.5) * (N_HI - N_LO) / k
    dgrid = d_lo + (np.arange(k) + 0.5) * (d_hi - d_lo) / k
    assert np.min(np.abs(ngrid - ns)) >= 0.0099, "band grid too close to index matching"
    rows = [decompose(n, d, ns) for n in ngrid for d in dgrid]
    r = np.array([x["sd_ratio"] for x in rows])
    nstar = np.sqrt(N0 * ns)
    return {"d_band": [d_lo, d_hi], "n_s": ns, "n_star": float(nstar), "grid": [k, k],
            "sd_ratio_median": float(np.median(r)), "sd_ratio_p25": float(np.percentile(r, 25)), "sd_ratio_p75": float(np.percentile(r, 75)),
            "sd_ratio_rms_bound_ratio": float(np.sqrt(np.mean([1.0 / x["I_unknown"] for x in rows]) /
                                                  np.mean([1.0 / x["I_known"] for x in rows]))),
            "median_abs_rho": float(np.median(np.abs([x["rho_ng_given_tau"] for x in rows]))),
            "max_cond_corr_ndg": float(np.max([x["cond_corr_ndg"] for x in rows])),
            "median_cond_corr_ndg": float(np.median([x["cond_corr_ndg"] for x in rows])),
            "median_cond_corr_nd": float(np.median([x["cond_corr_nd"] for x in rows])),
            "median_abs_corr_n_d": float(np.median(np.abs([x["corr_n_d"] for x in rows]))),
            "frac_prior_n_within_0p05_of_n_star": float(np.mean(np.abs(ngrid - nstar) < 0.05))}


def curves(ns, d=650.0):
    ng = np.linspace(1.10, 2.40, 1301)
    keep = np.abs(ng - ns) > 0.004
    ng = ng[keep]
    rows = [decompose(n, d, ns) for n in ng]
    vis = []
    for n in ng:
        r01, r12 = (N0 - n) / (N0 + n), (n - ns) / (n + ns)
        Rp, Rm = (r01 + r12) ** 2 / (1 + r01 * r12) ** 2, (r01 - r12) ** 2 / (1 - r01 * r12) ** 2
        vis.append(abs(Rp - Rm) / (Rp + Rm))
    return {"n_s": ns, "d_nm": d, "n": ng.tolist(), "visibility": vis,
            "crb_sd_n_known": [x["crb_sd_n_known"] for x in rows], "crb_sd_n_unknown": [x["crb_sd_n_unknown"] for x in rows],
            "sd_ratio": [x["sd_ratio"] for x in rows], "rho_ng_given_tau": [x["rho_ng_given_tau"] for x in rows]}


def main():
    out = {"derivative_checks": derivative_checks()}
    print("R check", out["derivative_checks"]["max_abs_R_difference_vs_optics_reflectance"])
    for row in out["derivative_checks"]["central_difference_table"]:
        print("  step %.0e: rel err dR/dn %.2e dR/dd %.2e" % (row["step_n"], row["max_rel_err_dRdn"], row["max_rel_err_dRdd"]))
    print("  fisher_theory steps: max rel change in ratio", out["derivative_checks"]["fisher_theory_steps_max_rel_change_in_sd_ratio"])

    # identities and the fixed point
    chk = [decompose(n, d, ns) for n, d, ns in ((1.8, 650, 1.5), (1.45, 650, 1.5), (2.0, 650, 2.3), (1.9, 1200, 1.75))]
    out["identity_checks"] = {"films": len(chk), "max_rel_diff_sd_ratio_rho_vs_projection": float(max(abs(c["sd_ratio_from_rho"] / c["sd_ratio"] - 1) for c in chk)),
                              "max_rel_diff_sd_ratio_ntau_vs_nd": float(max(abs(c["sd_ratio_from_nd_fisher"] / c["sd_ratio"] - 1) for c in chk))}
    fp = {}
    for ns in (1.5, 1.75, 2.0, 2.3):
        nstar = np.sqrt(N0 * ns)
        _, Fn, _ = airy(LAM, nstar, nstar * 650.0, ns)
        _, Fn2, _ = airy(LAM, nstar + 0.05, (nstar + 0.05) * 650.0, ns)
        fp[f"{ns:.2f}"] = {"n_star": float(nstar), "max_abs_fresnel_vector_at_n_star": float(np.max(np.abs(Fn))),
                           "max_abs_fresnel_vector_at_n_star_plus_0p05": float(np.max(np.abs(Fn2)))}
    out["fixed_point"] = fp
    print("identities", out["identity_checks"]); print("fixed point", fp)

    # the two reference films used in the text
    out["reference_films"] = {"n1p80_d650_ns1p5": decompose(1.8, 650.0, 1.5), "n1p45_d650_ns1p5": decompose(1.45, 650.0, 1.5),
                              "n2p00_d650_ns2p30": decompose(2.0, 650.0, 2.3)}
    for k, v in out["reference_films"].items():
        print(k, {q: round(v[q], 4) for q in ("sd_ratio", "rho_ng_given_tau", "corr_n_d", "crb_sd_n_known", "crb_sd_n_unknown", "cond_corr_ndg")})

    # sweep axes (sigma cancels in the ratio, so the noise axis is one value)
    axes = {"fringe_count": [band(lo, hi, 1.5) for lo, hi in ((150, 350), (350, 650), (600, 1000), (1000, 1600), (1600, 2400))],
            "substrate_index": [band(400, 900, ns) for ns in (1.35, 1.5, 1.75, 2.0, 2.3)]}
    out["axes"] = axes
    for ax, rows in axes.items():
        for r in rows:
            print(ax, r["d_band"], r["n_s"], "median sd ratio %.2f [%.2f, %.2f] rms-bound ratio %.2f cond ndg median %.0f max %.0f |rho_nd| %.3f" % (
                r["sd_ratio_median"], r["sd_ratio_p25"], r["sd_ratio_p75"], r["sd_ratio_rms_bound_ratio"], r["median_cond_corr_ndg"], r["max_cond_corr_ndg"], r["median_abs_corr_n_d"]))

    # per-film bounds for the E1 films, against their Monte Carlo errors
    tr = np.load(os.path.join(RESULTS, "rev_matched_trials.npz"))
    films = tr["e12_e1_truth"]
    ek, ep = tr["e12_e1_known_n_mean_err"], tr["e12_e1_profile_n_mean_err"]
    dec = [decompose(n, d, 1.5) for n, d in films]
    sk = np.array([x["crb_sd_n_known"] for x in dec]); su = np.array([x["crb_sd_n_unknown"] for x in dec])
    ratio = su / sk
    terc = np.digitize(ratio, np.percentile(ratio, [100 / 3, 200 / 3]))
    by = {}
    for j in range(3):
        m = terc == j
        by[f"tercile_{j + 1}"] = {"count": int(m.sum()), "sd_ratio_range": [float(ratio[m].min()), float(ratio[m].max())],
                                  "mc_rmse_known": float(np.sqrt(np.mean(ek[m] ** 2))), "mc_rmse_profile": float(np.sqrt(np.mean(ep[m] ** 2))),
                                  "mc_rmse_ratio": float(np.sqrt(np.mean(ep[m] ** 2) / np.mean(ek[m] ** 2))),
                                  "rms_crb_known": float(np.sqrt(np.mean(sk[m] ** 2))), "rms_crb_unknown": float(np.sqrt(np.mean(su[m] ** 2)))}
    out["e1_films"] = {"count": int(len(films)), "sd_ratio_median": float(np.median(ratio)),
                       "sd_ratio_p90": float(np.percentile(ratio, 90)), "frac_sd_ratio_gt_10": float(np.mean(ratio > 10)),
                       "rms_crb_known": float(np.sqrt(np.mean(sk ** 2))), "rms_crb_unknown": float(np.sqrt(np.mean(su ** 2))),
                       "mc_rmse_known": float(np.sqrt(np.mean(ek ** 2))), "mc_rmse_profile": float(np.sqrt(np.mean(ep ** 2))),
                       "prior_width_n": N_HI - N_LO, "by_sd_ratio_tercile": by}
    n_true = films[:, 0]
    pmap = n_true + tr["e12_e1_profile_n_map_err"]
    bands = {}
    for lo, hi in ((1.35, 1.45), (1.45, 1.55), (1.55, 1.65), (1.65, 1.85), (1.85, 2.15)):
        m = (n_true >= lo) & (n_true < hi)
        bands[f"[{lo:.2f},{hi:.2f})"] = {"count": int(m.sum()), "sd_ratio_median": float(np.median(ratio[m])),
                                          "mc_rmse_known": float(np.sqrt(np.mean(ek[m] ** 2))), "mc_rmse_profile": float(np.sqrt(np.mean(ep[m] ** 2))),
                                          "mc_rmse_ratio": float(np.sqrt(np.mean(ep[m] ** 2) / np.mean(ek[m] ** 2))),
                                          "frac_abs_profile_err_gt_0p1": float(np.mean(np.abs(ep[m]) > 0.1))}
    low = n_true < 1.45
    out["e1_films"]["by_true_index"] = bands
    out["e1_films"]["films_below_1p45_profile_map_above_ns"] = {"count": int(np.sum(pmap[low] > 1.5)), "of": int(low.sum()),
                                                                "fraction": float(np.mean(pmap[low] > 1.5))}
    print("E1 films", {k: v for k, v in out["e1_films"].items() if k not in ("by_sd_ratio_tercile", "by_true_index")})
    for k, v in list(by.items()) + list(bands.items()):
        print("  ", k, {q: (round(x, 4) if isinstance(x, float) else x) for q, x in v.items()})

    # linearized gain-error bias on the rev_structural films, and the calibration tolerance on the E1 films
    rng = np.random.default_rng(21)
    films150 = [(rng.uniform(1.35, 2.15), rng.uniform(400, 900)) for _ in range(150)]
    bn = np.array([linear_gain_bias(n, d, 1.5)[0] for n, d in films150])
    bd = np.array([linear_gain_bias(n, d, 1.5)[1] for n, d in films150])
    matched = np.array([abs(n - 1.5) < 0.03 for n, _ in films150])
    struct = json.load(open(os.path.join(RESULTS, "rev_structural.json")))["gain_error_bias"]["by_gain_error"]
    bE1 = np.array([abs(linear_gain_bias(n, d, 1.5)[0]) for n, d in films])
    tol = sk / bE1
    out["gain_error"] = {"linear_median_abs_bias_n_per_1pct": float(0.01 * np.median(np.abs(bn))),
                         "linear_median_abs_bias_d_nm_per_1pct": float(0.01 * np.median(np.abs(bd))),
                         "nonlinear_median_abs_bias_n_at_minus_1pct": struct["-0.01"]["median_abs_bias_n"],
                         "nonlinear_median_abs_bias_n_at_plus_1pct": struct["+0.01"]["median_abs_bias_n"],
                         "nonlinear_median_abs_bias_n_at_plus_10pct": struct["+0.10"]["median_abs_bias_n"],
                         "n_index_matched_films_150": int(matched.sum()),
                         "tolerance_note": "gain error at which the first-order index bias equals the calibrated Cramer-Rao sd at sigma = 0.02, per E1 film",
                         "tolerance_median": float(np.median(tol)), "tolerance_p10": float(np.percentile(tol, 10)),
                         "tolerance_p90": float(np.percentile(tol, 90)),
                         "e1_median_abs_bias_n_per_1pct": float(0.01 * np.median(bE1)),
                         "e3_share_positive_index_coefficient": float(np.mean([linear_gain_bias(n, d, 1.5)[0] > 0 for n, d in tr["e3_truth"][:, :2]])),
                         "e3_films": int(len(tr["e3_truth"]))}
    print("gain error", out["gain_error"])

    out["curves"] = [curves(ns) for ns in (1.5, 2.3)]
    out["config"] = {"lam_nm": [float(LAM[0]), float(LAM[-1]), len(LAM)], "sigma_for_bounds": SIGMA, "n0": N0,
                     "band_grid": "20 x 20 cell midpoints over n in [1.35, 2.15] and the thickness band",
                     "curves": "n in [1.10, 2.40], 1301 points, points within 0.004 of n_s dropped, d = 650 nm"}
    with open(os.path.join(RESULTS, "rev_fisher.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("done")


if __name__ == "__main__":
    main()

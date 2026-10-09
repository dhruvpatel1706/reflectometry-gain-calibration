"""
Matched-model experiments (29 Sep 2026), with per-trial outputs saved.

Experiments:
  E1  calibrated (gain known) versus shape-only (gain profiled) versus Bayesian unknown gain (gain marginalized) on
      the same spectra; three point estimators (posterior or weight mean, median, MAP); paired bootstrap intervals
      for every RMSE ratio.
  E2  interval coverage under the correct and misstated noise, with mean interval widths, and coverage within
      terciles of the true index and thickness.
  E3  a true gain drawn from U[0.85, 1.20], each spectrum analysed three ways: gain fixed at its true value (the
      calibrated baseline), fixed at one (naive), and marginalized under U[0.5, 1.5]; RMSE, bias, coverage and
      widths overall and by |g - 1|.
  E3b the same with noise standard deviation g*sigma instead of sigma (a reference-scale error scales signal and
      noise together, so the gain no longer changes the signal-to-noise ratio).
  GC  E1 and E2 repeated on a grid twice as fine in each direction.
Writes data/results/rev_matched.json and data/results/rev_matched_trials.npz. CPU-only, deterministic.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import optics as O
import fastgrid as F

RESULTS = os.path.join(os.path.dirname(__file__), "..", "data", "results")
SIGMA, N_SIMS, N_S = 0.02, 600, 1.5
N_LO, N_HI, D_LO, D_HI = 1.35, 2.15, 400.0, 900.0
GAIN_LO, GAIN_HI, PRIOR = 0.85, 1.20, (0.5, 1.5)
N_GRID, D_GRID = np.linspace(1.20, 2.30, 330), np.linspace(350.0, 950.0, 1400)
MEAN_N, MEAN_D = 0.5 * (N_LO + N_HI), 0.5 * (D_LO + D_HI)
MEAN = {"n": MEAN_N, "d": MEAN_D, "tau": MEAN_N * MEAN_D}
LAM = O.default_grid()
B_BOOT = 2000
PARS = ("n", "d", "tau")


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def boot_ratio(ea, eb, seed=7):
    """Paired bootstrap of RMSE(ea) / RMSE(eb) over trials: point value and 95% percentile interval."""
    ea, eb = np.asarray(ea), np.asarray(eb)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, ea.size, size=(B_BOOT, ea.size))
    r = np.sqrt(np.mean(ea[idx] ** 2, axis=1)) / np.sqrt(np.mean(eb[idx] ** 2, axis=1))
    return {"ratio": rmse(ea) / rmse(eb), "ci95": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}


def boot_rmse(e, seed=8):
    e = np.asarray(e); rng = np.random.default_rng(seed)
    idx = rng.integers(0, e.size, size=(B_BOOT, e.size))
    r = np.sqrt(np.mean(e[idx] ** 2, axis=1))
    return {"rmse": rmse(e), "ci95": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}


class Rec:
    """Per-trial record of one analysis."""
    def __init__(self):
        self.d = {}

    def add(self, s, truth):
        for p in PARS:
            for est in ("mean", "median", "map"):
                self.d.setdefault(f"{p}_{est}_err", []).append(s[f"{p}_{est}"] - truth[p])
            for q in (68, 90):
                lo, hi = s[f"{p}_ci{q}"]
                self.d.setdefault(f"{p}_cov{q}", []).append(float(lo <= truth[p] <= hi))
                self.d.setdefault(f"{p}_width{q}", []).append(hi - lo)

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.d.items()}


def summary(a, credible=True):
    out = {}
    for p in PARS:
        e = a[f"{p}_mean_err"]
        out[p] = {"rmse": rmse(e), "rel_rmse": rmse(e) / MEAN[p], "bias": float(np.mean(e)),
                  "rmse_median_est": rmse(a[f"{p}_median_err"]), "rmse_map_est": rmse(a[f"{p}_map_err"])}
        if credible:
            for q in (68, 90):
                c = a[f"{p}_cov{q}"]
                out[p][f"cov{q}"] = float(np.mean(c)); out[p][f"cov{q}_se"] = float(np.sqrt(np.mean(c) * (1 - np.mean(c)) / c.size))
                out[p][f"mean_width{q}"] = float(np.mean(a[f"{p}_width{q}"]))
    return out


def draws(seed, with_gain):
    rng = np.random.default_rng(seed)
    for _ in range(N_SIMS):
        nt, dt = rng.uniform(N_LO, N_HI), rng.uniform(D_LO, D_HI)
        g = rng.uniform(GAIN_LO, GAIN_HI) if with_gain else 1.0
        yield nt, dt, g, rng


def e1_e2(G, tag=""):
    recs = {k: Rec() for k in ("known", "profile", "marginal", "half", "double")}
    truths, ghat_map, modul = [], [], []
    for nt, dt, _, rng in draws(0, False):
        R = O.reflectance(LAM, nt, dt, n_s=N_S)
        y = R + rng.normal(0.0, SIGMA, size=R.shape)
        truth = {"n": nt, "d": dt, "tau": nt * dt}
        truths.append((nt, dt)); modul.append(float(np.ptp(R)))
        for k, gain, sig in (("known", ("fixed", 1.0), SIGMA), ("profile", ("profile",), SIGMA),
                             ("marginal", ("marginal",) + PRIOR, SIGMA), ("half", ("fixed", 1.0), SIGMA / 2),
                             ("double", ("fixed", 1.0), SIGMA * 2)):
            ll, SyM = G.loglik(y, sig, gain)
            w = G.weights(ll)
            recs[k].add(G.summarize(w, names=PARS), truth)
            if k == "profile":
                i = int(np.argmax(w)); ghat_map.append(float(SyM[i] / G.SMM[i]))
    A = {k: r.arrays() for k, r in recs.items()}
    tr = np.asarray(truths)
    e1 = {"calibrated_known_gain": summary(A["known"]), "shape_only_profiled_gain": summary(A["profile"], credible=False),
          "unknown_gain_marginalized": summary(A["marginal"])}
    ratios = {}
    for est in ("mean", "median", "map"):
        ratios[f"profile_over_known_n_{est}"] = boot_ratio(A["profile"][f"n_{est}_err"], A["known"][f"n_{est}_err"])
        ratios[f"marginal_over_known_n_{est}"] = boot_ratio(A["marginal"][f"n_{est}_err"], A["known"][f"n_{est}_err"])
    ratios["profile_over_known_tau_mean"] = boot_ratio(A["profile"]["tau_mean_err"], A["known"]["tau_mean_err"])
    ratios["profile_over_known_d_mean"] = boot_ratio(A["profile"]["d_mean_err"], A["known"]["d_mean_err"])
    e1["ratios"] = ratios
    e1["profiled_gain_at_map_cell"] = {"min": float(np.min(ghat_map)), "max": float(np.max(ghat_map)), "n_nonpositive": int(np.sum(np.asarray(ghat_map) <= 0))}
    e1["fringe_modulation"] = {"note": "peak-to-peak reflectance over the band, per film", "median": float(np.median(modul)),
                               "q25": float(np.percentile(modul, 25)), "q75": float(np.percentile(modul, 75)),
                               "median_over_sigma": float(np.median(modul) / SIGMA)}
    e2 = {k: summary(A[k]) for k in ("known", "half", "double")}
    terc = {}
    for p, col in (("n", 0), ("d", 1)):
        cuts = np.percentile(tr[:, col], [100 / 3, 200 / 3])
        b = np.digitize(tr[:, col], cuts)
        terc[p] = {f"tercile_{j + 1}": {f"{q}_cov90": float(np.mean(A["known"][f"{q}_cov90"][b == j])) for q in PARS} | {"count": int(np.sum(b == j))} for j in range(3)}
    e2["coverage_by_true_parameter_tercile_correct_noise"] = terc
    trials = {f"e1_{k}_{kk}": v for k in ("known", "profile", "marginal") for kk, v in A[k].items()}
    trials.update({f"e2_{k}_{kk}": v for k in ("half", "double") for kk, v in A[k].items() if "cov" in kk or "width" in kk})
    trials["e1_truth"] = tr
    return e1, e2, trials


def e3(G, scaled):
    recs = {k: Rec() for k in ("true_gain", "gain_one", "marginal", "profile")}
    truths = []
    for nt, dt, g, rng in draws(1, True):
        R = O.reflectance(LAM, nt, dt, n_s=N_S)
        y = g * R + rng.normal(0.0, SIGMA * (g if scaled else 1.0), size=R.shape)
        truth = {"n": nt, "d": dt, "tau": nt * dt}
        truths.append((nt, dt, g))
        for k, gain in (("true_gain", ("fixed", g)), ("gain_one", ("fixed", 1.0)), ("marginal", ("marginal",) + PRIOR), ("profile", ("profile",))):
            ll, _ = G.loglik(y, SIGMA, gain, scaled_noise=scaled and k != "profile")
            recs[k].add(G.summarize(G.weights(ll), names=PARS), truth)
    A = {k: r.arrays() for k, r in recs.items()}
    tr = np.asarray(truths)
    out = {k: summary(A[k], credible=(k != "profile")) for k in A}
    out["ratios"] = {"marginal_over_true_gain_n": boot_ratio(A["marginal"]["n_mean_err"], A["true_gain"]["n_mean_err"]),
                     "marginal_over_true_gain_d": boot_ratio(A["marginal"]["d_mean_err"], A["true_gain"]["d_mean_err"]),
                     "gain_one_over_true_gain_n": boot_ratio(A["gain_one"]["n_mean_err"], A["true_gain"]["n_mean_err"]),
                     "profile_over_true_gain_n": boot_ratio(A["profile"]["n_mean_err"], A["true_gain"]["n_mean_err"])}
    dev = np.abs(tr[:, 2] - 1.0)
    bins = {"|g-1|<0.05": dev < 0.05, "0.05<=|g-1|<0.10": (dev >= 0.05) & (dev < 0.10), "|g-1|>=0.10": dev >= 0.10}
    out["by_gain_error"] = {}
    for name, m in bins.items():
        out["by_gain_error"][name] = {"count": int(m.sum())}
        for k in ("true_gain", "gain_one", "marginal"):
            out["by_gain_error"][name][k] = {f"{p}_cov90": float(np.mean(A[k][f"{p}_cov90"][m])) for p in PARS} | \
                {f"{p}_rmse": rmse(A[k][f"{p}_mean_err"][m]) for p in ("n", "d")} | {"n_bias": float(np.mean(A[k]["n_mean_err"][m]))}
    below, above = tr[:, 2] < 1.0, tr[:, 2] > 1.0
    out["gain_one_bias_by_sign"] = {"g<1": {"n_bias": float(np.mean(A["gain_one"]["n_mean_err"][below])), "d_bias": float(np.mean(A["gain_one"]["d_mean_err"][below])), "count": int(below.sum())},
                                    "g>1": {"n_bias": float(np.mean(A["gain_one"]["n_mean_err"][above])), "d_bias": float(np.mean(A["gain_one"]["d_mean_err"][above])), "count": int(above.sum())}}
    trials = {f"{k}_{kk}": v for k in A for kk, v in A[k].items()}
    trials["truth"] = tr
    return out, trials


def main():
    t0 = time.time()
    M, P = F.airy_matrix(LAM, N_GRID, D_GRID, n_s=N_S)
    G = F.Grid(M, P); del M
    e1, e2, tr12 = e1_e2(G)
    e3a, tr3 = e3(G, scaled=False)
    e3b, tr3b = e3(G, scaled=True)
    del G
    Mf, Pf = F.airy_matrix(LAM, np.linspace(1.20, 2.30, 659), np.linspace(350.0, 950.0, 2799), n_s=N_S)
    Gf = F.Grid(Mf, Pf); del Mf
    gc1, gc2, _ = e1_e2(Gf)
    del Gf
    gc = {"grid": {"n_pts": 659, "d_pts": 2799, "note": "spacing halved in n and d"},
          "E1_calibrated": gc1["calibrated_known_gain"], "E1_shape_only": gc1["shape_only_profiled_gain"],
          "E1_ratio_profile_over_known_n_mean": gc1["ratios"]["profile_over_known_n_mean"], "E2_correct": gc2["known"]}
    res = {"config": {"sigma": SIGMA, "n_sims": N_SIMS, "n_substrate": N_S, "n_range": [N_LO, N_HI], "d_range_nm": [D_LO, D_HI],
                      "true_gain_range": [GAIN_LO, GAIN_HI], "gain_prior_uniform": list(PRIOR), "n_grid": [1.20, 2.30, len(N_GRID)],
                      "d_grid_nm": [350.0, 950.0, len(D_GRID)], "n_wavelengths": len(LAM), "bootstrap_resamples": B_BOOT,
                      "seeds": {"E1_E2": 0, "E3": 1}},
           "E1": e1, "E2": e2, "E3_fixed_noise": e3a, "E3b_noise_scales_with_gain": e3b, "grid_convergence": gc}
    with open(os.path.join(RESULTS, "rev_matched.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    trials = {**{f"e12_{k}": v for k, v in tr12.items()}, **{f"e3_{k}": v for k, v in tr3.items()}, **{f"e3b_{k}": v for k, v in tr3b.items()}}
    np.savez_compressed(os.path.join(RESULTS, "rev_matched_trials.npz"), **{k: np.asarray(v, dtype=np.float64) for k, v in trials.items()})
    print(f"done in {time.time() - t0:.0f}s")
    k, p, m = e1["calibrated_known_gain"], e1["shape_only_profiled_gain"], e1["unknown_gain_marginalized"]
    print("E1 rel n: known %.4f profile %.4f marginal %.4f" % (k["n"]["rel_rmse"], p["n"]["rel_rmse"], m["n"]["rel_rmse"]))
    print("E1 ratios", {a: (round(b["ratio"], 2), [round(x, 2) for x in b["ci95"]]) for a, b in e1["ratios"].items()})
    print("E1 ghat", e1["profiled_gain_at_map_cell"], "modulation", e1["fringe_modulation"])
    print("E2 correct cov", {q: (e2["known"][q]["cov68"], e2["known"][q]["cov90"], round(e2["known"][q]["mean_width90"], 4)) for q in PARS})
    for tag, E in (("E3", e3a), ("E3b", e3b)):
        for kk in ("true_gain", "gain_one", "marginal"):
            print(tag, kk, {q: (round(E[kk][q]["cov90"], 3), round(E[kk][q]["rmse"], 4), round(E[kk][q]["mean_width90"], 4)) for q in PARS})
        print(tag, "ratios", {a: (round(b["ratio"], 2), [round(x, 2) for x in b["ci95"]]) for a, b in E["ratios"].items()})
    print("GC", gc["E1_calibrated"]["n"]["rel_rmse"], gc["E1_shape_only"]["n"]["rel_rmse"], gc["E2_correct"]["n"]["cov68"], gc["E2_correct"]["n"]["cov90"])


if __name__ == "__main__":
    main()

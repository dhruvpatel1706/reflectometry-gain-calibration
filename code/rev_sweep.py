"""
Branch-resolved identifiability sweep (29 Sep 2026), with per-trial outputs and bootstrap intervals.

Three axes (fringe count through the thickness band, noise level, substrate index), 400 films per point, known
gain versus profiled gain. Two details matter.
  * The exact swap symmetry n -> n0 ns / n, d -> n^2 d / (n0 ns) gives every film a mirror twin with an identical
    spectrum. For each film we record whether its twin lies inside the inference grid and the posterior mass on the
    twin's side of the fixed point n* = sqrt(n0 ns). Each analysis is run twice: on the full grid, and with the prior
    restricted to the side of n* that contains the truth (a branch-resolved analysis, as if the analyst knew
    whether the film index is above or below sqrt(n0 ns)).
  * Every RMSE ratio carries a paired bootstrap 95% interval; the per-film fringe modulation (peak-to-peak
    reflectance over the band, divided by sigma) replaces the nominal 0.5 / sigma.
Two points (n_s = 1.35 and 2.30) are repeated on a grid with half the spacing in both directions.
Writes data/results/rev_sweep.json and data/results/rev_sweep_trials.npz. CPU-only, deterministic.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import optics as O
import fastgrid as F
from rev_matched import boot_ratio, rmse

RESULTS = os.path.join(os.path.dirname(__file__), "..", "data", "results")
LAM = O.default_grid()
N_SIMS = 400
N_LO, N_HI = 1.35, 2.15
MEAN_N = 0.5 * (N_LO + N_HI)
N0 = 1.0


def grids(d_lo, d_hi, fine=False):
    pad = 0.12 * (d_hi - d_lo) + 60.0
    return (np.linspace(1.20, 2.30, 479 if fine else 240),
            np.linspace(max(20.0, d_lo - pad), d_hi + pad, 1799 if fine else 900))


def point(d_lo, d_hi, sigma, ns, seed, fine=False):
    ng, dg = grids(d_lo, d_hi, fine)
    M, P = F.airy_matrix(LAM, ng, dg, n_s=ns)
    G = F.Grid(M, P); del M
    nstar = np.sqrt(N0 * ns)
    above = G.params["n"] > nstar
    rng = np.random.default_rng(seed)
    rec = {k: [] for k in ("n_true", "d_true", "mirror_n", "mirror_in_grid", "modulation_over_sigma",
                           "known_full_n", "known_full_d", "known_full_tau", "profile_full_n", "profile_full_d", "profile_full_tau",
                           "known_branch_n", "known_branch_d", "profile_branch_n", "profile_branch_d",
                           "known_full_mirror_mass", "profile_full_mirror_mass")}
    for _ in range(N_SIMS):
        nt, dt = rng.uniform(N_LO, N_HI), rng.uniform(d_lo, d_hi)
        R = O.reflectance(LAM, nt, dt, n_s=ns)
        y = R + rng.normal(0.0, sigma, size=R.shape)
        nm = N0 * ns / nt
        side = above if nt > nstar else ~above
        rec["n_true"].append(nt); rec["d_true"].append(dt); rec["mirror_n"].append(nm)
        rec["mirror_in_grid"].append(float(ng[0] <= nm <= ng[-1])); rec["modulation_over_sigma"].append(float(np.ptp(R)) / sigma)
        for k, gain in (("known", ("fixed", 1.0)), ("profile", ("profile",))):
            ll, _ = G.loglik(y, sigma, gain)
            w = G.weights(ll)
            s = G.summarize(w, levels=(), names=("n", "d", "tau"))
            rec[f"{k}_full_n"].append(s["n_mean"] - nt); rec[f"{k}_full_d"].append(s["d_mean"] - dt); rec[f"{k}_full_tau"].append(s["tau_mean"] - nt * dt)
            rec[f"{k}_full_mirror_mass"].append(float(w[~side].sum()))
            wb = G.weights(np.where(side, ll, -np.inf))
            sb = G.summarize(wb, levels=(), names=("n", "d"))
            rec[f"{k}_branch_n"].append(sb["n_mean"] - nt); rec[f"{k}_branch_d"].append(sb["d_mean"] - dt)
    a = {k: np.asarray(v) for k, v in rec.items()}
    mean_d = 0.5 * (d_lo + d_hi)
    out = {"d_band": [d_lo, d_hi], "sigma": sigma, "n_s": ns, "n_star": float(nstar), "seed": seed, "grid": [len(ng), len(dg)],
           "fringes_mid": float(2.0 * MEAN_N * mean_d * (1 / LAM.min() - 1 / LAM.max())),
           "modulation_over_sigma_median": float(np.median(a["modulation_over_sigma"])),
           "modulation_over_sigma_iqr": [float(np.percentile(a["modulation_over_sigma"], 25)), float(np.percentile(a["modulation_over_sigma"], 75))],
           "frac_mirror_in_grid": float(np.mean(a["mirror_in_grid"])),
           "mean_mirror_mass_known": float(np.mean(a["known_full_mirror_mass"])),
           "frac_mirror_mass_gt_0p1_known": float(np.mean(a["known_full_mirror_mass"] > 0.1))}
    for scope in ("full", "branch"):
        for k in ("known", "profile"):
            out[f"rel_n_{k}_{scope}"] = rmse(a[f"{k}_{scope}_n"]) / MEAN_N
            out[f"rel_d_{k}_{scope}"] = rmse(a[f"{k}_{scope}_d"]) / mean_d
        out[f"rho_sep_{scope}"] = boot_ratio(a[f"profile_{scope}_n"], a[f"known_{scope}_n"], seed=seed + 1)
    out["rel_tau_known_full"] = rmse(a["known_full_tau"]) / (MEAN_N * mean_d)
    out["rel_tau_profile_full"] = rmse(a["profile_full_tau"]) / (MEAN_N * mean_d)
    return out, a


def main():
    t0 = time.time()
    axes = {"fringe_count": [((150.0, 350.0), 0.02, 1.5), ((350.0, 650.0), 0.02, 1.5), ((600.0, 1000.0), 0.02, 1.5),
                             ((1000.0, 1600.0), 0.02, 1.5), ((1600.0, 2400.0), 0.02, 1.5)],
            "noise": [((400.0, 900.0), s, 1.5) for s in (0.005, 0.01, 0.02, 0.04, 0.08)],
            "substrate_index": [((400.0, 900.0), 0.02, ns) for ns in (1.35, 1.50, 1.75, 2.00, 2.30)]}
    res, trials = {"axes": {}}, {}
    seed, substrate_seed = 100, {}
    for ax, pts in axes.items():
        res["axes"][ax] = []
        for (dlo, dhi), s, ns in pts:
            if ax == "substrate_index":
                substrate_seed[ns] = seed
            o, a = point(dlo, dhi, s, ns, seed)
            res["axes"][ax].append(o)
            for k, v in a.items():
                trials[f"{ax}_{seed}_{k}"] = v
            print(f"{ax:16s} d[{dlo:.0f},{dhi:.0f}] s={s} ns={ns}: rho full {o['rho_sep_full']['ratio']:.2f} {[round(x, 2) for x in o['rho_sep_full']['ci95']]} "
                  f"branch {o['rho_sep_branch']['ratio']:.2f} {[round(x, 2) for x in o['rho_sep_branch']['ci95']]} rel_n known full/branch "
                  f"{o['rel_n_known_full']:.4f}/{o['rel_n_known_branch']:.4f} mirror in grid {o['frac_mirror_in_grid']:.2f} mass {o['mean_mirror_mass_known']:.3f}", flush=True)
            seed += 10
    res["grid_convergence"] = []
    for ns in (1.35, 2.30):
        o, _ = point(400.0, 900.0, 0.02, ns, substrate_seed[ns], fine=True)
        res["grid_convergence"].append(o)
        print(f"fine grid ns={ns}: rho full {o['rho_sep_full']['ratio']:.3f} branch {o['rho_sep_branch']['ratio']:.3f}", flush=True)
    res["config"] = {"n_sims": N_SIMS, "n_range": [N_LO, N_HI], "lam_nm": [float(LAM[0]), float(LAM[-1]), len(LAM)],
                     "grid": "n: 240 points on [1.20, 2.30]; d: 900 points on the band padded by 12% + 60 nm (fine: 479 x 1799)",
                     "bootstrap_resamples": 2000, "seeds": "100 + 10 k for the k-th point in axis order; fine-grid points reuse the seeds of the matching coarse points"}
    with open(os.path.join(RESULTS, "rev_sweep.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    np.savez_compressed(os.path.join(RESULTS, "rev_sweep_trials.npz"), **trials)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

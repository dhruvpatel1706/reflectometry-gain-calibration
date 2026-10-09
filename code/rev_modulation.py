"""
E1 errors by fringe modulation (29 Sep 2026). The modulation of a film is the peak-to-peak reflectance over the band
divided by the noise standard deviation (sigma = 0.02), measured per film rather than set to a nominal 0.5 / sigma.
Reads data/results/rev_matched_trials.npz (the 600 E1 films and their per-trial errors); writes
data/results/rev_modulation.json. Deterministic.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import optics as O

RESULTS = os.path.join(os.path.dirname(__file__), "..", "data", "results")
SIGMA = 0.02


def main():
    tr = np.load(os.path.join(RESULTS, "rev_matched_trials.npz"))
    films = tr["e12_e1_truth"]
    lam = O.default_grid()
    mod = np.array([np.ptp(O.reflectance(lam, n, d, n_s=1.5)) for n, d in films]) / SIGMA
    rms = lambda e: float(np.sqrt(np.mean(np.square(e))))
    bands = []
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 8), (8, np.inf)):
        m = (mod >= lo) & (mod < hi)
        row = {"modulation_over_sigma": [lo, None if np.isinf(hi) else hi], "count": int(m.sum()),
               "true_index_range": [float(films[m, 0].min()), float(films[m, 0].max())]}
        for k in ("known", "profile", "marginal"):
            row[f"{k}_rmse_n"] = rms(tr[f"e12_e1_{k}_n_mean_err"][m])
            row[f"{k}_rmse_tau_nm"] = rms(tr[f"e12_e1_{k}_tau_mean_err"][m])
            row[f"{k}_median_abs_tau_err_nm"] = float(np.median(np.abs(tr[f"e12_e1_{k}_tau_mean_err"][m])))
        row["profile_over_known_rmse_n"] = row["profile_rmse_n"] / row["known_rmse_n"]
        bands.append(row)
    out = {"sigma": SIGMA, "n_films": int(len(films)), "modulation_over_sigma_median": float(np.median(mod)),
           "modulation_over_sigma_q25_q75": [float(np.percentile(mod, 25)), float(np.percentile(mod, 75))],
           "frac_modulation_below_sigma": float(np.mean(mod < 1)), "by_modulation": bands,
           "known_median_abs_tau_err_nm_all": float(np.median(np.abs(tr["e12_e1_known_tau_mean_err"])))}
    with open(os.path.join(RESULTS, "rev_modulation.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for b in bands:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()})
    print("median modulation/sigma", out["modulation_over_sigma_median"], "below sigma", out["frac_modulation_below_sigma"])


if __name__ == "__main__":
    main()

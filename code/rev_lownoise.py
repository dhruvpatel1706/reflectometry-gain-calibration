"""
Low-noise check (29 Sep 2026): continuous least-squares fits, without a grid, on the 600 E1 films of rev_matched.py at
several noise levels, compared with the local (Cramer-Rao) prediction of rev_fisher.py.

Each spectrum y = R(n, d) + noise (gain one) is fitted twice by Levenberg-Marquardt started at the true parameters:
with the gain known (parameters n, d) and with the gain profiled (n, d, g). Starting at the truth measures local
precision; a fit cannot reach the far mode, so the check describes the small-noise regime in which the local analysis
applies, not the multimodal behaviour of the grid analysis at sigma = 0.02. For each sigma: index RMSE for both
treatments, their ratio with a paired bootstrap 95% interval, the ratio of root-mean-square Cramer-Rao bounds (the
local prediction, the same at every sigma), the share of profiled fits that end more than 0.05 from the true index,
and the calibration tolerance (the gain error at which the first-order index bias of a gain-one analysis equals the
calibrated bound), which scales with sigma.
Writes data/results/rev_lownoise.json. CPU-only, deterministic.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.optimize import least_squares
import optics as O
from rev_fisher import decompose, linear_gain_bias, LAM, SIGMA as SIGMA_REF
from rev_matched import boot_ratio, rmse

RESULTS = os.path.join(os.path.dirname(__file__), "..", "data", "results")
NS = 1.5
SIGMAS = [0.001, 0.002, 0.005, 0.01, 0.02]
SEED = 61


def spectrum(n, d, g=1.0):
    return g * O.reflectance(LAM, n, d, n_s=NS)


def fit_known(y, n, d):
    r = least_squares(lambda p: spectrum(p[0], p[1]) - y, x0=[n, d], x_scale=[0.01, 1.0], method="lm", xtol=1e-12, ftol=1e-12)
    return r.x[0], r.success


def fit_profiled(y, n, d):
    r = least_squares(lambda p: spectrum(p[0], p[1], p[2]) - y, x0=[n, d, 1.0], x_scale=[0.01, 1.0, 0.01], method="lm", xtol=1e-12, ftol=1e-12)
    return r.x[0], r.success


def main():
    tr = np.load(os.path.join(RESULTS, "rev_matched_trials.npz"))
    films = tr["e12_e1_truth"][:, :2]
    dec = [decompose(n, d, NS) for n, d in films]
    sk0 = np.array([x["crb_sd_n_known"] for x in dec]); su0 = np.array([x["crb_sd_n_unknown"] for x in dec])
    b1 = np.array([abs(linear_gain_bias(n, d, NS)[0]) for n, d in films])
    predicted = float(np.sqrt(np.mean(su0 ** 2) / np.mean(sk0 ** 2)))
    rng = np.random.default_rng(SEED)
    rows = []
    for sigma in SIGMAS:
        ek, ep, ok = [], [], 0
        for n, d in films:
            y = spectrum(n, d) + rng.normal(0.0, sigma, LAM.size)
            nk, sk_ok = fit_known(y, n, d)
            npf, sp_ok = fit_profiled(y, n, d)
            ok += int(sk_ok and sp_ok)
            ek.append(nk - n); ep.append(npf - n)
        ek, ep = np.array(ek), np.array(ep)
        scale = sigma / SIGMA_REF
        sk, su = sk0 * scale, su0 * scale
        row = {"sigma": sigma, "films": int(len(films)), "fits_converged": ok,
               "rmse_n_known": rmse(ek), "rmse_n_profile": rmse(ep), "ratio_profile_over_known": boot_ratio(ep, ek),
               "rms_crb_known": float(np.sqrt(np.mean(sk ** 2))), "rms_crb_unknown": float(np.sqrt(np.mean(su ** 2))),
               "rmse_over_rms_crb_known": rmse(ek) / float(np.sqrt(np.mean(sk ** 2))),
               "rmse_over_rms_crb_unknown": rmse(ep) / float(np.sqrt(np.mean(su ** 2))),
               "share_profile_abs_err_gt_0p05": float(np.mean(np.abs(ep) > 0.05)),
               "tolerance_median": float(np.median(sk / b1)), "tolerance_p10": float(np.percentile(sk / b1, 10)),
               "tolerance_p90": float(np.percentile(sk / b1, 90))}
        rows.append(row)
        print("sigma %.3f  known %.5f  profile %.5f  ratio %.2f [%.2f, %.2f]  pred %.2f  known/CRB %.3f  prof/CRB %.3f  far %.3f  tol %.4f  conv %d" % (
            sigma, row["rmse_n_known"], row["rmse_n_profile"], row["ratio_profile_over_known"]["ratio"],
            *row["ratio_profile_over_known"]["ci95"], predicted, row["rmse_over_rms_crb_known"], row["rmse_over_rms_crb_unknown"],
            row["share_profile_abs_err_gt_0p05"], row["tolerance_median"], ok))
    out = {"config": {"n_substrate": NS, "films": "the 600 E1 films of rev_matched_trials.npz", "sigmas": SIGMAS, "seed": SEED,
                      "fit": "Levenberg-Marquardt from the true parameters (local precision)", "gain_true": 1.0},
           "predicted_ratio_rms_crb": predicted, "median_sd_ratio": float(np.median(su0 / sk0)), "by_sigma": rows}
    with open(os.path.join(RESULTS, "rev_lownoise.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("done")


if __name__ == "__main__":
    main()

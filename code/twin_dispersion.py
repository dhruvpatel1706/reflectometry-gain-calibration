"""
The twin of section 4.3 with a dispersive film.

For real, wavelength-independent indices the swap n' = n0 ns / n, d' = n^2 d / (n0 ns) leaves the normal-incidence
reflectance unchanged at any gain (rev_structural.py). For a dispersive film the swapped index n0 ns / n(lam) would
need the thickness n(lam)^2 d / (n0 ns), which depends on the wavelength, so no film of one thickness is an exact twin.

T1  How far from exact is the twin? The film of figure 2(a) (index 1.40 at the band centre, 700 nm, on a substrate
    of index 2.30, 120 wavelengths from 420 to 750 nm) is given the Cauchy dispersion of rev_dispersion.py,
    n(lam) = n_ref + B (1/lam^2 - 1/lam_c^2) with lam_c = 585 nm, for B = 0 (control), 1.2e4, 1.8e4 and 2.4e4 nm^2
    (the two ends and the middle of the range studied). The best film on the twin side of n* = sqrt(n0 ns) with the
    same two-parameter Cauchy form, (n_ref', B', d') free, is fitted to the noiseless spectrum, with the gain known
    (one) and with it profiled, by a grid search over the whole twin side followed by least squares from the best
    cells. The true film has RSS 0, so RSS / sigma^2 of the best twin-side film is the chi-square excess of the twin;
    it is reported at sigma = 0.02 and 0.002. A three-term Cauchy twin (an added C (1/lam^4 - 1/lam_c^4) term) is
    fitted as a sensitivity check.
T2  What does the swap preserve? For the non-dispersive film of figure 2(a) and its twin: |r|^2 and the phase of the
    amplitude reflection coefficient r at normal incidence, and the ellipsometric angles Psi and Delta at 70 degrees
    incidence, from an independent characteristic-matrix computation checked against the Airy form.
Writes data/results/twin_dispersion.json. CPU-only, deterministic, no random numbers.
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.optimize import least_squares
import optics as O

RESULTS = os.path.join(os.path.dirname(__file__), "..", "data", "results")
N0, NS, N_REF, D_TRUE = 1.0, 2.30, 1.40, 700.0
LAM = O.default_grid()
LAM_C = 0.5 * (LAM[0] + LAM[-1])
W2 = 1.0 / LAM ** 2 - 1.0 / LAM_C ** 2
W4 = 1.0 / LAM ** 4 - 1.0 / LAM_C ** 4
B_VALUES = (0.0, 1.2e4, 1.8e4, 2.4e4)
SIGMAS = (0.02, 0.002)
N_STAR = math.sqrt(N0 * NS)
THETA_ELLIPS = 70.0


def model(n_ref, B, d, C=0.0):
    return O.reflectance(LAM, n_ref + B * W2 + C * W4, d, n0=N0, n_s=NS)


def rss(y, M, gain):
    if gain == "known":
        return float(((M - y) ** 2).sum())
    return float(y @ y - (M @ y) ** 2 / (M @ M))


def grid_search(y, gain, n_step=0.01, B_step=5e3, d_step=1.0):
    """RSS over the whole twin side: n_ref' in (n*, 3.0], B' in [-6e4, 6e4], d' in [150, 1500] nm."""
    n_vals = np.arange(N_STAR + 0.005, 3.0 + 1e-9, n_step)
    B_vals = np.arange(-6e4, 6e4 + 1, B_step)
    d_vals = np.arange(150.0, 1500.0 + 1e-9, d_step)
    yy = y @ y
    best = []
    for nv in n_vals:
        for Bv in B_vals:
            nl = nv + Bv * W2
            if nl.min() <= N_STAR:
                continue
            r01, r12 = (N0 - nl) / (N0 + nl), (nl - NS) / (nl + NS)
            s, p = r01 ** 2 + r12 ** 2, r01 * r12
            c = np.cos(4.0 * np.pi * np.outer(d_vals, nl / LAM))
            M = (s + 2 * p * c) / (1 + p ** 2 + 2 * p * c)
            if gain == "known":
                r = ((M - y) ** 2).sum(axis=1)
            else:
                r = yy - (M @ y) ** 2 / (M * M).sum(axis=1)
            k = int(np.argmin(r))
            best.append((float(r[k]), float(nv), float(Bv), float(d_vals[k])))
    best.sort()
    return best, int(n_vals.size * B_vals.size * d_vals.size)


def refine(y, gain, x0, three_term=False):
    def res(x):
        M = model(x[0], x[1], x[2], x[3] if three_term else 0.0)
        if gain == "known":
            return M - y
        return (M @ y) / (M @ M) * M - y
    lo, hi = [N_STAR, -1e5, 100.0], [3.2, 1e5, 2000.0]
    xs = [0.01, 1e3, 1.0]
    if three_term:
        lo.append(-1e11); hi.append(1e11); xs.append(1e8)
    f = least_squares(res, x0, bounds=(lo, hi), x_scale=xs, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
    return f


def twin_fit(B, gain):
    y = model(N_REF, B, D_TRUE)
    grid, n_cells = grid_search(y, gain)
    starts = [np.array(g[1:]) for g in grid[:40]]
    starts.append(np.array([N0 * NS / N_REF, -B * NS / N_REF ** 2, N_REF ** 2 * D_TRUE / (N0 * NS)]))   # swap at the band centre
    best = None
    for x0 in starts:
        f = refine(y, gain, x0)
        if best is None or f.cost < best.cost:
            best = f
    x = best.x
    M = model(*x)
    R_twin = rss(y, M, gain)
    nl = x[0] + x[1] * W2
    g_hat = float((M @ y) / (M @ M)) if gain == "profiled" else 1.0
    resid = g_hat * M - y
    # three-term Cauchy twin, started from the two-term optimum with several curvatures
    best3 = None
    for C0 in (0.0, 5e8, -5e8, 2e9, -2e9):
        f3 = refine(y, gain, np.array([x[0], x[1], x[2], C0]), three_term=True)
        if best3 is None or f3.cost < best3.cost:
            best3 = f3
    M3 = model(*best3.x)
    R3 = rss(y, M3, gain)
    # the true side: the true film is recovered with zero residual (control)
    ft = least_squares(lambda z: (model(*z) - y) if gain == "known" else ((model(*z) @ y) / (model(*z) @ model(*z)) * model(*z) - y),
                       [N_REF + 0.003, B * 0.9, D_TRUE + 1.0], bounds=([1.0, -1e5, 100.0], [N_STAR, 1e5, 2000.0]),
                       x_scale=[0.01, 1e3, 1.0], xtol=1e-15, ftol=1e-15, gtol=1e-15)
    need = (N_REF + B * W2) ** 2 * D_TRUE / (N0 * NS)          # thickness the exact swap would need at each wavelength
    return {
        "B_nm2": B, "gain": gain, "grid_cells": n_cells, "grid_best_rss": grid[0][0],
        "twin_n_ref": float(x[0]), "twin_B_nm2": float(x[1]), "twin_d_nm": float(x[2]), "twin_gain": g_hat,
        "twin_index_min_over_band": float(nl.min()), "twin_index_max_over_band": float(nl.max()),
        "twin_rss": R_twin, "twin_rms_dR": math.sqrt(R_twin / LAM.size), "twin_max_abs_dR": float(np.abs(resid).max()),
        "chi2_excess": {f"{s:g}": R_twin / s ** 2 for s in SIGMAS},
        "sigma_at_chi2_excess_3p84": math.sqrt(R_twin / 3.84),
        "three_term_twin": {"x": [float(v) for v in best3.x], "rss": R3, "rms_dR": math.sqrt(R3 / LAM.size),
                            "chi2_excess": {f"{s:g}": R3 / s ** 2 for s in SIGMAS}},
        "true_side_refit_rss": float(2 * ft.cost), "true_side_refit_x": [float(v) for v in ft.x],
        "exact_swap_thickness_range_nm": [float(need.min()), float(need.max())],
    }


# T2: what the swap preserves
def r_airy(n, d, ns, lam, theta0_deg=0.0, pol="s", n0=N0):
    """Amplitude reflection coefficient of one layer (Airy form), s or p, at incidence angle theta0 in medium n0."""
    st0 = n0 * math.sin(math.radians(theta0_deg))
    c0, c1, c2 = math.sqrt(1 - (st0 / n0) ** 2), np.sqrt(1 - (st0 / n) ** 2 + 0j), np.sqrt(1 - (st0 / ns) ** 2 + 0j)
    if pol == "s":
        r01 = (n0 * c0 - n * c1) / (n0 * c0 + n * c1); r12 = (n * c1 - ns * c2) / (n * c1 + ns * c2)
    else:
        r01 = (n * c0 - n0 * c1) / (n * c0 + n0 * c1); r12 = (ns * c1 - n * c2) / (ns * c1 + n * c2)
    e = np.exp(-4j * np.pi * n * d * c1 / lam)
    return (r01 + r12 * e) / (1 + r01 * r12 * e)


def r_matrix(n, d, ns, lam, theta0_deg=0.0, pol="s", n0=N0):
    """Independent check: characteristic (Abeles) matrix of one layer, s or p (tilted admittances; the p sign convention
    differs from r_airy, so only |r|^2 is compared between the two)."""
    st0 = n0 * math.sin(math.radians(theta0_deg))
    c0, c1, c2 = math.sqrt(1 - (st0 / n0) ** 2), np.sqrt(1 - (st0 / n) ** 2 + 0j), np.sqrt(1 - (st0 / ns) ** 2 + 0j)
    if pol == "s":
        y0, y1, y2 = n0 * c0, n * c1, ns * c2
    else:
        y0, y1, y2 = n0 / c0, n / c1, ns / c2
    dl = 2 * np.pi * n * d * c1 / lam
    m11, m12, m21, m22 = np.cos(dl), 1j * np.sin(dl) / y1, 1j * y1 * np.sin(dl), np.cos(dl)
    Bv, Cv = m11 + m12 * y2, m21 + m22 * y2
    r = (y0 * Bv - Cv) / (y0 * Bv + Cv)
    return r


def swap_properties():
    n2, d2 = N0 * NS / N_REF, N_REF ** 2 * D_TRUE / (N0 * NS)
    ra, rb = r_airy(N_REF, D_TRUE, NS, LAM), r_airy(n2, d2, NS, LAM)
    dphase = np.angle(ra * np.conj(rb))                            # wrapped phase difference in (-pi, pi]
    z = np.exp(-4j * np.pi * N_REF * D_TRUE / LAM)
    r01, r12 = (N0 - N_REF) / (N0 + N_REF), (N_REF - NS) / (N_REF + NS)
    ident = np.abs(rb - z * np.conj(r01 + r12 * z) / (1 + r01 * r12 * z)).max()   # r' = z conj(r01 + r12 z) / (1 + p z)
    cplx = max(float(np.abs(r_airy(n, d, NS, LAM) - r_matrix(n, d, NS, LAM)).max()) for (n, d) in ((N_REF, D_TRUE), (n2, d2)))
    out = {"film": [N_REF, D_TRUE], "twin": [n2, d2], "ns": NS, "max_abs_complex_r_airy_vs_matrix_normal": cplx,
           "max_abs_diff_R_normal": float(np.abs(np.abs(ra) ** 2 - np.abs(rb) ** 2).max()),
           "max_abs_phase_diff_rad_normal": float(np.abs(dphase).max()),
           "min_abs_phase_diff_rad_normal": float(np.abs(dphase).min()),
           "max_abs_identity_err": float(ident)}
    chk = 0.0
    for th in (0.0, THETA_ELLIPS):
        for pol in ("s", "p"):
            for (n, d) in ((N_REF, D_TRUE), (n2, d2)):
                chk = max(chk, float(np.abs(np.abs(r_airy(n, d, NS, LAM, th, pol)) ** 2 - np.abs(r_matrix(n, d, NS, LAM, th, pol)) ** 2).max()))
    rho = lambda n, d: r_matrix(n, d, NS, LAM, THETA_ELLIPS, "p") / r_matrix(n, d, NS, LAM, THETA_ELLIPS, "s")
    ra_, rb_ = rho(N_REF, D_TRUE), rho(n2, d2)
    psi_a, psi_b = np.degrees(np.arctan(np.abs(ra_))), np.degrees(np.arctan(np.abs(rb_)))
    ddel = np.degrees(np.abs(np.angle(ra_ * np.conj(rb_))))
    Rs_a, Rs_b = np.abs(r_matrix(N_REF, D_TRUE, NS, LAM, THETA_ELLIPS, "s")) ** 2, np.abs(r_matrix(n2, d2, NS, LAM, THETA_ELLIPS, "s")) ** 2
    out.update({"theta_ellipsometry_deg": THETA_ELLIPS, "max_abs_R_airy_vs_matrix": chk,
                "max_abs_diff_psi_deg": float(np.abs(psi_a - psi_b).max()), "max_abs_diff_delta_deg": float(ddel.max()),
                "max_abs_diff_Rs_oblique": float(np.abs(Rs_a - Rs_b).max())})
    return out


def main():
    fits = [twin_fit(B, gain) for B in B_VALUES for gain in ("known", "profiled")]
    out = {"config": {"n0": N0, "n_s": NS, "n_ref": N_REF, "d_nm": D_TRUE, "lam_nm": [float(LAM[0]), float(LAM[-1]), int(LAM.size)],
                      "lam_c_nm": LAM_C, "B_values_nm2": list(B_VALUES), "sigmas": list(SIGMAS), "n_star": N_STAR,
                      "twin_side_search": {"n_ref": [N_STAR + 0.005, 3.0, 0.01], "B_nm2": [-6e4, 6e4, 5e3], "d_nm": [150.0, 1500.0, 1.0]},
                      "chi2_threshold": 3.84},
           "twin_fits": fits, "swap": swap_properties()}
    with open(os.path.join(RESULTS, "twin_dispersion.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for t in fits:
        print(f"B={t['B_nm2']:>7.0f} {t['gain']:>8}: twin (n_ref', B', d') = ({t['twin_n_ref']:.4f}, {t['twin_B_nm2']:.0f}, {t['twin_d_nm']:.2f}) g={t['twin_gain']:.4f} "
              f"rms dR {t['twin_rms_dR']:.3e}, excess {t['chi2_excess']['0.02']:.3f} (0.02) {t['chi2_excess']['0.002']:.1f} (0.002), "
              f"sigma_c {t['sigma_at_chi2_excess_3p84']:.4f}; 3-term {t['three_term_twin']['chi2_excess']['0.002']:.1f}; grid {t['grid_best_rss']:.2e}; true side {t['true_side_refit_rss']:.1e}")
    print(json.dumps(out["swap"], indent=1))


if __name__ == "__main__":
    main()

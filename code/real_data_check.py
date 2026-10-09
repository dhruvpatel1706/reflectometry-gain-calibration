"""
Measured spectra: the estimators of the paper applied to a public record (section "Measured spectra").

Input (data/external_2026-09-29; the file is checked against the SHA-256 in MANIFEST.tsv before use):
  Schwarz, Dick, Beuer, Rommel, Hutzler, Micron 198 (2025) 103878, dataset doi:10.5281/zenodo.15342753 (CC BY 4.0),
  Data_Figure_7.xlsx, sheet 7a: a PECVD silicon nitride film on a 500 um BOROFLOAT 33 wafer, measured with four
  objectives (10x, 50xUL, 100xUL, 100x; NA 0.25, 0.35, 0.60, 0.90), reflectance referenced to a silicon sample. The film
  thickness is 536.4 nm by HAADF-STEM/EDX of a FIB lamella, and the 10x objective collects all orders of backside
  reflection while the others collect only the front reflection (Schwarz, Bauer, Rommel, Hutzler, arXiv:2607.03077,
  CC BY 4.0). The sheet also holds the authors' simulated spectra; a single-layer fit to them identifies the sample.
  Substrate: BOROFLOAT 33, the SCHOTT data sheet reference values (quoted below; the sheet is not redistributed).
  Two other public records were examined and not used; README_MST.md gives the reasons.

Model. The Airy reflectance of the paper for one film on a semi-infinite substrate (front reflection), the film index
a two-term Cauchy form n_c + B (1/lam^2 - 1/lam_c^2) with lam_c = 695 nm, non-absorbing unless stated. Each objective
is modelled by the unpolarized average of s and p reflectance over its aperture, with four weightings of the angle of
incidence: uniform in sin^2(theta) (primary; a uniformly filled pupil), uniform in sin(theta), uniform in theta, and
normal incidence only (the model of the rest of the paper). A fifth variant adds a constant extinction coefficient k.
For the 10x objective the incoherent backside reflection (all orders) is added; fitting it without that term is the
departure test. The backside sum uses the film stack's power reflectance and transmittance in both directions, so it also
holds for the absorbing variant, where T is not 1 - R and the reflectance seen from the substrate differs from the front
one (checked in code/test_real_data_backside.py). Spectra are interpolated linearly to a 1 nm grid from 440 to 950 nm (the instrument range stated by
the data authors, who used the same grid).

Estimators, as in the paper: gain fixed at one (the silicon-referenced scale) and gain profiled (variable projection).
Reported: the fitted parameters, the root-mean-square residual, the standard-deviation-bound ratio kappa of the index
at the fitted film, and the first-order shift of equation (bias) against the refit with the gain fixed.

Writes data/results/real_data_check.json. CPU only, deterministic, no random numbers.
"""
import os, json, hashlib, math
import numpy as np
from scipy.optimize import least_squares
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXT = os.path.join(ROOT, "data", "external_2026-09-29")
OUT = os.path.join(ROOT, "data", "results", "real_data_check.json")

FILES = {"A": "zenodo_15342753/Data_Figure_7.xlsx"}
D_STEM = 536.4
OBJECTIVES = [("10x", 1, 0.25, True), ("50xUL", 2, 0.35, False), ("100xUL", 3, 0.60, False), ("100x", 4, 0.90, False)]
WEIGHTINGS = ("sin2", "sin", "theta", "normal")
K_ANG = 24
# BOROFLOAT 33, SCHOTT "Optical Properties" data sheet (2021-06), reference values: wavelength (nm) and index
BF33_TABLE = [(248.3, 1.525), (296.7, 1.504), (365.0, 1.489), (435.8, 1.480), (480.0, 1.477), (546.1, 1.473),
              (587.6, 1.471), (632.8, 1.470), (643.8, 1.470), (852.1, 1.465), (1014.0, 1.463), (1529.6, 1.456)]


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def check_inputs():
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(EXT, "MANIFEST.tsv"))]
    col = {h: i for i, h in enumerate(rows[0])}
    want = {r[col["file"]]: r[col["sha256"]] for r in rows[1:] if len(r) > 1}
    out = {}
    for f in FILES.values():
        h = sha256(os.path.join(EXT, f))
        assert want.get(f) == h, f"SHA-256 mismatch for {f}"
        out[f] = h
    return out


def bf33_fit():
    t = np.array([r for r in BF33_TABLE if 365.0 <= r[0] <= 1014.0])
    A = np.vstack([np.ones(len(t)), (1000.0 / t[:, 0]) ** 2]).T
    c, *_ = np.linalg.lstsq(A, t[:, 1], rcond=None)
    return c, float(np.abs(A @ c - t[:, 1]).max()), int(len(t))


BF_C, BF_DEV, BF_N = bf33_fit()
n_bf33 = lambda lam: BF_C[0] + BF_C[1] * (1000.0 / lam) ** 2


# ---------------------------------------------------------------- forward model
def angles(NA, mode, K=K_ANG):
    if mode == "normal" or NA == 0:
        return np.array([0.0]), np.array([1.0])
    t = (np.arange(K) + 0.5) / K * math.asin(NA)
    w = {"sin2": np.sin(t) * np.cos(t), "sin": np.cos(t), "theta": np.ones(K)}[mode]
    return t, w / w.sum()


def r_film(lam, n1, d, n2, th):
    """s and p amplitude reflectance of one film (index n1, thickness d) on a semi-infinite substrate n2, from air
    at external angle th; complex indices are n - ik."""
    u = math.sin(th) ** 2
    c0 = math.cos(th); c1 = np.sqrt(1 - u / n1 ** 2 + 0j); c2 = np.sqrt(1 - u / n2 ** 2 + 0j)
    e = np.exp(-4j * np.pi * n1 * c1 * d / lam)
    out = []
    for a0, a1, a2 in ((c0, n1 * c1, n2 * c2), (1 / c0, n1 / c1, n2 / c2)):
        r01, r12 = (a0 - a1) / (a0 + a1), (a1 - a2) / (a1 + a2)
        out.append((r01 + r12 * e) / (1 + r01 * r12 * e))
    return out


def film_power_rt(lam, n1, d, n2, th):
    """Power coefficients (R_front, T_forward, R_reverse, T_reverse) of one film on the substrate, s and p. Air and
    substrate are non-absorbing; the film may absorb (n - ik). Admittances are for tangential fields, so a power
    transmittance carries the ratio of exit to entry admittance. Reverse incidence, from the substrate, keeps the same
    tangential wavevector."""
    u = math.sin(th) ** 2
    c0 = math.cos(th); c1 = np.sqrt(1 - u / n1 ** 2 + 0j); c2 = np.sqrt(1 - u / n2 ** 2 + 0j)
    one = np.exp(-2j * np.pi * n1 * c1 * d / lam)
    two = one * one

    def powers(y0, y1, y2):
        r01, r12 = (y0 - y1) / (y0 + y1), (y1 - y2) / (y1 + y2)
        den = 1 + r01 * r12 * two
        t = (2 * y0 / (y0 + y1)) * (2 * y1 / (y1 + y2)) * one / den
        return np.abs((r01 + r12 * two) / den) ** 2, np.real(y2) / np.real(y0) * np.abs(t) ** 2

    out = []
    for a0, a1, a2 in ((c0, n1 * c1, n2 * c2), (1 / c0, n1 / c1, n2 / c2)):
        out.append(powers(a0, a1, a2) + powers(a2, a1, a0))
    return out


def refl(lam, n1, d, n2, NA, mode, backside=False, K=K_ANG):
    t, w = angles(NA, mode, K)
    R = 0.0
    lossy = backside and bool(np.any(np.imag(n1) != 0))
    for th, wt in zip(t, w):
        pw = film_power_rt(lam, n1, d, n2, th) if lossy else None
        for pol, r in enumerate(r_film(lam, n1, d, n2, th)):
            Rf = np.abs(r) ** 2
            if backside:                                          # incoherent round trips in the 500 um wafer
                u = math.sin(th) ** 2; cg = np.sqrt(1 - u / n2 ** 2); c0 = math.cos(th)
                a_g, a_0 = (n2 * cg, c0) if pol == 0 else (n2 / cg, 1 / c0)
                Rb = np.abs((a_g - a_0) / (a_g + a_0)) ** 2
                if pw is None:                                    # lossless film: T = 1 - Rf, reflectance equal from both sides
                    Rf = Rf + (1 - Rf) ** 2 * Rb / (1 - Rf * Rb)
                else:                                             # absorbing film: Rf + Tf Rb Tr / (1 - Rr Rb)
                    _, Tf, Rr, Tr = pw[pol]
                    Rf = Rf + Tf * Rb * Tr / (1 - Rr * Rb)
            R = R + wt * 0.5 * Rf
    return R


def r_matrix(lam, n1, d, n2, th):
    """Independent check of r_film: the characteristic-matrix (admittance) form for one layer, s and p."""
    u = math.sin(th) ** 2
    c0 = math.cos(th); c1 = np.sqrt(1 - u / n1 ** 2 + 0j); c2 = np.sqrt(1 - u / n2 ** 2 + 0j)
    dl = 2 * np.pi * n1 * c1 * d / lam
    out = []
    for y0, y1, y2 in ((c0, n1 * c1, n2 * c2), (1 / c0, n1 / c1, n2 / c2)):
        Bv = np.cos(dl) + 1j * np.sin(dl) / y1 * y2
        Cv = 1j * y1 * np.sin(dl) + np.cos(dl) * y2
        out.append((y0 * Bv - Cv) / (y0 * Bv + Cv))
    return out


LAM_A = np.arange(440.0, 951.0, 1.0)
LC_A = 0.5 * (LAM_A[0] + LAM_A[-1])
NS_A = n_bf33(LAM_A)


def model_A(p, NA, mode, backside=False, absorbing=False):
    k = p[3] if absorbing else 0.0
    n = p[0] + p[1] * (1 / LAM_A ** 2 - 1 / LC_A ** 2) - 1j * k
    return refl(LAM_A, n, p[2], NS_A, NA, mode, backside)


def vp(m, y):
    return float((m @ y) / (m @ m))


def rss(m, y, prof):
    return float(y @ y - (m @ y) ** 2 / (m @ m)) if prof else float(((m - y) ** 2).sum())


def grid_starts_A(y, prof, n_keep=8):
    """Coarse search at normal incidence over the whole plausible region; distinct cells become starting points."""
    cells = []
    for nc in np.arange(1.60, 2.4001, 0.02):
        for B in (0.0, 1e4, 2e4, 4e4):
            best = None
            for d in np.arange(300.0, 900.1, 2.0):
                r = rss(model_A([nc, B, d], 0.0, "normal"), y, prof)
                if best is None or r < best[0]:
                    best = (r, d)
            cells.append((best[0], float(nc), float(B), best[1]))
    cells.sort()
    keep = []
    for c in cells:
        if all(abs(c[3] - k[3]) > 20.0 or abs(c[1] - k[1]) > 0.05 for k in keep):
            keep.append(c)
        if len(keep) == n_keep:
            break
    return [[c[1], c[2], c[3]] for c in keep]


def fit_A(y, prof, NA, mode, backside=False, absorbing=False, starts=None):
    def res(p):
        m = model_A(p, NA, mode, backside, absorbing)
        return vp(m, y) * m - y if prof else m - y
    lo, hi, xs = [1.3, -2e5, 100.0], [3.0, 2e5, 1500.0], [0.01, 1e3, 1.0]
    if absorbing:
        lo.append(0.0); hi.append(0.2); xs.append(1e-3)
    sols = []
    for s in starts:
        for k0 in ((0.0, 0.005, 0.01) if absorbing else (None,)):
            x0 = list(s) + ([k0] if absorbing else [])
            f = least_squares(res, x0, bounds=(lo, hi), x_scale=xs, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=5000)
            sols.append(f)
    sols.sort(key=lambda f: f.cost)
    b = sols[0]
    alt = [f for f in sols if abs(f.x[2] - b.x[2]) > 50.0]
    m = model_A(b.x, NA, mode, backside, absorbing)
    g = vp(m, y) if prof else 1.0
    out = {"n_c": float(b.x[0]), "B_nm2": float(b.x[1]), "d_nm": float(b.x[2]), "tau_c_nm": float(b.x[0] * b.x[2]),
           "gain": g, "rms": math.sqrt(2 * b.cost / y.size), "starts": len(sols),
           "alt_order_rms": math.sqrt(2 * alt[0].cost / y.size) if alt else None,
           "alt_order_d_nm": float(alt[0].x[2]) if alt else None}
    if absorbing:
        out["k"] = float(b.x[3])
    return out, np.array(b.x)


def jac_A(x, NA, mode, backside=False):
    steps = (1e-6, 1e-2, 1e-4)
    cols = []
    for i, h in enumerate(steps):
        a, b = np.array(x, float), np.array(x, float)
        a[i] += h; b[i] -= h
        cols.append((model_A(a, NA, mode, backside) - model_A(b, NA, mode, backside)) / (2 * h))
    return np.array(cols).T


def kappa_from(J, m):
    """Ratio of the Cramer-Rao standard-deviation bounds of the first parameter with the gain unknown and known."""
    Fk = J.T @ J
    Fu = np.block([[Fk, (J.T @ m)[:, None]], [(m @ J)[None, :], np.array([[m @ m]])]])
    return float(math.sqrt(np.linalg.inv(Fu)[0, 0] / np.linalg.inv(Fk)[0, 0]))


def record_A():
    wb = openpyxl.load_workbook(os.path.join(EXT, FILES["A"]), read_only=True, data_only=True)
    res = {}
    sheets = {}
    for sh in ("7a", "7b"):
        rows = [list(r) for r in wb[sh].iter_rows(values_only=True)]
        meas = np.array([r[0:5] for r in rows[1:] if r[0] is not None], float)
        sim = np.array([r[6:11] for r in rows[1:] if r[6] is not None], float)
        sheets[sh] = (rows[0], meas, sim)
    hdr, meas, sim = sheets["7a"]
    assert [h for h in hdr[1:5]] == ["Reflectance, Meas 10x", "Reflectance, Meas 50xUL", "Reflectance, Meas 100xUL", "Reflectance, Meas 100x"]
    assert np.allclose(sim[:, 0], LAM_A) and meas[0, 0] < LAM_A[0] and meas[-1, 0] > LAM_A[-1]
    band = (meas[:, 0] >= LAM_A[0]) & (meas[:, 0] <= LAM_A[-1])
    res["data"] = {"sheet": "7a", "measured_points": int(meas.shape[0]), "measured_range_nm": [float(meas[0, 0]), float(meas[-1, 0])],
                   "points_in_band": int(band.sum()), "grid_nm": [float(LAM_A[0]), float(LAM_A[-1]), int(LAM_A.size)], "lam_c_nm": float(LC_A),
                   "second_difference_sd": {h.split()[-1]: float(np.std(np.diff(meas[band, j], 2)) / math.sqrt(6)) for j, h in enumerate(hdr[1:5], 1)}}
    # identification: the authors' simulated spectra of panel 7a are a single film of about 536 nm; those of 7b are not
    ident = {}
    for sh in ("7a", "7b"):
        _, _, sm = sheets[sh]
        for name, j, NA, _b in OBJECTIVES[1:3]:
            y = sm[:, j]
            r, _ = fit_A(y, False, NA, "sin2", starts=grid_starts_A(y, False))
            ident[f"{sh}_{name}_authors_sim"] = r
    res["identification"] = ident
    fits = {}
    for name, j, NA, back in OBJECTIVES:
        y = np.interp(LAM_A, meas[:, 0], meas[:, j])
        fits[name] = {"NA": NA, "backside_modelled": back}
        xs = {}
        for prof in (False, True):
            st = grid_starts_A(y, prof)
            key = "profiled" if prof else "fixed"
            for mode in WEIGHTINGS:
                r, x = fit_A(y, prof, NA, mode, backside=back, starts=st)
                fits[name].setdefault(mode, {})[key] = r
                if mode == "sin2":
                    xs[key] = x
            r, _ = fit_A(y, prof, NA, "sin2", backside=back, absorbing=True, starts=st)
            fits[name].setdefault("sin2_absorbing", {})[key] = r
            if back:                                              # departure test: the same spectrum without the backside term
                r, _ = fit_A(y, prof, NA, "sin2", backside=False, starts=st)
                fits[name].setdefault("sin2_front_only", {})[key] = r
        # kappa and the first-order shift at the profiled optimum of the primary model
        xp, xf = xs["profiled"], xs["fixed"]
        m = model_A(xp, NA, "sin2", back)
        J = jac_A(xp, NA, "sin2", back)
        g = fits[name]["sin2"]["profiled"]["gain"]
        S = np.linalg.solve(J.T @ J, J.T @ m)                     # d(theta) per unit gain error with the gain fixed at one
        fits[name]["kappa"] = kappa_from(J, m)
        fits[name]["first_order"] = {"slope_n_per_pct": float(0.01 * S[0]), "slope_d_nm_per_pct": float(0.01 * S[2]),
                                     "pred_shift_n": float((g - 1) * S[0]), "obs_shift_n": float(xf[0] - xp[0]),
                                     "pred_shift_d_nm": float((g - 1) * S[2]), "obs_shift_d_nm": float(xf[2] - xp[2])}
        m2 = refl(LAM_A, xp[0] + xp[1] * (1 / LAM_A ** 2 - 1 / LC_A ** 2), xp[2], NS_A, NA, "sin2", back, K=2 * K_ANG)
        fits[name]["angle_convergence_max_dR"] = float(np.abs(m2 - m).max())
        if name == "50xUL":
            res["curves_50xUL"] = {"lam_nm": LAM_A.tolist(), "measured": y.tolist(),
                                   "fit_gain_fixed": model_A(xf, NA, "sin2", back).tolist(), "fit_gain_profiled": (g * m).tolist()}
    res["fits"] = fits
    # the model's cost for a non-dispersive film of this thickness on this substrate and band (normal incidence)
    pred = {}
    for n in (1.8, 1.9, 2.0):
        x = np.array([n, 0.0, D_STEM])
        pred[f"{n:.1f}"] = kappa_from(jac_A(x, 0.0, "normal")[:, [0, 2]], model_A(x, 0.0, "normal"))
    res["kappa_nondispersive_normal"] = pred
    # the aperture model against the characteristic-matrix form, every angle and both polarizations, clear and absorbing film
    dev = 0.0
    for k in (0.0, 0.01):
        n = 1.9 + 1.2e4 * (1 / LAM_A ** 2 - 1 / LC_A ** 2) - 1j * k
        for th in angles(0.90, "theta")[0]:
            for ra, rm in zip(r_film(LAM_A, n, D_STEM, NS_A, th), r_matrix(LAM_A, n, D_STEM, NS_A, th)):
                dev = max(dev, float(np.abs(np.abs(ra) ** 2 - np.abs(rm) ** 2).max()))
    res["matrix_check_max_dR"] = dev
    return res


def main():
    hashes = check_inputs()
    res = {"config": {"inputs_sha256": hashes, "d_stem_nm": D_STEM, "weightings": list(WEIGHTINGS), "angles_per_weighting": K_ANG,
                      "objectives": [{"name": o[0], "NA": o[2], "backside": o[3]} for o in OBJECTIVES],
                      "bf33": {"coef_a_b": [float(c) for c in BF_C], "form": "a + b (1000/lam)^2", "table_points_used": BF_N,
                               "max_abs_dev_from_table": BF_DEV, "n_440": float(n_bf33(440.0)), "n_950": float(n_bf33(950.0))}}}
    res["A_nitride_on_bf33"] = record_A()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    A = res["A_nitride_on_bf33"]["fits"]
    for name, j, NA, back in OBJECTIVES:
        f, p = A[name]["sin2"]["fixed"], A[name]["sin2"]["profiled"]
        print(f"{name:7s} NA {NA:.2f}: fixed n_c {f['n_c']:.4f} d {f['d_nm']:.1f} | profiled g {p['gain']:.4f} n_c {p['n_c']:.4f} d {p['d_nm']:.1f} "
              f"| tau {f['tau_c_nm']:.1f}/{p['tau_c_nm']:.1f} | kappa {A[name]['kappa']:.2f}")
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()

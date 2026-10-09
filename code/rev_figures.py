"""
Figures for the Measurement Science and Technology manuscript. Reads only data/results/rev_*.json,
data/results/real_data_check.json and the forward model; writes vector PDFs to paper/mst/figures/. Deterministic (no Monte
Carlo).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import optics as O

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "data", "results")
OUT = os.path.join(HERE, "..", "paper", "mst", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.1})
LAM = O.default_grid()
J = lambda name: json.load(open(os.path.join(RES, name)))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def figure1():
    """Measurement chain of a reference-based reflectometer and where each error enters."""
    fig, ax = plt.subplots(figsize=(5.9, 2.35))
    ax.set_xlim(0, 100); ax.set_ylim(0, 43.5); ax.axis("off")
    def box(x, y, w, h, text, fc="#f2f2f2"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="0.25", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2)
    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="0.2"))
    box(1, 29, 15, 10, "lamp\nspectrum $L(\\lambda)$")
    box(22, 29, 17, 10, "optics, fibre\nthroughput $T(\\lambda)$")
    box(45, 35.5, 22, 7, "sample: film on\nsubstrate, $R(\\lambda)$", fc="#e3edf7")
    box(45, 25.5, 22, 7, "reference, tabulated\n$R_{\\mathrm{ref}}(\\lambda)$", fc="#f7ede3")
    box(73, 29, 25, 10, "spectrometer:\ncounts $I$, dark $D$")
    arrow(16.8, 34, 21.2, 34); arrow(39.8, 34, 44.2, 38.5); arrow(39.8, 34, 44.2, 29)
    arrow(67.8, 39, 72.2, 35.5); arrow(67.8, 29, 72.2, 32.5)
    ax.text(50, 19.5, r"$\widehat R(\lambda)=R_{\mathrm{ref}}(\lambda)\,\dfrac{I_{\mathrm{s}}(\lambda)-D_{\mathrm{s}}(\lambda)}{I_{\mathrm{r}}(\lambda)-D_{\mathrm{r}}(\lambda)}$",
            ha="center", va="center", fontsize=9)
    rows = [("lamp or throughput drift between the two measurements, uniform in $\\lambda$", "scalar gain $g$ (this paper)"),
            ("error in the tabulated $R_{\\mathrm{ref}}$, or drift that varies with $\\lambda$", "spectral gain $g(\\lambda)$"),
            ("stray light, incomplete dark subtraction", "additive offset"),
            ("detector response shared by both measurements (linear, stable)", "cancels in the ratio")]
    for k, (cause, effect) in enumerate(rows):
        y = 12.2 - 3.6 * k
        ax.text(3, y, cause, ha="left", va="center", fontsize=6.8)
        ax.text(97, y, effect, ha="right", va="center", fontsize=6.8, style="italic")
    ax.plot([2, 98], [14.6, 14.6], color="0.6", lw=0.5)
    save(fig, "figure1.pdf")


def figure2():
    S = J("rev_structural.json")
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.2))
    a = axs[0]
    n1, d1, ns = 1.40, 700.0, 2.30
    nm, dm = S["symmetry"]["example_ns_2p30"]["n_mirror"], S["symmetry"]["example_ns_2p30"]["d_mirror_nm"]
    a.plot(LAM, O.reflectance(LAM, n1, d1, n_s=ns), color="C0", label=f"$n$={n1:.2f}, $d$={d1:.0f} nm")
    a.plot(LAM, O.reflectance(LAM, nm, dm, n_s=ns), color="C3", ls=(0, (3, 2)), label=f"$n$={nm:.4f}, $d$={dm:.1f} nm")
    a.plot(LAM, 1.10 * O.reflectance(LAM, n1, d1, n_s=ns), color="0.45", lw=0.8, ls=":", label="first film, $g$ = 1.10")
    a.set_xlabel("wavelength (nm)"); a.set_ylabel("reflectance"); a.set_title("(a) mirror twins, $n_s$ = 2.30")
    a.set_ylim(0, 0.265)
    a.legend(frameon=False, loc="upper left", fontsize=5.8, handlelength=2.2)
    for ax, key, title in ((axs[1], "main_regime_ns1p5", "(b) $n$ = 1.80, $n_s$ = 1.5"), (axs[2], "reviewer_example_ns2p30", "(c) $n$ = 1.40, $n_s$ = 2.30")):
        c = S["index_profiles"][key]["curve"]
        n = np.array(c["n"])
        ax.semilogy(n, np.maximum(np.array(c["known_over_sigma2"]), 1e-3), color="C0", label="gain known")
        ax.semilogy(n, np.maximum(np.array(c["unknown_over_sigma2"]), 1e-3), color="C1", ls=(0, (4, 1.5)), label="gain profiled")
        ax.axhline(3.84, color="0.4", lw=0.6, ls=":")
        ax.set_ylim(1e-3, 3e4); ax.set_xlim(1.2, 2.3)
        ax.set_xlabel("candidate index $n$"); ax.set_title(title)
    axs[1].set_ylabel(r"min RSS / $\sigma^2$  ($\sigma$ = 0.02)")
    axs[1].legend(frameon=False, loc="upper right", fontsize=6)
    fig.tight_layout(w_pad=0.8)
    save(fig, "figure2.pdf")


def figure3():
    F = J("rev_fisher.json")
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.3), sharey=True)
    for ax, c, tag in zip(axs, F["curves"], ("(a)", "(b)")):
        n = np.array(c["n"]); ns = c["n_s"]; nstar = np.sqrt(ns)
        gap = np.diff(n) > 0.005
        split = np.where(gap)[0] + 1
        for seg in np.split(np.arange(n.size), split):
            ax.semilogy(n[seg], np.array(c["crb_sd_n_known"])[seg], color="C0", label="gain known" if seg[0] == 0 else None)
            ax.semilogy(n[seg], np.array(c["crb_sd_n_unknown"])[seg], color="C1", ls=(0, (4, 1.5)), label="gain unknown" if seg[0] == 0 else None)
        ax.axvspan(1.35, 2.15, color="0.93", zorder=0)
        ax.axvline(nstar, color="C2", lw=0.8, ls="-."); ax.axvline(ns, color="0.3", lw=0.8, ls=":")
        ax.text(nstar, 3.2, r" $\sqrt{n_0 n_s}$", color="C2", fontsize=6.5, va="top")
        ax.text(ns, 3.2, r" $n_s$", color="0.3", fontsize=6.5, va="top")
        ax.set_ylim(2e-3, 5); ax.set_xlim(1.1, 2.4)
        ax.set_xlabel("film index $n$"); ax.set_title(f"{tag} $n_s$ = {ns:.2f}, $d$ = {c['d_nm']:.0f} nm")
    axs[0].set_ylabel(r"Cramér–Rao sd of $n$ ($\sigma$ = 0.02)")
    axs[0].legend(frameon=False, loc="upper right", fontsize=6.5)
    fig.tight_layout(w_pad=0.8)
    save(fig, "figure3.pdf")


def figure_sweep():
    W = J("rev_sweep.json")["axes"]
    Fh = J("rev_fisher.json")["axes"]
    fig, axs = plt.subplots(1, 4, figsize=(6.6, 2.1))
    def errbars(ax, x, rows, key, **kw):
        y = np.array([r[key]["ratio"] for r in rows]); lo = np.array([r[key]["ci95"][0] for r in rows]); hi = np.array([r[key]["ci95"][1] for r in rows])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], capsize=2, ms=3.5, lw=0.9, **kw)
    fr = W["fringe_count"]; x = [r["fringes_mid"] for r in fr]
    errbars(axs[0], x, fr, "rho_sep_full", marker="o", color="C1", label="Monte Carlo")
    axs[0].plot(x, [r["sd_ratio_median"] for r in Fh["fringe_count"]], "s:", color="0.35", ms=3, lw=0.8, label="local bound (median)")
    axs[0].set_xlabel("fringes across band"); axs[0].set_title("(a) thickness band")
    nz = W["noise"]; x = [r["sigma"] for r in nz]
    errbars(axs[1], x, nz, "rho_sep_full", marker="o", color="C1")
    axs[1].axhline(Fh["substrate_index"][1]["sd_ratio_median"], color="0.35", ls=":", lw=0.8)
    axs[1].set_xscale("log"); axs[1].set_xticks([0.005, 0.01, 0.02, 0.04, 0.08]); axs[1].set_xticklabels(["0.005", "0.01", "0.02", "0.04", "0.08"], rotation=40)
    axs[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[1].set_xlabel(r"noise $\sigma$"); axs[1].set_title("(b) noise level")
    sb = W["substrate_index"]; x = np.array([r["n_s"] for r in sb])
    errbars(axs[2], x - 0.012, sb, "rho_sep_full", marker="o", color="C1", label="full support")
    errbars(axs[2], x + 0.012, sb, "rho_sep_branch", marker="^", color="C4", mfc="white", label="true branch only")
    axs[2].plot(x, [r["sd_ratio_median"] for r in Fh["substrate_index"]], "s:", color="0.35", ms=3, lw=0.8)
    axs[2].set_xlim(1.28, 2.37); axs[2].set_xlabel("substrate index $n_s$"); axs[2].set_title("(c) substrate")
    for a in axs[:3]:
        a.set_ylim(0, 13)
    axs[0].set_ylabel("index RMSE ratio, profiled / known")
    axs[0].legend(frameon=False, fontsize=5.8, loc="lower left"); axs[2].legend(frameon=False, fontsize=5.8, loc="upper right")
    a = axs[3]
    a.plot(x, [r["rel_n_known_full"] for r in sb], "o-", color="C0", ms=3.5, label="full support")
    a.plot(x, [r["rel_n_known_branch"] for r in sb], "^--", color="C4", mfc="white", ms=3.5, label="true branch only")
    a2 = a.twinx()
    a2.bar(x, [r["frac_mirror_in_grid"] for r in sb], width=0.08, color="0.85", zorder=0)
    a2.set_ylim(0, 1); a2.set_ylabel("share of films with twin in support", fontsize=6.5)
    a.set_zorder(a2.get_zorder() + 1); a.patch.set_visible(False)
    a.set_xlabel("substrate index $n_s$"); a.set_ylabel("calibrated relative index RMSE"); a.set_title("(d) calibrated error")
    a.legend(frameon=False, fontsize=5.8, loc="upper left")
    fig.tight_layout(w_pad=0.5)
    save(fig, "figure5.pdf")


def figure_bias():
    S = J("rev_structural.json")["gain_error_bias"]["by_gain_error"]
    F = J("rev_fisher.json")["gain_error"]
    keys = sorted(S, key=float)
    dlt = np.array([float(k) for k in keys])
    med = np.array([S[k]["median_abs_bias_n"] for k in keys])
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    ax.plot(np.abs(dlt[dlt < 0]) * 100, med[dlt < 0], "o", color="C0", mfc="white", ms=7, mew=0.9, label="nonlinear fits, $g<1$")
    ax.plot(dlt[dlt > 0] * 100, med[dlt > 0], "s", color="C3", ms=3, label="nonlinear fits, $g>1$")
    xx = np.linspace(0, 21, 50)
    ax.plot(xx, F["linear_median_abs_bias_n_per_1pct"] * xx, "-", color="0.3", lw=0.9, label="first order")
    ax.set_xlabel("gain error $|g-1|$ (%)"); ax.set_ylabel(r"median $|$index bias$|$")
    ax.legend(frameon=False, fontsize=6.3, loc="upper left")
    fig.tight_layout()
    save(fig, "figure4.pdf")


def figure_measured():
    """Measured spectrum of the silicon nitride film (objective of NA 0.35) with the fits for the two gain treatments."""
    R = J("real_data_check.json")["A_nitride_on_bf33"]
    c, f = R["curves_50xUL"], R["fits"]["50xUL"]["sin2"]
    lam, y = np.array(c["lam_nm"]), np.array(c["measured"])
    mf, mp = np.array(c["fit_gain_fixed"]), np.array(c["fit_gain_profiled"])
    a, b = f["fixed"], f["profiled"]
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.3))
    ax = axs[0]
    ax.plot(lam, y, color="0.55", lw=2.2, label="measured")
    ax.plot(lam, mf, color="C0", lw=0.9, label=f"gain fixed at 1: $n_c$ = {a['n_c']:.3f}, $d$ = {a['d_nm']:.1f} nm")
    ax.plot(lam, mp, color="C3", lw=0.9, ls=(0, (4, 1.5)), label=f"gain profiled: $\\hat g$ = {b['gain']:.3f}, $n_c$ = {b['n_c']:.3f}, $d$ = {b['d_nm']:.1f} nm")
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("reflectance"); ax.set_title("(a) silicon nitride on BOROFLOAT 33, NA 0.35")
    ax.set_ylim(0, 0.26); ax.set_xlim(lam[0], lam[-1])
    ax.legend(frameon=False, loc="upper left", fontsize=5.6, handlelength=2.2)
    ax = axs[1]
    ax.axhline(0, color="0.6", lw=0.5)
    ax.plot(lam, mf - y, color="C0", lw=0.9, label="gain fixed at 1")
    ax.plot(lam, mp - y, color="C3", lw=0.9, ls=(0, (4, 1.5)), label="gain profiled")
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("fit minus measured"); ax.set_title("(b) residuals")
    ax.set_xlim(lam[0], lam[-1])
    ax.legend(frameon=False, loc="lower right", fontsize=6)
    fig.tight_layout(w_pad=0.8)
    save(fig, "figure6.pdf")


if __name__ == "__main__":
    figure1(); figure2(); figure3(); figure_bias(); figure_sweep(); figure_measured()
    print(sorted(os.listdir(OUT)))

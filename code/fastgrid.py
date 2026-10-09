"""
Grid inference for the thin-film problem (September 2026).

Every likelihood used here depends on the data only through Syy = sum_l y_l^2 and, for each grid cell,
SyM = sum_l y_l M_l, because the residual sum of squares at gain g is
    RSS(g) = Syy - 2 g SyM + g^2 SMM,      SMM = sum_l M_l^2 (data-independent).
SyM for all cells is one matrix-vector product with the precomputed model matrix, so a posterior over half a million
cells takes a few tens of milliseconds. Parameter summaries use sort orders computed once per grid.

Gain treatments (sigma is the noise standard deviation in reflectance units at unit gain):
  ("fixed", g0)                    gain fixed at g0; noise sd sigma (or g0*sigma with scaled_noise=True).
  ("profile",)                     gain maximized cell by cell, g_hat = SyM / SMM, unconstrained in sign; normalized
                                   profile-likelihood weights, used for point summaries, never called a posterior.
  ("marginal", lo, hi)             gain integrated under g ~ Uniform(lo, hi) in closed form (Gaussian CDF), noise sd sigma.
  ("marginal", lo, hi) with scaled_noise=True
                                   gain integrated under the same prior when the noise sd is g*sigma (a reference-scale error
                                   scales signal and noise together); numerical quadrature on 401 gain nodes.
The closed form for the fixed-noise marginal is
    p(y | theta) = (hi - lo)^-1 (2 pi sigma^2)^(-L/2) exp(-RSS(g_hat) / (2 sigma^2)) * sigma sqrt(2 pi / SMM)
                   * [Phi((hi - g_hat) sqrt(SMM) / sigma) - Phi((lo - g_hat) sqrt(SMM) / sigma)],
so, up to terms common to all cells, log p = -RSS(g_hat)/(2 sigma^2) - log(SMM)/2 + log[Phi(z_hi) - Phi(z_lo)].
"""
import numpy as np
from scipy.special import log_ndtr, ndtr, logsumexp


def log_normal_interval(z_lo, z_hi):
    """Stable log(Phi(z_hi) - Phi(z_lo)) for arrays with z_lo <= z_hi."""
    z_lo, z_hi = np.broadcast_arrays(np.asarray(z_lo, float), np.asarray(z_hi, float))
    out = np.empty(z_lo.shape)
    left, right = z_hi <= 0, z_lo >= 0
    mid = ~(left | right)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        a, b = log_ndtr(z_hi[left]), log_ndtr(z_lo[left])
        out[left] = a + np.log1p(-np.exp(np.minimum(0.0, b - a)))
        a, b = log_ndtr(-z_lo[right]), log_ndtr(-z_hi[right])
        out[right] = a + np.log1p(-np.exp(np.minimum(0.0, b - a)))
        out[mid] = np.log(ndtr(z_hi[mid]) - ndtr(z_lo[mid]))
    return out


class Grid:
    """A flattened parameter grid with its model spectra.

    M      : (cells, L) model reflectance at unit gain (float64)
    params : dict name -> (cells,) parameter value per cell (e.g. n, d, tau)
    """

    def __init__(self, M, params):
        self.M = np.ascontiguousarray(M, dtype=float)
        self.L = self.M.shape[1]
        self.SMM = np.einsum("ij,ij->i", self.M, self.M)
        self.params = {k: np.asarray(v, float) for k, v in params.items()}
        self.order = {k: np.argsort(v, kind="stable") for k, v in self.params.items()}
        self.sorted_vals = {k: v[self.order[k]] for k, v in self.params.items()}

    def loglik(self, y, sigma, gain, scaled_noise=False, mask=None):
        y = np.asarray(y, float)
        Syy = float(y @ y)
        SyM = self.project(y)
        kind = gain[0]
        if kind == "fixed":
            g0 = float(gain[1])
            s = g0 * sigma if scaled_noise else sigma
            ll = -(Syy - 2.0 * g0 * SyM + g0 * g0 * self.SMM) / (2.0 * s * s)
        elif kind == "profile":
            ll = -np.maximum(0.0, Syy - SyM * SyM / self.SMM) / (2.0 * sigma * sigma)
        elif kind == "marginal":
            lo, hi = float(gain[1]), float(gain[2])
            if not scaled_noise:
                g_hat = SyM / self.SMM
                rss = np.maximum(0.0, Syy - SyM * g_hat)
                root = np.sqrt(self.SMM) / sigma
                ll = -rss / (2.0 * sigma * sigma) - 0.5 * np.log(self.SMM) + log_normal_interval((lo - g_hat) * root, (hi - g_hat) * root)
            else:
                # Screen with the integrand at the clipped profile gain, then integrate exactly (trapezoid, 401 nodes)
                # on the cells within 60 log-units of the best; the others carry weight below exp(-60) each.
                gc = np.clip(SyM / self.SMM, lo, hi)
                approx = -self.L * np.log(gc) - (Syy - 2.0 * gc * SyM + gc * gc * self.SMM) / (2.0 * (gc * sigma) ** 2)
                keep = np.flatnonzero(approx > approx.max() - 60.0)
                gs = np.linspace(lo, hi, 401)
                w = np.full(gs.size, gs[1] - gs[0]); w[0] *= 0.5; w[-1] *= 0.5
                SyMk, SMMk = SyM[keep][:, None], self.SMM[keep][:, None]
                t = -self.L * np.log(gs)[None, :] - (Syy - 2.0 * gs[None, :] * SyMk + gs[None, :] ** 2 * SMMk) / (2.0 * (gs[None, :] * sigma) ** 2) + np.log(w)[None, :]
                ll = np.full(self.SMM.shape, -np.inf)
                ll[keep] = logsumexp(t, axis=1)
        else:
            raise ValueError(gain)
        if mask is not None:
            ll = np.where(mask, ll, -np.inf)
        return ll, SyM

    def project(self, y):
        """SyM for every cell (overridden by grids that store the model matrix differently)."""
        return self.M @ y

    def weights(self, ll):
        w = np.exp(ll - np.max(ll))
        return w / w.sum()

    def summarize(self, w, levels=(0.68, 0.90), names=None):
        out = {}
        imax = int(np.argmax(w))
        for k in (names or self.params):
            v = self.params[k]
            ws = w[self.order[k]]
            c = np.cumsum(ws)
            sv = self.sorted_vals[k]
            mean = float(w @ v)
            out[k + "_mean"] = mean
            out[k + "_median"] = float(sv[min(np.searchsorted(c, 0.5), sv.size - 1)])
            out[k + "_map"] = float(v[imax])
            out[k + "_sd"] = float(np.sqrt(max(0.0, w @ (v - mean) ** 2)))
            for q in levels:
                lo = sv[min(np.searchsorted(c, (1 - q) / 2), sv.size - 1)]
                hi = sv[min(np.searchsorted(c, 1 - (1 - q) / 2), sv.size - 1)]
                out[f"{k}_ci{int(round(q * 100))}"] = (float(lo), float(hi))
        return out


def airy_matrix(lam, n_grid, d_grid, n0=1.0, n_s=1.5):
    """Model matrix of the non-dispersive single-film Airy reflectance over an (n, d) grid, flattened n-major."""
    n_grid, d_grid, lam = (np.asarray(v, float) for v in (n_grid, d_grid, lam))
    M = np.empty((n_grid.size, d_grid.size, lam.size))
    d = d_grid[None, :, None]; L = lam[None, None, :]
    for i in range(0, n_grid.size, 32):                     # in chunks of 32 indices to bound memory
        nf = n_grid[i:i + 32, None, None]
        r01 = (n0 - nf) / (n0 + nf)
        r12 = (nf - n_s) / (nf + n_s)
        e = np.exp(-4j * np.pi * nf * d / L)
        r = (r01 + r12 * e) / (1.0 + r01 * r12 * e)
        M[i:i + 32] = np.abs(r) ** 2
    M = M.reshape(n_grid.size * d_grid.size, lam.size)
    NN, DD = np.meshgrid(n_grid, d_grid, indexing="ij")
    return M, {"n": NN.ravel(), "d": DD.ravel(), "tau": (NN * DD).ravel()}

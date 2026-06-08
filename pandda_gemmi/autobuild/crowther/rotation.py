"""SH-Crowther fast rotation function — gemmi-free core.

Lifted verbatim from FragVol ``inspect_mr_sh.py`` (2026-06-08). Every function
here operates on a plain ``(n, n, n)`` float32 array plus a Cartesian
``origin`` (3-vec, A) plus a *scalar isotropic* ``spacing`` (A). There is no
``gemmi.UnitCell`` anywhere in this math.

INVARIANT (silent-wrong if violated): isotropic scalar spacing, Cartesian
origin, C-order ``[i, j, k]`` indexing, rotations as bare 3x3 matmuls. The
adapter in ``fit.py`` guarantees this by cutting the cube with a transform whose
matrix is ``spacing * I`` (PanDDA2 ``SampleFrame``). Do not hand these functions
a raw native-cell grid.

Algorithm (Crowther 1972 in SH coefficients):
    C(R) = Sum_l Sum_m Sum_m' D^l_{mm'}(R) . X_l[m, m']
with the cross-correlation tensor
    X_l[m, m'] = Sum_r conj(a_lm(r)) . b_lm'(r) . r^2 . dr
precomputed once per (target, probe) pair. Per-rotation cost is O(L^3),
independent of grid size.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import roots_legendre

# scipy >= 1.15 renamed sph_harm -> sph_harm_y AND changed the argument order
# from (m, l, phi, theta) to (l, m, theta, phi). Shim so the verbatim
# sph_harm_y(l, m, theta, phi) calls below work on either.
try:  # scipy >= 1.15
    from scipy.special import sph_harm_y
except ImportError:  # scipy < 1.15
    from scipy.special import sph_harm as _sph_harm

    def sph_harm_y(l, m, theta, phi):  # noqa: E741 - match upstream signature
        return _sph_harm(m, l, phi, theta)


# Precision policy (FragVol v0.18): fp32 grids; complex64 SH/Wigner/X tensors;
# Wigner small-d keeps fp64 internally then casts. Rank-stable to recall@10.
DTYPE_GRID = np.float32
DTYPE_SH = np.complex64
DTYPE_REAL = np.float32


def compute_patterson(density: np.ndarray) -> np.ndarray:
    """Patterson (autocorrelation) of a real voxel density: P = IFFT(|FFT(rho)|^2).

    Returned grid is centrosymmetric, centred at the grid midpoint (fftshift);
    P(0) sits at (N//2, N//2, N//2) and equals Sum|rho|^2. Translation-invariant
    -> this is the input the Crowther rotation function expects. Requires a
    wrap-around-clean box (extent >= ~2x the density's largest internal vector);
    see fit.py for the clean-box headroom and target masking.
    """
    F = np.fft.rfftn(density.astype(np.float32))
    intensity = (F * F.conj()).real
    P = np.fft.irfftn(intensity, s=density.shape)
    return np.fft.fftshift(P).astype(np.float32)


def trilinear_sample(grid: np.ndarray, origin: np.ndarray, spacing: float,
                     points: np.ndarray) -> np.ndarray:
    """Trilinear interpolation of a 3D grid at arbitrary Cartesian points.
    ``points`` is (N, 3). Returns (N,) values; out-of-grid points are 0."""
    n = grid.shape[0]
    rel = (points - origin[None, :]) / spacing
    i0 = np.floor(rel).astype(np.int32)
    f = rel - i0
    in_bounds = ((i0[:, 0] >= 0) & (i0[:, 0] < n - 1) &
                 (i0[:, 1] >= 0) & (i0[:, 1] < n - 1) &
                 (i0[:, 2] >= 0) & (i0[:, 2] < n - 1))
    out = np.zeros(points.shape[0], dtype=np.float64)
    if not in_bounds.any():
        return out
    idx = np.where(in_bounds)[0]
    i = i0[idx]
    g = f[idx]
    fx, fy, fz = g[:, 0], g[:, 1], g[:, 2]
    c000 = grid[i[:, 0],     i[:, 1],     i[:, 2]]
    c100 = grid[i[:, 0] + 1, i[:, 1],     i[:, 2]]
    c010 = grid[i[:, 0],     i[:, 1] + 1, i[:, 2]]
    c001 = grid[i[:, 0],     i[:, 1],     i[:, 2] + 1]
    c110 = grid[i[:, 0] + 1, i[:, 1] + 1, i[:, 2]]
    c101 = grid[i[:, 0] + 1, i[:, 1],     i[:, 2] + 1]
    c011 = grid[i[:, 0],     i[:, 1] + 1, i[:, 2] + 1]
    c111 = grid[i[:, 0] + 1, i[:, 1] + 1, i[:, 2] + 1]
    v = (c000 * (1 - fx) * (1 - fy) * (1 - fz) +
         c100 * fx       * (1 - fy) * (1 - fz) +
         c010 * (1 - fx) * fy       * (1 - fz) +
         c001 * (1 - fx) * (1 - fy) * fz       +
         c110 * fx       * fy       * (1 - fz) +
         c101 * fx       * (1 - fy) * fz       +
         c011 * (1 - fx) * fy       * fz       +
         c111 * fx       * fy       * fz)
    out[idx] = v
    return out


def make_spherical_grid(L_max: int, N_r: int, r_max: float):
    """Arrays for (r, cos theta, phi) and integration weights.

    Angular grid: Gauss-Legendre on cos theta with N_theta = L_max + 1 nodes,
    uniform on phi with N_phi = 2 L_max + 2 nodes. Exactly integrates angular
    polynomials of degree <= 2 L_max, which is what SH expansion to L_max needs.

    NB (fit.py tunable): r ranges over (0, r_max). To focus the FRF on the
    intramolecular-vector annulus, set r_max ~ ligand diameter and consider
    dropping the innermost shell (the Patterson origin peak carries no
    orientational signal).
    """
    dr = r_max / N_r
    r = (np.arange(N_r, dtype=np.float64) + 0.5) * dr
    cos_t, w_t = roots_legendre(L_max + 1)
    sin_t = np.sqrt(1.0 - cos_t * cos_t)
    theta = np.arccos(cos_t)
    N_phi = 2 * L_max + 2
    phi = np.arange(N_phi, dtype=np.float64) * (2 * np.pi / N_phi)
    w_phi = 2 * np.pi / N_phi
    return r, dr, theta, cos_t, sin_t, w_t, phi, w_phi


def sample_density_on_spheres(grid_density: np.ndarray, origin: np.ndarray,
                              spacing: float, centre: np.ndarray,
                              r, theta, phi) -> np.ndarray:
    """Sample the voxel density at (centre + r.n_hat(theta, phi)) for all
    shell/angle combinations. Returns (N_r, N_theta, N_phi)."""
    N_r = r.shape[0]
    N_t = theta.shape[0]
    N_p = phi.shape[0]
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    x = (r[:, None, None] * sin_t[None, :, None] * cos_p[None, None, :]
         + centre[0])
    y = (r[:, None, None] * sin_t[None, :, None] * sin_p[None, None, :]
         + centre[1])
    z = (r[:, None, None] * cos_t[None, :, None]
         + centre[2])
    x_b, y_b, z_b = np.broadcast_arrays(x, y, z)
    pts = np.stack([x_b, y_b, z_b], axis=-1).reshape(-1, 3)
    vals = trilinear_sample(grid_density, origin, spacing, pts)
    return vals.reshape(N_r, N_t, N_p)


def precompute_Y_conj(theta: np.ndarray, phi: np.ndarray,
                      w_t: np.ndarray, w_phi: float, L_max: int) -> np.ndarray:
    """Precompute conj(Y_lm(theta_i, phi_j)) x w_t(i) x w_phi for SH analysis.

    Returns (L_max+1, 2 L_max + 1, N_theta, N_phi) complex. The forward SH
    transform of any rho is then einsum('rij,lmij->rlm', rho, Yc). Depends only
    on (L, angular grid) -> precompute once and broadcast (see fit.py)."""
    N_t = theta.shape[0]
    N_p = phi.shape[0]
    out = np.zeros((L_max + 1, 2 * L_max + 1, N_t, N_p), dtype=DTYPE_SH)
    for l in range(L_max + 1):  # noqa: E741
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, m, theta[:, None], phi[None, :])
            out[l, m + L_max] = (np.conj(Y) * w_t[:, None] * w_phi).astype(DTYPE_SH)
    return out


def sh_expand_fast(rho: np.ndarray, Y_conj: np.ndarray) -> np.ndarray:
    """Forward SH transform using precomputed conj-Y-with-weights.

    rho:    (N_r, N_theta, N_phi) real fp32; Y_conj from precompute_Y_conj.
    Returns f_lm: (N_r, L+1, 2L+1) complex64.

    The fp32 guard below is load-bearing: a fp64 rho would silently promote the
    einsum contraction to complex128 and lose the speed of the complex64 path.
    """
    if rho.dtype != DTYPE_REAL:
        rho = rho.astype(DTYPE_REAL)
    out = np.einsum("rij,lmij->rlm", rho, Y_conj, optimize=True)
    return out.astype(DTYPE_SH, copy=False)


def sh_expand(rho, theta, phi, w_t, w_phi, L_max):
    """Slow path kept for clarity — builds Y_conj then calls sh_expand_fast."""
    Y_conj = precompute_Y_conj(theta, phi, w_t, w_phi, L_max)
    return sh_expand_fast(rho, Y_conj)


def wigner_small_d(l: int, beta: float) -> np.ndarray:  # noqa: E741
    """Real Wigner small-d matrix d^l_{m,m'}(beta), shape (2l+1, 2l+1), indexed
    at (m+l, m'+l). Explicit Jacobi-sum form, stable for l up to ~20 in fp64."""
    out = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.float64)
    c = math.cos(beta / 2.0)
    s = math.sin(beta / 2.0)
    log_fact = np.zeros(2 * l + 2, dtype=np.float64)
    for i in range(1, 2 * l + 2):
        log_fact[i] = log_fact[i - 1] + math.log(i)
    for m in range(-l, l + 1):
        for mp in range(-l, l + 1):
            k_lo = max(0, m - mp)
            k_hi = min(l + m, l - mp)
            if k_lo > k_hi:
                continue
            pref_log = 0.5 * (log_fact[l + m] + log_fact[l - m]
                              + log_fact[l + mp] + log_fact[l - mp])
            s_acc = 0.0
            for k in range(k_lo, k_hi + 1):
                ec = 2 * l + m - mp - 2 * k
                es = mp - m + 2 * k
                denom_log = (log_fact[k] + log_fact[l + m - k]
                             + log_fact[l - mp - k] + log_fact[mp - m + k])
                t_log = pref_log - denom_log
                t = ((-1.0) ** k) * math.exp(t_log)
                if ec > 0:
                    t *= c ** ec
                if es > 0:
                    t *= s ** es
                s_acc += t
            out[m + l, mp + l] = s_acc
    return out


def wigner_D(l: int, alpha: float, beta: float, gamma: float) -> np.ndarray:  # noqa: E741
    """Complex Wigner D-matrix D^l_{m,m'}(alpha, beta, gamma), (2l+1, 2l+1).
    Convention: D^l_{m,m'}(R) = exp(-i m alpha) . d^l_{m,m'}(beta) . exp(-i m' gamma)."""
    d = wigner_small_d(l, beta)
    m = np.arange(-l, l + 1, dtype=np.float64)
    phase_a = np.exp(-1j * m * alpha)
    phase_c = np.exp(-1j * m * gamma)
    return (phase_a[:, None] * d * phase_c[None, :]).astype(DTYPE_SH)


def cross_corr_tensor(f_target: np.ndarray, f_probe: np.ndarray,
                      r: np.ndarray, dr: float, L_max: int) -> list:
    """X_l[m, m'] = Sum_r conj(f_target_lm(r)) . f_probe_lm'(r) . r^2 . dr.
    Returns a list of L_max+1 complex64 matrices of shape (2l+1, 2l+1).
    Target is conjugated -> the rotation function is the plain contraction
    C(R) = Sum_l Sum_{m,m'} X_l[m,m'] . D^l_{m,m'}(R) (no transpose/conjugate)."""
    rsq_dr = (r * r * dr).astype(DTYPE_REAL)
    out = []
    for l in range(L_max + 1):  # noqa: E741
        m_slice = slice(L_max - l, L_max + l + 1)
        a = f_target[:, l, m_slice]
        b = f_probe[:, l, m_slice]
        X = np.einsum("r,rm,rn->mn", rsq_dr, np.conj(a), b, optimize=True)
        out.append(X.astype(DTYPE_SH, copy=False))
    return out


def rotation_score(X_l: list, alpha: float, beta: float,
                   gamma: float, L_max: int) -> float:
    """Scalar rotation-function value at one (alpha, beta, gamma)."""
    total = 0.0 + 0.0j
    for l in range(L_max + 1):  # noqa: E741
        D = wigner_D(l, alpha, beta, gamma)
        total += np.tensordot(D, X_l[l], axes=2)
    return float(total.real)


def precompute_D_batch(euler: np.ndarray, L_max: int) -> list:
    """Wigner-D matrices for all (alpha, beta, gamma) in ``euler`` (N x 3, ZYZ)
    and every l <= L_max. Returns a list of (N, 2l+1, 2l+1) complex64 arrays.
    Depends only on (L, rotation set) -> precompute once and broadcast."""
    N = euler.shape[0]
    alpha = euler[:, 0]
    beta = euler[:, 1]
    gamma = euler[:, 2]
    out = []
    for l in range(L_max + 1):  # noqa: E741
        sz = 2 * l + 1
        D_l = np.zeros((N, sz, sz), dtype=DTYPE_SH)
        m_arr = np.arange(-l, l + 1, dtype=np.float64)
        for k in range(N):
            d = wigner_small_d(l, float(beta[k]))
            phase_a = np.exp(-1j * m_arr * alpha[k])
            phase_c = np.exp(-1j * m_arr * gamma[k])
            D_l[k] = (phase_a[:, None] * d * phase_c[None, :]).astype(DTYPE_SH)
        out.append(D_l)
    return out


def score_all_rotations(X_l: list, D_batch: list) -> np.ndarray:
    """For precomputed D_batch[l] of shape (N, 2l+1, 2l+1), evaluate the
    rotation function for every rotation. Returns (N,) real fp32."""
    N = D_batch[0].shape[0]
    score = np.zeros(N, dtype=DTYPE_SH)
    for D_l, X in zip(D_batch, X_l):
        score += np.einsum("rmn,mn->r", D_l, X, optimize=True)
    return score.real.astype(DTYPE_REAL)

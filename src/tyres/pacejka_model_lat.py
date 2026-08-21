import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.optimize import least_squares, differential_evolution
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root

TYRE_DATA_PATH = ROOT / "data/tyres/RunData_Cornering_Matlab_SI_Round9"  # folder containing the B2356run*.mat TTC run files

# tire filter — Hoosier 43075 16x7.5-10 R20, 8" rim
TIRE_ID_SUBSTRINGS = ["43075", "16x7.5", "8 inch rim"]

FZ_MIN, FZ_MAX = 100.0, 1300.0   # [N] valid load window
SA_MAX = 11.0                    # [deg] valid slip angle window; sweeps reverse near +/-12
                                 # and the samples around the reversal are pure transient
PRESSURE_TARGET = 83.0           # [kPa]
PRESSURE_TOL = 5.0               # [kPa]
V_MIN = 10.0                     # [kph] drop stopped / creep segments (belt not rolling)
SMOOTH_REVS = 1.0                # [-] rolling mean window in wheel revolutions (0 disables)
FZ_STEP_MAX = 50.0               # [N] drop samples whose smoothing window straddles a load step
DECIMATE = 3                     # keep every Nth sample after filtering

_data_cache = None
_coeffs_cache = None
_mz_coeffs_cache = None


def _mat_str(val):
    """Coerce a loadmat string field (char array / object array) to a plain str."""
    arr = np.asarray(val)
    return str(arr.item()).strip() if arr.size == 1 else str(arr).strip()


def _rolling_mean(x, win):
    """Centered rolling mean, win being a per-sample window width in samples."""
    n = x.size
    c = np.concatenate([[0.0], np.cumsum(x, dtype=float)])
    i = np.arange(n)
    half = win // 2
    lo = np.clip(i - half, 0, n)
    hi = np.clip(i + half + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)


def _smooth_run(run, fs):
    """
    Average every channel over one wheel revolution.
    The rig carries a once-per-rev force variation (tyre/wheel non-uniformity, ~9 Hz at
    40 kph) worth roughly +/-3 Nm on MZ at every load — invisible against 57 Nm at 1200 N,
    but bigger than the 5 Nm signal itself at 200 N. One revolution of averaging cancels it.
    Returns the smoothed channels and a mask dropping windows that straddle a load step.
    """
    v = np.maximum(run["V"] / 3.6, 0.5)                          # [m/s]
    circ = 2 * np.pi * np.maximum(run["RL"], 1.0) / 100.0        # [m], RL is in cm
    win = np.clip(np.round(SMOOTH_REVS * fs * circ / v), 3, 201).astype(int)

    out = {k: _rolling_mean(v_, win) for k, v_ in run.items() if k != "RL"}
    fz_var = _rolling_mean(run["FZ"] ** 2, win) - out["FZ"] ** 2
    keep = np.sqrt(np.maximum(fz_var, 0.0)) < FZ_STEP_MAX
    return out, keep


def load_data():
    """
    Load and filter all matching TTC cornering runs from TYRE_DATA_PATH.
    Each run is averaged over one wheel revolution first, see _smooth_run.
    Returns a dict of 1D numpy arrays: SA [deg], FY [N], MZ [Nm], FZ [N, positive], IA [deg].
    """
    global _data_cache
    if _data_cache is not None:
        return _data_cache

    run_files = sorted(glob.glob(os.path.join(TYRE_DATA_PATH, "B2356run*.mat")))
    if not run_files:
        raise FileNotFoundError(f"no B2356run*.mat files found in '{TYRE_DATA_PATH}'")

    chans = ["SA", "FY", "MZ", "FZ", "IA", "P", "V"]
    acc = {c: [] for c in chans}
    keep_list = []
    n_raw = 0

    print("loading hoosier 43075 16x7.5-10 r20 (8 inch rim) runs...")
    for fpath in run_files:
        # cheap check first: just the tire id
        info = loadmat(fpath, variable_names=["tireid"])
        tire_str = _mat_str(info["tireid"])
        if not all(sub in tire_str for sub in TIRE_ID_SUBSTRINGS):
            continue

        d = loadmat(fpath, variable_names=chans + ["RL", "ET"])
        run = {c: np.asarray(d[c]).squeeze().astype(float) for c in chans + ["RL"]}
        n_raw += run["SA"].size

        # smooth per run so the rolling window never spans a file boundary
        if SMOOTH_REVS > 0:
            et = np.asarray(d["ET"]).squeeze().astype(float)
            run, keep = _smooth_run(run, 1.0 / np.median(np.diff(et)))
        else:
            keep = np.ones(run["SA"].size, dtype=bool)

        for c in chans:
            acc[c].append(run[c])
        keep_list.append(keep)
        print(f"  loaded: {os.path.basename(fpath)}  ({tire_str})")

    if not keep_list:
        raise RuntimeError(
            "no Hoosier 43075 16x7.5-10 R20 (8 inch rim) runs matched. "
            "check TYRE_DATA_PATH and the tire id filter."
        )

    A = {c: np.concatenate(acc[c]) for c in chans}
    keep = np.concatenate(keep_list)
    print(f"total raw data points: {n_raw}\n")

    FZ_pos = np.abs(A["FZ"])  # TTC stores FZ negative

    valid = (
        keep
        & (FZ_pos > FZ_MIN) & (FZ_pos < FZ_MAX)
        & (np.abs(A["SA"]) < SA_MAX)
        & (np.abs(A["P"] - PRESSURE_TARGET) < PRESSURE_TOL)
        & (A["V"] > V_MIN)
    )

    sl = slice(None, None, DECIMATE)
    SA, FY, MZ = A["SA"][valid][sl], A["FY"][valid][sl], A["MZ"][valid][sl]
    FZ, IA = FZ_pos[valid][sl], A["IA"][valid][sl]

    print(f"data points after filtering: {valid.sum()}  "
          f"(using every {DECIMATE}-th -> {SA.size} points)\n")

    _data_cache = {"SA": SA, "FY": FY, "MZ": MZ, "FZ": FZ, "IA": IA}
    return _data_cache


def _mf_lateral(p, alpha_deg, Fz, gamma_deg=0.0):
    """
    Pacejka BNP 1989 lateral magic formula.
    alpha_deg, gamma_deg in DEGREES (matches this fit's calibration). Fz in N.
    Vectorized: alpha_deg / Fz / gamma_deg may be arrays (broadcastable).
    """
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 = p

    C = a0
    D = a1 * Fz**2 + a2 * Fz
    BCD = a3 * np.sin(2 * np.arctan(Fz / a4)) * (1 - a5 * np.abs(gamma_deg))
    B = np.divide(BCD, C * D, out=np.zeros_like(np.asarray(BCD, dtype=float)), where=(C * D) != 0)
    E = a6 * Fz + a7
    SH = a8 * gamma_deg + a9 * Fz + a10
    SV = a11 * Fz * gamma_deg + a12 * Fz + a13

    phi = alpha_deg + SH
    FY = D * np.sin(C * np.arctan(B * phi - E * (B * phi - np.arctan(B * phi)))) + SV
    return FY


# def generate_coeffs(seed=0):
#     global _coeffs_cache
#     if _coeffs_cache is not None:
#         return _coeffs_cache

#     data = load_data()
#     SA, FY, FZ, IA = data["SA"], data["FY"], data["FZ"], data["IA"]

#     # ---- stage 1: anchor peak-force load curve (unchanged) ----
#     fz_bin_centers = np.arange(150, 1251, 50)
#     fz_bin_tol = 55.0
#     fzc, dpk = [], []
#     for fz in fz_bin_centers:
#         sel = (np.abs(FZ - fz) < fz_bin_tol) & (np.abs(SA) > 5) & (np.abs(IA) < 0.6)
#         if sel.sum() > 40:
#             v = np.sort(np.abs(FY[sel]))
#             fzc.append(fz)
#             dpk.append(v[max(0, round(0.90 * len(v)) - 1)])
#     fzc = np.asarray(fzc, dtype=float)
#     dpk = np.asarray(dpk, dtype=float)

#     A = np.column_stack([fzc**2, fzc])
#     ab, *_ = np.linalg.lstsq(A, -dpk, rcond=None)
#     a1_fix, a2_fix = ab
#     print(f"stage 1 — anchored load curve: a1 = {a1_fix:.4e}, a2 = {a2_fix:.4f}")

#     def expand(free):
#         a0, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 = free
#         return np.array([a0, a1_fix, a2_fix, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13])

#     def residuals(free):
#         p = expand(free)
#         model = _mf_lateral(p, SA, FZ, IA)
#         return (FY - model) / FZ

#     def cost_fn(free):
#         r = residuals(free)
#         return np.sum(r ** 2)

#     lb = np.array([1.2, -5000, 100, 0.0, -5e-4, -1.0, -2, -1e-2, -5, -10, -2, -200])
#     ub = np.array([1.8, 0, 5000, 0.5, 5e-4, 1.5, 2, 1e-2, 5, 10, 2, 200])

#     # ---- stage 2: global search with differential_evolution ----
#     print("stage 2 — global search (differential_evolution)...")
#     de_result = differential_evolution(
#         cost_fn,
#         bounds=list(zip(lb, ub)),
#         seed=seed,
#         maxiter=300,
#         popsize=25,
#         tol=1e-9,
#         mutation=(0.5, 1.5),
#         recombination=0.7,
#         polish=False,   # we'll do our own robust-loss polish below
#         updating="deferred",
#         # workers=-1,     # parallelize across cores; drop this if it causes issues on your setup
#     )
#     print(f"  DE best cost (L2, normalized) = {de_result.fun:.6f}")

#     # ---- polish with robust loss to guard against outliers ----
#     print("stage 2b — polishing with robust loss (least_squares, soft_l1)...")
#     polish_result = least_squares(
#         residuals,
#         de_result.x,
#         bounds=(lb, ub),
#         method="trf",
#         loss="cauchy",
#         f_scale=0.05,   # tune against your residual scale — see note below
#     )

#     p_fit = expand(polish_result.x)
#     rms_err = np.sqrt(np.mean((FY - _mf_lateral(p_fit, SA, FZ, IA)) ** 2))
#     print(f"fit complete — rms error: {rms_err:.2f} N\n")

#     coeff_names = ["a0 (C)", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
#                    "a8", "a9", "a10", "a11", "a12", "a13"]
#     print("fitted coefficients:")
#     for name, val in zip(coeff_names, p_fit):
#         print(f"  {name:<14} = {val:12.6f}")

#     savemat(ROOT / "data/coeffs/hoosier_r20_tire_params_lat.mat", {
#         "coeffs": p_fit,
#         "coeff_names": coeff_names,
#         "rms_error_N": rms_err,
#         "tire": "Hoosier 43075 16x7.5-10 R20",
#         "source": "FSAE TTC Round 9, Calspan Tire Research Facility",
#         "model": "Pacejka BNP 1989 Lateral",
#     })
#     print("tire parameters saved to hoosier_r20_tire_params_lat.mat")

#     _coeffs_cache = p_fit
#     return p_fit

def generate_coeffs(n_starts=8, seed=0):
    """
    Two-stage fit:
      stage 1 — anchor D(Fz) = a1*Fz^2 + a2*Fz to the data's peak |FY| per load bin
      stage 2 — multi-start bounded nonlinear least squares on the remaining
                 12 shape parameters, with a1/a2 held fixed at stage-1 values
    Returns the fitted 14-element coefficient array and caches it.
    """
    global _coeffs_cache
    if _coeffs_cache is not None:
        return _coeffs_cache

    data = load_data()
    SA, FY, FZ, IA = data["SA"], data["FY"], data["FZ"], data["IA"]

    # ---- stage 1: anchor peak-force load curve ----
    fz_bin_centers = np.arange(150, 1251, 50)
    fz_bin_tol = 55.0
    fzc, dpk = [], []
    for fz in fz_bin_centers:
        sel = (np.abs(FZ - fz) < fz_bin_tol) & (np.abs(SA) > 5) & (np.abs(IA) < 0.6)
        if sel.sum() > 40:
            v = np.sort(np.abs(FY[sel]))
            fzc.append(fz)
            dpk.append(v[max(0, round(0.90 * len(v)) - 1)])  # 90th percentile proxy
    fzc = np.asarray(fzc, dtype=float)
    dpk = np.asarray(dpk, dtype=float)

    A = np.column_stack([fzc**2, fzc])
    ab, *_ = np.linalg.lstsq(A, -dpk, rcond=None)
    a1_fix, a2_fix = ab
    print(f"stage 1 — anchored load curve: a1 = {a1_fix:.4e}, a2 = {a2_fix:.4f}")
    print(f"          peak mu  {abs(a1_fix*200 + a2_fix):.2f} (at 200 N) -> "
          f"{abs(a1_fix*1200 + a2_fix):.2f} (at 1200 N)")

    # ---- stage 2: multi-start fit of remaining 12 shape parameters ----
    # free vector order: [a0, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13]
    def expand(free):
        a0, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 = free
        return np.array([a0, a1_fix, a2_fix, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13])

    def residuals(free):
        p = expand(free)
        model = _mf_lateral(p, SA, FZ, IA)
        return (FY - model) / FZ  # load-normalized residual

    lb = np.array([1.2, -5000, 100, 0.0, -5e-4, -1.0, -2, -1e-2, -5, -10, -2, -200])
    ub = np.array([1.8, 0, 5000, 0.5, 5e-4, 1.5, 2, 1e-2, 5, 10, 2, 200])

    p_nom_free = np.array([1.45, -800, 1900, 0.01, 0.0, 0.60, 0.0, 1e-5, 0.0, 0.0, 0.0, 0.0])

    rng = np.random.default_rng(seed)
    best_cost = np.inf
    best_free = p_nom_free

    print(f"stage 2 — fitting shape parameters ({n_starts} starts)...")
    for s in range(n_starts):
        x0 = p_nom_free if s == 0 else rng.uniform(lb, ub)
        try:
            result = least_squares(residuals, x0, bounds=(lb, ub), method="trf", loss="cauchy", f_scale=0.05)
        except Exception as err:
            print(f"  start {s+1}/{n_starts}  skipped ({err})")
            continue

        cost = 2 * result.cost  # least_squares cost is 0.5*sum(res^2)
        print(f"  start {s+1}/{n_starts}  normalized cost = {cost:.5f}")
        if cost < best_cost:
            best_cost = cost
            best_free = result.x

    p_fit = expand(best_free)
    rms_err = np.sqrt(np.mean((FY - _mf_lateral(p_fit, SA, FZ, IA)) ** 2))
    print(f"\nbest normalized cost = {best_cost:.5f}")
    print(f"fit complete — rms error: {rms_err:.2f} N\n")

    coeff_names = ["a0 (C)", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
                   "a8", "a9", "a10", "a11", "a12", "a13"]
    print("fitted coefficients:")
    for name, val in zip(coeff_names, p_fit):
        print(f"  {name:<14} = {val:12.6f}")

    savemat(ROOT / "data/coeffs/hoosier_r20_tire_params_lat.mat", {
        "coeffs": p_fit,
        "coeff_names": coeff_names,
        "rms_error_N": rms_err,
        "tire": "Hoosier 43075 16x7.5-10 R20",
        "source": "FSAE TTC Round 9, Calspan Tire Research Facility",
        "model": "Pacejka BNP 1989 Lateral",
    })
    print("\ntire parameters saved to hoosier_r20_tire_params.mat")

    _coeffs_cache = p_fit
    return p_fit


def _mf_aligning(p, alpha_deg, Fz, gamma_deg=0.0):
    """
    Pacejka BNP 1989 aligning moment magic formula.
    alpha_deg, gamma_deg in DEGREES. Fz in N. Returns MZ in Nm.
    Vectorized: alpha_deg / Fz / gamma_deg may be arrays (broadcastable).
    """
    (c0, c1, c2, c3, c4, c5, c6, c7, c8,
     c9, c10, c11, c12, c13, c14, c15, c16, c17) = p

    C = c0
    D = c1 * Fz**2 + c2 * Fz
    BCD = (c3 * Fz**2 + c4 * Fz) * (1 - c6 * np.abs(gamma_deg)) * np.exp(-c5 * Fz)
    B = np.divide(BCD, C * D, out=np.zeros_like(np.asarray(BCD, dtype=float)), where=(C * D) != 0)
    E = (c7 * Fz**2 + c8 * Fz + c9) * (1 - c10 * np.abs(gamma_deg))
    SH = c11 * gamma_deg + c12 * Fz + c13
    SV = (c14 * Fz**2 + c15 * Fz) * gamma_deg + c16 * Fz + c17

    phi = alpha_deg + SH
    MZ = D * np.sin(C * np.arctan(B * phi - E * (B * phi - np.arctan(B * phi)))) + SV
    return MZ


def generate_coeffs_mz():
    """
    Three-stage aligning moment fit:
      stage 1 — anchor D(Fz) = c1*Fz^2 + c2*Fz to the signed peak MZ per load bin
      stage 2 — per-load-bin magic formula curves sharing one C, giving B/E/SH/SV vs Fz
      stage 3 — regress the remaining c-coefficients from those, then polish on all data
    A single blind 18-parameter fit collapses to a step function, hence the staging.
    Returns the fitted 18-element coefficient array and caches it.
    """
    global _mz_coeffs_cache
    if _mz_coeffs_cache is not None:
        return _mz_coeffs_cache

    data = load_data()
    SA, MZ, FZ, IA = data["SA"], data["MZ"], data["FZ"], data["IA"]

    # ---- stage 1: anchor peak-moment load curve ----
    # MZ peaks at low slip angle, so bin on the positive-SA side between 0.5 and 6 deg
    fz_bin_centers = np.arange(150, 1251, 50)
    fz_bin_tol = 55.0
    fzc, dpk = [], []
    for fz in fz_bin_centers:
        sel = (np.abs(FZ - fz) < fz_bin_tol) & (SA > 0.5) & (SA < 6.0) & (np.abs(IA) < 0.6)
        if sel.sum() > 40:
            v = np.sort(np.abs(MZ[sel]))
            peak = v[max(0, round(0.90 * len(v)) - 1)]  # 90th percentile proxy
            fzc.append(fz)
            dpk.append(np.sign(np.median(MZ[sel])) * peak)
    fzc = np.asarray(fzc, dtype=float)
    dpk = np.asarray(dpk, dtype=float)

    A = np.column_stack([fzc**2, fzc])
    ab, *_ = np.linalg.lstsq(A, dpk, rcond=None)
    c1_fix, c2_fix = ab
    print(f"stage 1 (mz) — anchored load curve: c1 = {c1_fix:.4e}, c2 = {c2_fix:.4e}")
    print(f"               peak |Mz|  {abs(c1_fix*200**2 + c2_fix*200):.1f} Nm (at 200 N) -> "
          f"{abs(c1_fix*1200**2 + c2_fix*1200):.1f} Nm (at 1200 N)")

    # ---- stage 2: per-bin curves with a shared C ----
    def _curve(B, C, D, E, SH, SV, alpha):
        phi = alpha + SH
        return D * np.sin(C * np.arctan(B * phi - E * (B * phi - np.arctan(B * phi)))) + SV

    centers, segs = [], []
    for fz in np.arange(200, 1251, 100):
        sel = (np.abs(FZ - fz) < 60.0) & (np.abs(IA) < 0.6)
        if sel.sum() < 200:
            continue
        centers.append(float(fz))
        segs.append((SA[sel], MZ[sel], c1_fix * fz**2 + c2_fix * fz))
    nb = len(centers)
    if nb < 3:
        raise RuntimeError("not enough populated load bins for the mz fit")

    # free vector: [C, B_i..., E_i..., SH_i..., SV_i...]
    def bin_residuals(x):
        C = x[0]
        B, E, SH, SV = (x[1:1+nb], x[1+nb:1+2*nb], x[1+2*nb:1+3*nb], x[1+3*nb:])
        return np.concatenate([
            (_curve(B[i], C, segs[i][2], E[i], SH[i], SV[i], segs[i][0]) - segs[i][1])
            / max(abs(segs[i][2]), 1.0)
            for i in range(nb)
        ])

    x0 = np.concatenate([[2.5], np.full(nb, 0.22), np.full(nb, 0.5), np.zeros(nb), np.zeros(nb)])
    lo = np.concatenate([[1.5], np.full(nb, 0.02), np.full(nb, -3.0), np.full(nb, -2.0), np.full(nb, -10.0)])
    hi = np.concatenate([[3.5], np.full(nb, 1.50), np.full(nb, 1.2), np.full(nb, 2.0), np.full(nb, 10.0)])

    print(f"stage 2 (mz) — per-bin fit over {nb} load bins...")
    rb = least_squares(bin_residuals, x0, bounds=(lo, hi), method="trf", loss="cauchy", f_scale=0.15)
    C_fit = rb.x[0]
    B_b, E_b, SH_b, SV_b = (rb.x[1:1+nb], rb.x[1+nb:1+2*nb], rb.x[1+2*nb:1+3*nb], rb.x[1+3*nb:])
    fz_b = np.asarray(centers)
    D_b = np.asarray([sg[2] for sg in segs])
    BCD_b = B_b * C_fit * D_b
    print(f"               shared C = {C_fit:.3f}, B = {B_b.min():.3f}..{B_b.max():.3f}")

    # ---- stage 3: regress c-coefficients, then polish ----
    def bcd_res(q):
        return (q[0] * fz_b**2 + q[1] * fz_b) * np.exp(-q[2] * fz_b) - BCD_b

    c3, c4, c5 = least_squares(bcd_res, [0.0, 0.026, 0.0]).x
    c7, c8, c9 = np.linalg.lstsq(np.column_stack([fz_b**2, fz_b, np.ones(nb)]), E_b, rcond=None)[0]
    c12, c13 = np.linalg.lstsq(np.column_stack([fz_b, np.ones(nb)]), SH_b, rcond=None)[0]
    c16, c17 = np.linalg.lstsq(np.column_stack([fz_b, np.ones(nb)]), SV_b, rcond=None)[0]

    # free vector: [c0, c3..c17]; camber terms start at zero and are freed in the polish
    free0 = np.array([C_fit, c3, c4, c5, 0.0, c7, c8, c9, 0.0, 0.0, c12, c13, 0.0, 0.0, c16, c17])

    def expand(free):
        return np.concatenate([[free[0], c1_fix, c2_fix], free[1:]])

    mz_scale = max(np.percentile(np.abs(MZ), 95), 1.0)  # keep residuals ~O(1)

    def residuals(free):
        return (MZ - _mf_aligning(expand(free), SA, FZ, IA)) / mz_scale

    span = np.array([0.5, abs(c3)*3 + 1e-5, abs(c4)*3 + 0.02, abs(c5)*3 + 5e-4, 0.5,
                     abs(c7)*3 + 2e-5, abs(c8)*3 + 0.02, abs(c9)*3 + 1.0, 0.5, 0.5,
                     abs(c12)*3 + 2e-3, abs(c13)*3 + 0.5, 1e-4, 0.05,
                     abs(c16)*3 + 0.01, abs(c17)*3 + 2.0])

    rms0 = np.sqrt(np.mean((MZ - _mf_aligning(expand(free0), SA, FZ, IA)) ** 2))
    print(f"stage 3 (mz) — seeded rms {rms0:.2f} Nm, polishing...")
    rp = least_squares(residuals, free0, bounds=(free0 - span, free0 + span),
                       method="trf", loss="cauchy", f_scale=0.1)

    p_fit = expand(rp.x)
    rms_err = np.sqrt(np.mean((MZ - _mf_aligning(p_fit, SA, FZ, IA)) ** 2))
    print(f"mz fit complete — rms error: {rms_err:.2f} Nm\n")

    coeff_names = [f"c{i}" for i in range(18)]
    coeff_names[0] = "c0 (C)"
    print("fitted mz coefficients:")
    for name, val in zip(coeff_names, p_fit):
        print(f"  {name:<14} = {val:14.6e}")

    savemat(ROOT / "data/coeffs/hoosier_r20_tire_params_mz.mat", {
        "coeffs": p_fit,
        "coeff_names": coeff_names,
        "rms_error_Nm": rms_err,
        "tire": "Hoosier 43075 16x7.5-10 R20",
        "source": "FSAE TTC Round 9, Calspan Tire Research Facility",
        "model": "Pacejka BNP 1989 Aligning Moment",
    })
    print("\ntire parameters saved to hoosier_r20_tire_params_mz.mat")

    _mz_coeffs_cache = p_fit
    return p_fit


def plot_mz_vs_sa(fn_buckets=None):
    """
    Aligning moment vs slip angle, one curve per Fz bucket in fn_buckets [N].
    Model curves (zero camber) overlaid on measured scatter near each bucket.
    Defaults to the loads the rig actually tested, see test_load_levels.
    """
    fn_buckets = test_load_levels() if fn_buckets is None else fn_buckets
    data = load_data()
    SA, MZ, FZ, IA = data["SA"], data["MZ"], data["FZ"], data["IA"]
    p_fit = generate_coeffs_mz()

    sa_vec = np.linspace(-14, 14, 300)
    fz_tol = 60.0
    cmap = plt.get_cmap("jet", len(fn_buckets))

    fig, ax = plt.subplots(figsize=(9, 6))
    for k, fz in enumerate(fn_buckets):
        idx = (np.abs(FZ - fz) < fz_tol) & (np.abs(IA) < 0.5)
        if idx.sum() < 30:
            print(f"skipping F_Z = {fz} N (only {idx.sum()} points nearby)")
            continue

        ax.scatter(SA[idx], MZ[idx], s=5, color=cmap(k), alpha=0.20)

        mz_pred = _mf_aligning(p_fit, sa_vec, fz * np.ones_like(sa_vec), 0.0)
        ax.plot(sa_vec, mz_pred, color=cmap(k), linewidth=2, label=f"F_Z = {fz:g} N")

    ax.axhline(0, linestyle="--", color="k", alpha=0.25)
    ax.axvline(0, linestyle="--", color="k", alpha=0.25)
    ax.set_xlabel("slip angle  \u03b1  [deg]")
    ax.set_ylabel("aligning moment  M_Z  [Nm]")
    ax.set_title("aligning moment vs slip angle (lines = model, dots = measured)")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(ROOT / "figures/tyres/Mz vs SA.png")


def test_load_levels(min_pts=200, gap=80.0):
    """
    The discrete Fz levels the tyre was actually tested at, rounded to 10 N.
    The rig steps load in ~220 N increments and holds each level for a full SA sweep, so
    plotting at round numbers like 1000 N compares a model curve against data taken 56 N
    away. Bucket on these instead.
    """
    FZ = np.sort(load_data()["FZ"])
    groups = np.split(FZ, np.where(np.diff(FZ) > gap)[0] + 1)
    return [round(g.mean() / 10) * 10 for g in groups if g.size >= min_pts]


def plot_fy_vs_sa(fn_buckets=None):
    """
    Lateral force vs slip angle, one curve per Fz bucket in fn_buckets [N].
    Model curves (zero camber) overlaid on measured scatter near each bucket.
    Defaults to the loads the rig actually tested, see test_load_levels.
    """
    fn_buckets = test_load_levels() if fn_buckets is None else fn_buckets
    data = load_data()
    SA, FY, FZ, IA = data["SA"], data["FY"], data["FZ"], data["IA"]
    p_fit = generate_coeffs()

    sa_vec = np.linspace(-14, 14, 300)
    fz_tol = 60.0
    cmap = plt.get_cmap("jet", len(fn_buckets))

    fig, ax = plt.subplots(figsize=(9, 6))
    for k, fz in enumerate(fn_buckets):
        idx = (np.abs(FZ - fz) < fz_tol) & (np.abs(IA) < 0.5)
        if idx.sum() < 30:
            print(f"skipping F_Z = {fz} N (only {idx.sum()} points nearby)")
            continue

        ax.scatter(SA[idx], FY[idx], s=5, color=cmap(k), alpha=0.20)

        fy_pred = _mf_lateral(p_fit, sa_vec, fz * np.ones_like(sa_vec), 0.0)
        ax.plot(sa_vec, fy_pred, color=cmap(k), linewidth=2, label=f"F_Z = {fz:g} N")

    ax.axhline(0, linestyle="--", color="k", alpha=0.25)
    ax.axvline(0, linestyle="--", color="k", alpha=0.25)
    ax.set_xlabel("slip angle  α  [deg]")
    ax.set_ylabel("lateral force  F_Y  [N]")
    ax.set_title("lateral force vs slip angle (lines = model, dots = measured)")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(ROOT / "figures/tyres/Fy vs SA.png")


def plot_fy_vs_fn(sa_buckets):
    """
    Lateral force vs normal force, one curve per slip angle in sa_buckets [deg].
    Model-only (continuous Fz sweep, zero camber) — this is the direct
    visualization of load sensitivity: each curve's sub-linear rise with Fz
    shows the diminishing-returns behavior discussed earlier.
    """
    p_fit = generate_coeffs()

    fz_vec = np.linspace(100, 1300, 300)
    cmap = plt.get_cmap("viridis", len(sa_buckets))

    fig, ax = plt.subplots(figsize=(9, 6))
    for k, sa in enumerate(sa_buckets):
        fy_pred = np.abs(_mf_lateral(p_fit, sa * np.ones_like(fz_vec), fz_vec, 0.0))
        ax.plot(fz_vec, fy_pred, color=cmap(k), linewidth=2, label=f"α = {sa:g}°")

    ax.set_xlabel("normal force  F_Z  [N]")
    ax.set_ylabel("lateral force  |F_Y|  [N]")
    ax.set_title("lateral force vs normal force — load sensitivity")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(ROOT / "figures/tyres/Fy vs FN.png")



if __name__ == "__main__":
    load_data()
    generate_coeffs()
    generate_coeffs_mz()
    plot_fy_vs_sa()
    plot_fy_vs_fn(sa_buckets=[1.92, 4, 8, 12])
    plot_mz_vs_sa()
    plt.show()
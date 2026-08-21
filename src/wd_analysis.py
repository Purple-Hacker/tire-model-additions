import numpy as np
import pprint
from scipy.io import loadmat
from scipy.optimize import minimize_scalar
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root

G = 9.81  # m/s^2

TOTAL_CAR_MASS = 288
WHEELBASE = 1.53          # m
CG_HEIGHT = 0.30          # m
TIRE_RADIUS = 0.2032      # m
BRAKE_BIAS_FRONT = 0.70   # fraction of total braking force carried by front axle

TRACK_WIDTH = 1.245  # m, front == rear
CAMBER = 0.0         # rad — set nonzero if you want camber effects included

WD_FRONT_CHOICES = [0.52, 0.5, 0.48, 0.46, 0.44, 0.42]
WD_LAT_ACCEL_OUTPUTS = dict()
WD_LONG_ACCEL_OUTPUTS = dict()
WD_BRAKE_ACCEL_OUTPUTS = dict()

# For weight distribution of around 0.48/0.5
INITIAL_WD_FRONT = 0.48
INITIAL_LAT_ACCEL = 1.5
INITIAL_LONG_ACCEL = 1.2
INITIAL_BRAKE_ACCEL = 1.2

CONVERGENCE_TOL = 0.005

PACEJKA_BRAKE_COEFFS_PATH = ROOT / "data/coeffs/hoosier_r20_tire_params_brake.mat"
PACEJKA_LONG_COEFFS_PATH = ROOT / "data/coeffs/hoosier_r20_tire_params_long.mat"
PACEJKA_LAT_COEFFS_PATH = ROOT / "data/coeffs/hoosier_r20_tire_params_lat.mat"

# For typical SA and SR values
LAT_ACCEL_SA = 5
LONG_ACCEL_SR = 0.15

_long_coeffs = None
_lat_coeffs = None


def load_pacejka_lat_coeffs():
    global _lat_coeffs
    mat = loadmat(PACEJKA_LAT_COEFFS_PATH)

    _lat_coeffs = np.asarray(mat["coeffs"]).flatten()  # expects a0..a13
    return _lat_coeffs


def load_pacejka_long_coeffs():
    global _long_coeffs
    mat = loadmat(PACEJKA_LONG_COEFFS_PATH)

    _long_coeffs = np.asarray(mat["coeffs"]).flatten()  # expects b0..b13
    return _long_coeffs


def _pacejka_fx(kappa, Fz, b):
    """
    Pacejka '89 (BNP) pure longitudinal slip model.
    Fz in N, kappa as a ratio (not %).
    """
    b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13 = b

    C = b0
    D = Fz * (b1 * Fz + b2)
    BCD = (b3 * Fz**2 + b4 * Fz) * np.exp(-b5 * Fz)
    B = BCD / (C * D) if C * D != 0 else 0.0
    E = b6 * Fz**2 + b7 * Fz + b8
    Sh = b9 * Fz + b10
    Sv = b11 * Fz + b12  # some BNP variants set this to 0; adjust if yours does

    kx = kappa + Sh
    Fx = D * np.sin(C * np.arctan(B * kx - E * (B * kx - np.arctan(B * kx)))) + Sv
    return Fx


def _max_long_fx(Fz, b, kappa_bounds=(0.001, 0.4)):
    """
    Traction-limited peak: numerically find the slip ratio that
    maximizes |Fx| at a given Fz, return (Fx_max, kappa_opt).
    """
    result = minimize_scalar(
        lambda k: -_pacejka_fx(k, Fz, b),
        bounds=kappa_bounds,
        method="bounded",
    )
    kappa_opt = result.x
    fx_max = -result.fun
    # return fx_max, kappa_opt

    return _pacejka_fx(LONG_ACCEL_SR, Fz, b), LONG_ACCEL_SR


def _pacejka_fy(alpha, Fz, a, gamma=CAMBER):
    """
    Pacejka '89 (BNP) pure lateral slip model.
    Fz in N, alpha in radians, gamma (camber) in radians.
    """
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 = a

    C = a0
    D = Fz * (a1 * Fz + a2)
    BCD = a3 * np.sin(2 * np.arctan(Fz / a4)) * (1 - a5 * abs(gamma))
    B = BCD / (C * D) if C * D != 0 else 0.0
    E = a6 * Fz + a7
    Sh = a8 * gamma + a9 * Fz + a10
    Sv = a11 * Fz * gamma + a12 * Fz + a13

    ax = alpha + Sh
    Fy = D * np.sin(C * np.arctan(B * ax - E * (B * ax - np.arctan(B * ax)))) + Sv
    return Fy

# alpha_bounds_deg indicates that slip angle is bounded between 0.5 and 15.0
def _max_lat_fy(Fz, a, alpha_bounds_deg=(-15.0, -0.5)):
    """
    Traction-limited peak: numerically find the slip angle that
    maximizes |Fy| at a given Fz, return (Fy_max, alpha_opt_rad).
    """
    # lo, hi = np.radians(alpha_bounds_deg[0]), np.radians(alpha_bounds_deg[1])
    # for angle in range
    lo = alpha_bounds_deg[0]
    hi = alpha_bounds_deg[1]

    result = minimize_scalar(
        lambda al: -_pacejka_fy(al, Fz, a),
        bounds=(lo, hi),
        method="bounded",
    )

    alpha_opt = result.x
    fy_max = -result.fun

    # fy_max : represents the max value, in N
    # alpha_out : represents the slip angle of the max value, in radians
    # return fy_max, alpha_opt
    return _pacejka_fy(LAT_ACCEL_SA, Fz, a), LAT_ACCEL_SA


def _calculate_swd(wd):
    """Static wheel loads (N), 50/50 left-right split assumed."""
    total_weight = TOTAL_CAR_MASS * G
    front_axle_load = wd * total_weight
    rear_axle_load = (1 - wd) * total_weight
    return {
        "FO": front_axle_load / 2,
        "FI": front_axle_load / 2,
        "RO": rear_axle_load / 2,
        "RI": rear_axle_load / 2,
    }


# ---------------------------------------------------------------------
# Lateral — stubbed pending track width (front/rear)
# ---------------------------------------------------------------------

def generate_lat_accels():
    for curr_wd in WD_FRONT_CHOICES:
        swd_loads = _calculate_swd(curr_wd)

        lat_accel_guess = INITIAL_LAT_ACCEL
        next_lat_accel_guess = _calculate_lat_accel(lat_accel_guess, swd_loads)

        while abs(lat_accel_guess - next_lat_accel_guess) > CONVERGENCE_TOL:
            lat_accel_guess = next_lat_accel_guess
            next_lat_accel_guess = _calculate_lat_accel(lat_accel_guess, swd_loads)

        WD_LAT_ACCEL_OUTPUTS[curr_wd] = next_lat_accel_guess


def _calculate_lat_accel(lat_accel_guess, swd_loads):
    """
    lat_accel_guess is in g's. Cornering transfers load from inner
    to outer wheels on BOTH axles (front and rear both engage,
    unlike the RWD-only longitudinal case).
    """
    ay = lat_accel_guess * G  # m/s^2
    weight_transfer = (TOTAL_CAR_MASS * ay * CG_HEIGHT) / TRACK_WIDTH  # N, per-axle

    front_axle_static = swd_loads["FO"] + swd_loads["FI"]
    rear_axle_static = swd_loads["RO"] + swd_loads["RI"]

    fo_load = front_axle_static / 2 + weight_transfer / 2
    fi_load = front_axle_static / 2 - weight_transfer / 2
    ro_load = rear_axle_static / 2 + weight_transfer / 2
    ri_load = rear_axle_static / 2 - weight_transfer / 2

    # clip in case weight transfer exceeds static load (inside wheel lifting)
    fi_load = max(fi_load, 0.0)
    ri_load = max(ri_load, 0.0)

    fy_fo, _ = _max_lat_fy(fo_load, _lat_coeffs)
    fy_fi, _ = _max_lat_fy(fi_load, _lat_coeffs)
    fy_ro, _ = _max_lat_fy(ro_load, _lat_coeffs)
    fy_ri, _ = _max_lat_fy(ri_load, _lat_coeffs)

    total_lateral_force = fy_fo + fy_fi + fy_ro + fy_ri
    new_lat_accel = total_lateral_force / (TOTAL_CAR_MASS * G)  # back to g's

    return new_lat_accel


# ---------------------------------------------------------------------
# Longitudinal — traction-limited, RWD
# ---------------------------------------------------------------------

def generate_long_accels():
    for curr_wd in WD_FRONT_CHOICES:
        swd_loads = _calculate_swd(curr_wd)

        long_accel_guess = INITIAL_LONG_ACCEL
        next_long_accel_guess = _calculate_long_accel(long_accel_guess, swd_loads)

        while abs(long_accel_guess - next_long_accel_guess) > CONVERGENCE_TOL:
            long_accel_guess = next_long_accel_guess
            next_long_accel_guess = _calculate_long_accel(long_accel_guess, swd_loads)

        WD_LONG_ACCEL_OUTPUTS[curr_wd] = next_long_accel_guess


def _calculate_long_accel(long_accel_guess, swd_loads):
    """
    long_accel_guess is in g's. RWD -> only rear corners produce
    tractive force. Weight transfers rearward under acceleration.
    """
    ax = long_accel_guess * G  # m/s^2
    weight_transfer = (TOTAL_CAR_MASS * ax * CG_HEIGHT) / WHEELBASE  # N, axle-level

    # RWD: acceleration moves load onto the rear, off the front
    rear_axle_static = swd_loads["RO"] + swd_loads["RI"]
    rear_axle_dynamic = rear_axle_static + weight_transfer

    # assume even left/right split (no lateral component in pure long accel)
    rear_corner_load = rear_axle_dynamic / 2

    fx_ro, _ = _max_long_fx(rear_corner_load, _long_coeffs)
    fx_ri, _ = _max_long_fx(rear_corner_load, _long_coeffs)

    total_tractive_force = fx_ro + fx_ri
    new_long_accel = total_tractive_force / (TOTAL_CAR_MASS * G)  # back to g's

    return new_long_accel

def scale_long_accels():
    max_long_accel_even_dist = WD_LONG_ACCEL_OUTPUTS[0.5]
    long_accel_scaling_factor = INITIAL_LONG_ACCEL / max_long_accel_even_dist

    for wd in WD_LONG_ACCEL_OUTPUTS:
        WD_LONG_ACCEL_OUTPUTS[wd] *= long_accel_scaling_factor

# ---------------------------------------------------------------------
# Braking — traction-limited, fixed 0.7/0.3 bias
# ---------------------------------------------------------------------

def generate_brake_accels():
    for curr_wd in WD_FRONT_CHOICES:
        swd_loads = _calculate_swd(curr_wd)

        brake_accel_guess = INITIAL_BRAKE_ACCEL
        next_brake_accel_guess = _calculate_brake_accel(brake_accel_guess, swd_loads)

        while abs(brake_accel_guess - next_brake_accel_guess) > CONVERGENCE_TOL:
            brake_accel_guess = next_brake_accel_guess
            next_brake_accel_guess = _calculate_brake_accel(brake_accel_guess, swd_loads)

        WD_BRAKE_ACCEL_OUTPUTS[curr_wd] = next_brake_accel_guess


def _calculate_brake_accel(brake_accel_guess, swd_loads):
    """
    brake_accel_guess is a positive magnitude in g's.
    Braking moves load onto the front, off the rear.
    Bias is FIXED (0.7 front / 0.3 rear) -- whichever axle hits its
    friction limit first sets the achievable deceleration.
    """
    ax = brake_accel_guess * G
    weight_transfer = (TOTAL_CAR_MASS * ax * CG_HEIGHT) / WHEELBASE

    front_axle_static = swd_loads["FO"] + swd_loads["FI"]
    rear_axle_static = swd_loads["RO"] + swd_loads["RI"]

    front_axle_dynamic = front_axle_static + weight_transfer
    rear_axle_dynamic = rear_axle_static - weight_transfer

    front_corner_load = front_axle_dynamic / 2
    rear_corner_load = rear_axle_dynamic / 2

    fx_fo, _ = _max_long_fx(front_corner_load, _long_coeffs)
    fx_fi, _ = _max_long_fx(front_corner_load, _long_coeffs)
    fx_ro, _ = _max_long_fx(rear_corner_load, _long_coeffs)
    fx_ri, _ = _max_long_fx(rear_corner_load, _long_coeffs)

    max_front_force = fx_fo + fx_fi
    max_rear_force = fx_ro + fx_ri

    # total achievable braking force, given the fixed bias, is capped
    # by whichever axle saturates first
    total_force_front_limited = max_front_force / BRAKE_BIAS_FRONT
    total_force_rear_limited = max_rear_force / (1 - BRAKE_BIAS_FRONT)

    total_braking_force = min(total_force_front_limited, total_force_rear_limited)
    new_brake_accel = total_braking_force / (TOTAL_CAR_MASS * G)

    return new_brake_accel

def scale_lat_accels():
    max_lat_accel_even_dist = WD_LAT_ACCEL_OUTPUTS[0.5]
    lat_accel_scaling_factor = INITIAL_LAT_ACCEL / max_lat_accel_even_dist

    for wd in WD_LAT_ACCEL_OUTPUTS:
        WD_LAT_ACCEL_OUTPUTS[wd] *= lat_accel_scaling_factor

if __name__ == "__main__":
    load_pacejka_lat_coeffs()
    load_pacejka_long_coeffs()

    generate_long_accels()
    scale_long_accels()

    generate_lat_accels()
    scale_lat_accels()

    pprint.pprint(WD_LONG_ACCEL_OUTPUTS, width=40)
    pprint.pprint(WD_LAT_ACCEL_OUTPUTS, width=40)

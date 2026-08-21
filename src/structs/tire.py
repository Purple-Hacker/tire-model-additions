from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import math

from utils.constants import MU_SCALE

"""
Pacejka BNP 1989 Magic Formula Structure
%
%   C   = a0                                       (shape factor)
%   D   = a1*Fz^2 + a2*Fz                          (peak force — load sensitivity)
%   BCD = a3*sin(2*atan(Fz/a4))*(1 - a5*|gamma|)   (cornering stiffness; negative in sae convention)
%   E   = a6*Fz + a7                               (curvature factor)
%   SH  = a8*gamma + a9*Fz + a10                   (horizontal shift — plysteer/camber)
%   SV  = a11*Fz*gamma + a12*Fz + a13              (vertical shift — conicity/camber thrust)
%
%   phi = alpha + SH
%   FY  = D*sin(C*atan(B*phi - E*(B*phi - atan(B*phi)))) + SV
%
% note: B = BCD / (C*D)
%       slip angle (alpha) and camber (gamma) in degrees, forces in newtons
"""

@dataclass
class Tire:
    front_static_camber: float = -1.901  # deg
    rear_static_camber: float = -2.635   # deg

    # TODO: fill in real values -- placeholders below are 0.0 so the class
    # is at least constructible; using these for a real sim run will
    # silently zero out camber gain until they're set.
    camber_gain_roll: dict[str, float] = field(default_factory=lambda: {
        "FO": 0.0,
        "FI": 0.0,
        "RO": 0.0,
        "RI": 0.0,
    })
    camber_gain_pitch: dict[str, float] = field(default_factory=lambda: {
        "F": 0.0,
        "R": 0.0,
    })

    all_static_camber: dict[str, float] = field(init=False)

    # Pacejka BNP 1989 coefficients, a0-a13 (14 values), set via load_coefficients
    coeffs: dict[str, float] | None = field(default=None, init=False)
    coeff_source: str | None = field(default=None, init=False)
    coeff_rms_error_n: float | None = field(default=None, init=False)

    COEFF_NAMES = (
        "a0", "a1", "a2", "a3", "a4", "a5", "a6",
        "a7", "a8", "a9", "a10", "a11", "a12", "a13",
    )

    def load_coefficients(self, path: str | Path) -> None:
        """Load fitted Pacejka BNP 1989 coefficients from a .mat file.

        Accepts both layouts: the flat one written by src/tyres/pacejka_model_*.py
        (top-level `coeffs`) and the legacy MATLAB `tireParams` struct. Populates
        self.coeffs as a name -> value dict, e.g. self.coeffs["a3"].

        Coefficients are raw TTC -- the friction derate lives in MU_SCALE, applied
        at force evaluation, not here.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tire coefficient file not found: {path}")

        mat = loadmat(path, struct_as_record=False, squeeze_me=True)

        if "tireParams" in mat:
            src = mat["tireParams"]
            raw_coeffs = src.coeffs
            get = lambda k, d: getattr(src, k, d)
        elif "coeffs" in mat:
            raw_coeffs = mat["coeffs"]
            get = lambda k, d: mat.get(k, d)
        else:
            raise KeyError(
                f"Expected 'coeffs' or 'tireParams' in {path}, "
                f"found: {[k for k in mat if not k.startswith('__')]}"
            )

        raw_coeffs = np.asarray(raw_coeffs).squeeze()

        if len(raw_coeffs) != len(self.COEFF_NAMES):
            raise ValueError(
                f"Expected {len(self.COEFF_NAMES)} coefficients "
                f"(a0-a{len(self.COEFF_NAMES) - 1}), got {len(raw_coeffs)} "
                f"from {path}. Check the MATLAB script's coeff_names still "
                f"matches the fit vector p_fit."
            )

        self.coeffs = dict(zip(self.COEFF_NAMES, raw_coeffs.tolist()))
        self.coeff_source = str(get("tire", path.name))
        self.coeff_rms_error_n = float(get("rms_error_N", float("nan")))

    def lat_force(self, vertical_load: float, camber_angle: float, slip_angle: float) -> float:
        """Pacejka BNP 1989 lateral force model.

        Args:
            vertical_load: normal (vertical) load, N. Must be positive -- the fit was
                done against abs(FZ) (see hoosier_r20_tire_model_lat.m section 2:
                `FZ_pos = abs(FZ_all)`), so a negative fz here is a caller bug,
                not a valid "unloaded tire" input.
            camber_angle: inclination angle, deg (gamma)
            slip_angle: slip angle, deg (alpha)

        Returns:
            Fy, N, scaled by MU_SCALE. Sign follows the SAE convention baked
            into the fit (D is negative -- see stage-1 fitting notes).
        """
        if self.coeffs is None:
            raise RuntimeError("Tire coefficients not loaded -- call load_coefficients() first")
        if vertical_load < 0:
            raise ValueError(f"fz must be a positive load magnitude, got {vertical_load}")

        c = self.coeffs  # shorthand

        C = c["a0"]
        D = c["a1"] * vertical_load**2 + c["a2"] * vertical_load
        BCD = c["a3"] * math.sin(2 * math.atan(vertical_load / c["a4"])) * (1 - c["a5"] * abs(camber_angle))
        E = c["a6"] * vertical_load + c["a7"]
        SH = c["a8"] * camber_angle + c["a9"] * vertical_load + c["a10"]
        SV = c["a11"] * vertical_load * camber_angle + c["a12"] * vertical_load + c["a13"]

        # B = BCD / (C*D); guard the degenerate case where D ~ 0 (e.g. fz ~ 0)
        if abs(C * D) < 1e-9:
            return 0.0
        B = BCD / (C * D)

        phi = slip_angle + SH
        fy = D * math.sin(C * math.atan(B * phi - E * (B * phi - math.atan(B * phi)))) + SV

        # derate from the ttc 3m belt to real asphalt -- coeffs are raw ttc
        return MU_SCALE * fy

    def long_force(self, vertical_load: float, camber_angle: float, slip_ratio: float) -> float:
                # TODO: implement, mirroring lat_force. remember the MU_SCALE derate.
        raise NotImplementedError

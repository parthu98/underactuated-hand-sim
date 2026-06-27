"""Joint-angle extraction for the tendon-driven finger validation rig.

Converts per-marker in-plane orientations (``phi``, in DEGREES) into the three
finger joint angles (MCP, PIP, DIP), zeroed at a reference (fully-extended)
pose.

Sign convention
---------------
Flexion (curling) is POSITIVE, matching the analytical model convention where
``theta >= 0`` denotes flexion. A configurable ``flexion_sign`` (+1 / -1)
multiplies all three joint angles so the physical curling direction maps to
positive angles regardless of how the markers are mounted / which way the
camera sees the finger.

Each joint angle is the change, relative to a captured zero pose, of the
in-plane orientation difference between two consecutive markers::

    theta_mcp = (phi_M1 - phi_M0) - base_mcp
    theta_pip = (phi_M2 - phi_M1) - base_pip
    theta_dip = (phi_M3 - phi_M2) - base_dip

The ``base_*`` terms are the relative orientations captured at the zero pose.
Because every constant marker-mounting offset appears identically in both the
live frame and the base capture, subtracting ``base_*`` cancels those offsets,
leaving a clean change-from-reference angle. All differences are unwrapped into
``(-180, 180]`` so a joint sweeping past +/-180 deg never produces a spurious
jump.
"""

from typing import Dict, Optional

import numpy as np


def wrap_deg(angle: float) -> float:
    """Wrap an angle (in degrees) into the half-open interval ``(-180, 180]``."""
    wrapped = (angle + 180.0) % 360.0 - 180.0
    # ``% 360`` maps -180 -> -180; we want the half-open (-180, 180] interval,
    # so fold the boundary value up to +180.
    if wrapped == -180.0:
        wrapped = 180.0
    return float(wrapped)


class JointAngles:
    """Compute zeroed, flexion-positive MCP/PIP/DIP angles from marker phis."""

    def __init__(self, flexion_sign: int = +1):
        self.flexion_sign = int(flexion_sign)
        # ``_base`` is either None (not yet zeroed) or a dict with the captured
        # relative orientations for each joint, in degrees.
        self._base: Optional[Dict[str, float]] = None
        # Separate zero baseline for the plane-free 3D angle path (see
        # ``set_zero_3d`` / ``compute_3d``). ``None`` until the straight pose is
        # captured; treated as 0 (raw bend) before then.
        self._base_3d: Optional[Dict[str, float]] = None

    def set_zero(self, phi: Dict[int, Optional[float]]) -> bool:
        """Record the reference (zero) pose from marker orientations.

        ``phi`` maps marker index 0..3 to an in-plane orientation in degrees
        (or ``None`` if that marker was not detected). All four markers must be
        present; otherwise the base is left unchanged and ``False`` is returned.
        """
        if any(phi.get(i) is None for i in range(4)):
            return False
        p0 = float(phi[0])
        p1 = float(phi[1])
        p2 = float(phi[2])
        p3 = float(phi[3])
        self._base = {
            "mcp": wrap_deg(p1 - p0),
            "pip": wrap_deg(p2 - p1),
            "dip": wrap_deg(p3 - p2),
        }
        return True

    def set_zero_3d(self, raw: Dict[str, Optional[float]]) -> bool:
        """Record the straight-pose reference for the plane-free 3D angle path.

        ``raw`` maps ``"mcp"/"pip"/"dip"`` to the signed 3D inter-segment bend
        (degrees) from the tracker. All three must be present; otherwise the
        baseline is left unchanged and ``False`` is returned. This is the 3D
        analogue of :meth:`set_zero` — the captured straight-pose bends cancel
        each segment's constant LED-mounting offset when subtracted in
        :meth:`compute_3d`.
        """
        if any(raw.get(j) is None for j in ("mcp", "pip", "dip")):
            return False
        self._base_3d = {j: float(raw[j]) for j in ("mcp", "pip", "dip")}
        return True

    def compute_3d(self, raw: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """Zeroed, flexion-signed joint angles from signed 3D inter-segment bends.

        ``raw`` is the tracker's signed 3D bend per joint (plane-free, so large
        out-of-plane curl is not inflated the way the planar ``phi`` difference
        is). Subtracts the straight-pose baseline (0 until :meth:`set_zero_3d`),
        wraps into ``(-180, 180]``, and applies ``flexion_sign`` — mirroring
        :meth:`compute` so the rest of the pipeline is unchanged.
        """
        base = self._base_3d if self._base_3d is not None else {
            "mcp": 0.0, "pip": 0.0, "dip": 0.0,
        }
        result: Dict[str, Optional[float]] = {}
        for name in ("mcp", "pip", "dip"):
            v = raw.get(name)
            if v is None:
                result[name] = None
                continue
            theta = wrap_deg(float(v) - base[name])
            result[name] = float(self.flexion_sign * theta)
        return result

    def is_zeroed(self) -> bool:
        """Return ``True`` once a reference pose has been captured."""
        return self._base is not None

    def set_flexion_sign(self, sign: int) -> None:
        """Set the flexion sign (+1 or -1) applied to all three joint angles."""
        self.flexion_sign = int(sign)

    def compute(self, phi: Dict[int, Optional[float]]) -> Dict[str, Optional[float]]:
        """Compute change-from-reference joint angles in degrees.

        Returns a dict ``{"mcp": ..., "pip": ..., "dip": ...}`` where each value
        is a float (flexion positive) or ``None`` if either of that joint's two
        markers was missing this frame. If no zero pose has been captured the
        base is treated as 0 (raw relative angle).
        """
        base = self._base if self._base is not None else {
            "mcp": 0.0, "pip": 0.0, "dip": 0.0,
        }

        # (joint name, distal marker idx, proximal marker idx)
        joint_markers = (
            ("mcp", 1, 0),
            ("pip", 2, 1),
            ("dip", 3, 2),
        )

        result: Dict[str, Optional[float]] = {}
        for name, hi, lo in joint_markers:
            phi_hi = phi.get(hi)
            phi_lo = phi.get(lo)
            if phi_hi is None or phi_lo is None:
                result[name] = None
                continue
            rel = wrap_deg(float(phi_hi) - float(phi_lo))
            theta = wrap_deg(rel - base[name])
            result[name] = float(self.flexion_sign * theta)
        return result

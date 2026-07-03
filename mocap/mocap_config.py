#!/usr/bin/env python3
"""Mocap-rig-only configuration (PhaseSpace finger tracking).

Kept SEPARATE from the repo's root ``config.py`` on purpose: the root config is
the single source of truth for the *physical* finger (geometry, joint limits,
spring stiffnesses, spool radius, moment arms) and is consumed by the analytical
model + the hardware servo. This file holds ONLY mocap-specific knobs — the
PhaseSpace server address, which LED ids belong to which finger segment, and
where calibration / results live. Nothing here changes the physics.

Marker layout (decided with the user)
-------------------------------------
Two LED markers per segment, on FOUR segments — base/palm, proximal, middle,
distal (8 LEDs total). Each segment's direction vector is ``p_far - p_near``;
the joint angle is the in-plane angle between consecutive segment vectors, which
two points give directly (no rigid body needed — the OWL SDK requires >=4
markers for a rigid body, and we only need a direction).

``MOCAP_SEGMENT_MARKER_IDS`` is ordered ``[base, proximal, middle, distal]``;
each entry is ``(near_id, far_id)`` where ``near`` is the LED nearer the palm and
``far`` is the LED nearer the fingertip, so ``far - near`` points distally along
the finger. EDIT these to match the LED ids you assigned in the PhaseSpace
Master Client.
"""

import os

# --- PhaseSpace server / streaming ------------------------------------------
MOCAP_SERVER_IP = "192.168.1.230"   # Impulse server address (same as the test scripts)
MOCAP_EVENT_TIMEOUT_US = 1_000_000  # nextEvent() timeout [microseconds]
# Connect as a read-only slave so the PhaseSpace Master Client can stay open on
# the same server at the same time. Set False to be the sole/primary client.
MOCAP_SLAVE = True

# --- LED id -> segment mapping ----------------------------------------------
# Ordered [base, proximal, middle, distal]; (near_id, far_id) per segment.
# Index 0..3 here lines up with hardware/joints.py's marker indices
# (0=base, 1=prox, 2=mid, 3=dist), so the existing MCP/PIP/DIP differencing and
# zeroing work unchanged.
#
# As-built LED placement (2026-07-03 remount, ids 8-15): near = nearer the
# palm, far = nearer the fingertip; far - near points distally.
#   base / palm        : near=15, far=14
#   MCP / proximal link: near=13, far=12
#   PIP / middle link  : near=11, far=10
#   DIP / distal link  : near=9,  far=8
MOCAP_SEGMENT_MARKER_IDS = (
    (38, 39),   # base / palm
    (36, 37),   # proximal phalanx  (MCP link)
    (34, 35),   # middle phalanx    (PIP link)
    (32, 33),     # distal phalanx    (DIP link)
)

SEGMENT_LABELS = ("base", "prox", "mid", "dist")

# --- flexion-plane definition -----------------------------------------------
# The finger flexes in a FIXED plane (no abduction). "MOCAP_VERTICAL_AXIS" is a
# misnomer left over from the original rig mount, where that plane happened to be
# the horizontal ground plane (normal = the lab's up axis, Y) -- it really means
# "the axis NORMAL to the flexion plane", whatever that plane is on THIS mount.
#
# 2026-07-03: on the current mount the finger curls in the vertical X-Y plane
# (confirmed live in PhaseSpace Master Client), not the horizontal plane, so the
# normal is Z, not Y. Using Y here (the old default) meant every joint-angle
# calculation was projecting against the wrong axis -- the real curling motion
# looked like constant out-of-plane drift that got worse at deep flexion, which
# is what caused the DIP sign to glitch (see signed_joint_angles_3d in tracker.py)
# even after that method's own sign-degeneracy guard was added; that guard is a
# safety net, this axis is the actual fix.
#
# We define the plane from the known normal — no calibration flex, no extra
# marker needed. The projection then strips every bit of out-of-plane marker
# tilt/rotation, and the straight-pose Set Zero cancels the residual constant
# in-plane offsets. Only the NORMAL affects joint angles (they are differences of
# consecutive segment angles, which are invariant to the in-plane long-axis
# reference). So the single thing to get right is which axis is normal to the
# plane the finger actually moves in.
#
# CHECK THIS LIVE: move a marker straight out of the plane the finger curls in
# and watch which raw coordinate changes.
MOCAP_VERTICAL_AXIS = 2      # axis NORMAL to the flexion plane: 0=X, 1=Y, 2=Z
# +1/-1 -- flip if flexion reads backwards after re-zeroing (try this before
# touching MOCAP_FLEXION_SIGN, which is a separate, coarser sign flip).
MOCAP_VERTICAL_SIGN = +1

# Sign of the joint angles. Flexion (servo pull) should read POSITIVE.
#
# 2026-07-03: after remounting the LEDs on the new marker ids (see
# MOCAP_SEGMENT_MARKER_IDS above), the mocap_validation_uniform_medium sweep
# showed flexion reading NEGATIVE (theta_mcp_exp went -2 -> -24 deg while the
# plane-free theta_mcp_3d for the same frames went +16 -> +38 deg — same pull,
# opposite sign; PIP/DIP cross the same way once flex is large enough). That
# means the finger curls opposite to what this sign used to produce, so it
# flips to +1. If a future remount reverses it again, use the dashboard's
# "FLIP θ SIGN" button to check live, then update this default to match.
MOCAP_FLEXION_SIGN = +1      # +1 or -1

# --- servo ΔL travel --------------------------------------------------------
# Manual/GO/jog tendon-pull ceiling [mm]. The hardware dashboard hard-caps the
# target field at 25 mm; the mocap rig raises it (and the servo soft cap) so the
# tendon can be driven well past 70 mm for the pull-out / large-flex tests.
MOCAP_MAX_DELTA_MM = 120.0

# --- paths (everything mocap-related lives inside mocap/) -------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
MOCAP_CALIB_PATH = os.path.join(_HERE, "mocap_calibration.json")
MOCAP_RESULTS_DIR = os.path.join(_HERE, "results")

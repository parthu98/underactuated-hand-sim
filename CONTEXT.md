# Session Context

**Current Task:** Hardened the load-test dashboard and servo layer for the
physical pull-out rig; committed and pushed all pending hardware changes.

**Key Decisions:**
- All three Dynamixels (finger A, B, pull) share one U2D2 bus; `_ensure_bus()`
  opens it once and attaches every motor. `--pull-port` argument removed.
- Load cell uses an empirically fitted 22.20 N/raw-lbf scale (origin-forced
  least-squares, 5 known-mass points); readouts switched from lb to kg.
- Pull-out uses hysteresis-based multi-peak detection (arm → valley → re-arm)
  instead of halting on the first force drop; capacity latches the largest peak.

**Next Steps:**
- Validate mocap angles against ArUco on the same sweep; reconcile any offset.
- Run physical pull-out trials and verify peak-detection hysteresis tuning.
- Re-add `setup-ai-context.sh` (or a hook) so the Gemini/Copilot mirrors auto-resync.

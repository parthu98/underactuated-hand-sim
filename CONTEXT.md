# Session Context

**Current Task:** Stiffness spring-set selector + START PULL logging fix on the
load-test dashboard; committed/pushed code plus distal/proximal trial data.

**Key Decisions:**
- Removed auto-tension (UI + logic + dead `TENSION_*` config); added a STIFFNESS
  dropdown (`LOAD_TEST_STIFFNESS_CONFIGS`/`_DEFAULT`) driving logged k_mcp/pip/dip.
- START PULL now opens the trial CSV first and writes a `pull_start` marker row
  so every pull is logged even if stopped before a release peak.
- Commits attributed to user only (NO-AI-WATERMARK).

**Next Steps:**
- Confirm START PULL logging fix works on the rig (couldn't run GUI locally — no Qt).
- Verify the 5 named spring sets match the real test matrix; adjust if not.
- Resolve the local cv2/NumPy 1.x↔2.x mismatch so the GUI can launch here.

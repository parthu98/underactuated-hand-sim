# Session Context

**Current Task:** Replaced load-test auto-tensioning with a stiffness
spring-set selector; committed/pushed it plus the day's trial-CSV data.

**Key Decisions:**
- Removed the current-onset auto-tension feature (UI + logic) and the dead
  `TENSION_*` config constants from the load-test dashboard.
- Added `LOAD_TEST_STIFFNESS_CONFIGS`/`_DEFAULT` in config.py; a STIFFNESS
  dropdown selects the installed spring set, driving logged k_mcp/k_pip/k_dip.
- Commits attributed to user only (NO-AI-WATERMARK); first two of the day
  back-dated to yesterday at the user's request.

**Next Steps:**
- Confirm the 5 named spring sets match the real test matrix; adjust if not.
- Run pull-out trials per stiffness set and verify CSV k-values are correct.
- Resolve the local cv2/NumPy 1.x↔2.x mismatch so the GUI can launch here.

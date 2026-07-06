# Session Context

**Current Task:** The finger-sweep CV pipeline (now `finger_sweep_cv/`, renamed
from the placeholder `Common Scripts/`) is merged into `main`. This session
fixed a proximal/distal case-label swap (labels must follow joint *stiffness*,
not observed angle -- the two conventions disagree for this mechanism) across
`run_all.py`'s `CASES` dict and ~20 renamed files + a README table on Google
Drive, and added a `--video <path> [--label ...]` flag to `run_all.py` so a
video anywhere on disk can be ingested without pre-copying it into `raw\`.
Files are edited/renamed but not yet committed -- left for the user to finish
by hand, per standing preference.

**Key Decisions:**
- Case-label convention is stiffness-based (paper-wide), not angle-based --
  don't re-derive `_dominant` labels from observed sweep angle again.
- Fingertip is derived as a per-frame geometric extremum (tip-cap circle fit),
  never tracked directly -- 6 prior approaches drifted/failed (see
  `track_points.py`'s module docstring).
- Raw videos + all generated outputs live in a local, git-ignored
  `Finger Sweep Videos/` folder created on demand; the finished dataset is
  archived on Google Drive (`Underactuated arm/`) instead of committed.
- Drive's `Proximal\`/`Distal\`/`Uniform\` folders predate this pipeline and
  are NOT reorganized to match case labels -- only the one matching filename
  in each was renamed in place (see that Drive README for the rationale).

**Next Steps:**
- Commit/push the label fix, folder rename, and CLI addition (user is
  handling this step directly).
- `Model Tracking/` (rejected CoTracker evaluation) intentionally stays local,
  out of git.

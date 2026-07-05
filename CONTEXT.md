# Session Context

**Current Task:** Added the finger-sweep CV pipeline (`Common Scripts/`) to this
repo -- joint angles + fingertip trajectory from video, for the RAL 2026 paper.
Commits are staged/organized on the `claude/bold-colden-6d2180` branch but not
yet merged to `main` or pushed -- left for the user to finish by hand.

**Key Decisions:**
- Fingertip is derived as a per-frame geometric extremum (tip-cap circle fit),
  never tracked directly -- 6 prior approaches drifted/failed (see
  `track_points.py`'s module docstring).
- Video output goes through a piped standalone ffmpeg (`video_writer.py`,
  CRF-based libx264), not `cv2.VideoWriter` -- OpenCV's own H.264 path has no
  bitrate control and bloated files ~3x for no quality gain.
- Raw videos + all generated outputs live in a local, git-ignored
  `Finger Sweep Videos/` folder created on demand; the finished dataset is
  archived on Google Drive (`Underactuated arm/`) instead of committed.

**Next Steps:**
- Finish committing/merging `Common Scripts/` into `main` and push (user is
  handling this step directly).
- `Model Tracking/` (rejected CoTracker evaluation) intentionally stays local,
  out of git.

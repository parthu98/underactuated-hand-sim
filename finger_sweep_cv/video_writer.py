"""H.264 (libx264) video writer with real CRF-based rate control.

OpenCV's own FFmpeg plugin in this environment can only encode H.264 via
Cisco's external OpenH264 library (a GPL-free, patent-encumbered-but-BSD-
licensed plugin OpenCV loads from a standalone DLL), and that plugin path
gives no way to control bitrate/quality -- `VIDEOWRITER_PROP_QUALITY` is
silently ignored (`VIDEOIO: unused parameter`), so OpenCV falls back to a
fixed per-pixel bitrate formula that comes out to ~60+ Mbps regardless of
content, nearly tripling file size versus the original mp4v renders for no
visual benefit.

This writer instead pipes raw BGR frames to the standalone ffmpeg binary
bundled by `imageio-ffmpeg`, which includes a real libx264 build with CRF
support. CRF is content-adaptive (near-static scenes compress far more than
a fixed bitrate target would allow) and is the standard way to hit a quality
target rather than a size target.
"""
import subprocess

import imageio_ffmpeg

__all__ = ["FFmpegWriter"]


class FFmpegWriter:
    """Drop-in replacement for cv2.VideoWriter's write()/isOpened()/release(),
    backed by a piped ffmpeg subprocess encoding libx264 at a fixed CRF."""

    def __init__(self, path, fps, frame_size, crf=18, preset="medium"):
        w, h = frame_size
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p",
            str(path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        self._proc.stdin.write(frame.tobytes())

    def isOpened(self):
        return self._proc.poll() is None

    def release(self):
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.wait()

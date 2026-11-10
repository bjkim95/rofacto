from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import numpy as np


class VideoWriter:
    def __init__(
        self,
        path: str | Path,
        fps: float,
        frame_size: tuple[int, int],
        input_colorspace: str = "rgb",
    ) -> None:
        self.path = Path(path)
        self.fps = float(fps)
        self.width, self.height = frame_size
        self.input_colorspace = input_colorspace
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._temp_path = self.path.parent / f".{self.path.stem}.partial-{uuid4().hex}{self.path.suffix}"
        self._frames: list[np.ndarray] = []
        self._proc: subprocess.Popen[bytes] | None = None
        self._closed = False

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            cmd = [
                ffmpeg,
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                f"{self.fps}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self._temp_path),
            ]
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def _cleanup_temp_file(self) -> None:
        self._temp_path.unlink(missing_ok=True)

    def _finalize_temp_file(self) -> None:
        os.replace(self._temp_path, self.path)

    def _to_rgb(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Frame shape {frame.shape[:2]} does not match expected {(self.height, self.width)}"
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 frame, got {frame.shape}")
        if self.input_colorspace == "rgb":
            return np.ascontiguousarray(frame)
        if self.input_colorspace == "bgr":
            return np.ascontiguousarray(frame[:, :, ::-1])
        raise ValueError(f"Unsupported input_colorspace: {self.input_colorspace}")

    def write(self, frame: np.ndarray) -> None:
        rgb = self._to_rgb(frame.astype(np.uint8, copy=False))
        if self._proc is not None:
            assert self._proc.stdin is not None
            self._proc.stdin.write(rgb.tobytes())
            return
        self._frames.append(rgb)

    def close(self) -> None:
        if self._closed:
            return
        if self._proc is not None:
            assert self._proc.stdin is not None
            self._proc.stdin.close()
            return_code = self._proc.wait()
            if return_code != 0:
                self._cleanup_temp_file()
                raise RuntimeError(f"ffmpeg exited with code {return_code} for {self.path}")
            self._proc = None
            self._finalize_temp_file()
            self._closed = True
            return
        if not self._frames:
            self._cleanup_temp_file()
            raise RuntimeError(f"No frames were written to {self.path}")
        import imageio.v3 as iio

        try:
            iio.imwrite(str(self._temp_path), np.stack(self._frames, axis=0), fps=self.fps)
            self._finalize_temp_file()
        except Exception:
            self._cleanup_temp_file()
            raise
        finally:
            self._frames.clear()
        self._closed = True

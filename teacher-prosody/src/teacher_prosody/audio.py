"""Audio I/O helpers. Decoding goes through the ffmpeg binary bundled by imageio-ffmpeg,
so mp4/mkv/webm lecture videos work without a system ffmpeg."""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_SR = 16000


def ffmpeg_exe() -> str:
    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class Audio:
    y: np.ndarray  # float32 mono, [-1, 1]
    sr: int

    @property
    def duration(self) -> float:
        return len(self.y) / self.sr

    def slice(self, start: float, end: float) -> "Audio":
        a, b = int(round(start * self.sr)), int(round(end * self.sr))
        return Audio(self.y[max(a, 0) : min(b, len(self.y))], self.sr)


def load(path: str | Path, sr: int = DEFAULT_SR, start: float | None = None, duration: float | None = None) -> Audio:
    """Decode any audio/video file to mono float32 at `sr` via ffmpeg."""
    cmd = [ffmpeg_exe(), "-nostdin", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, check=True, capture_output=True).stdout
    return Audio(np.frombuffer(out, dtype=np.float32).copy(), sr)


def save(path: str | Path, audio: Audio, subtype: str = "PCM_16") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio.y, -1, 1), audio.sr, subtype=subtype)
    return path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def frame_times(n_frames: int, hop_s: float, offset_s: float = 0.0) -> np.ndarray:
    return offset_s + np.arange(n_frames) * hop_s


def concat(chunks: list[Audio]) -> Audio:
    if not chunks:
        raise ValueError("nothing to concatenate")
    sr = chunks[0].sr
    assert all(c.sr == sr for c in chunks), "sample rates differ"
    return Audio(np.concatenate([c.y for c in chunks]).astype(np.float32), sr)


def silence(seconds: float, sr: int = DEFAULT_SR) -> Audio:
    return Audio(np.zeros(int(round(seconds * sr)), dtype=np.float32), sr)

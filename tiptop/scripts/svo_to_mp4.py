"""Offline conversion of recorded SVO files to MP4.

During a `tiptop-run --enable-recording`, cameras are recorded to ZED's native
SVO format and MP4 conversion is skipped (it OOMs the GPU while pi0.5 is loaded).
This script converts those SVO files to playable MP4 videos after the fact.
"""

import argparse
import logging
from pathlib import Path

from tiptop.perception.cameras.zed_camera import CorruptSVOError, SVOGPUMemoryError, convert_svo_to_mp4

_log = logging.getLogger(__name__)


def _find_svo_files(path: Path, recursive: bool) -> list[Path]:
    """Return SVO/SVO2 files for a single file or directory."""
    if path.is_file():
        return [path]
    pattern = "**/*.svo*" if recursive else "*.svo*"
    return sorted(p for p in path.glob(pattern) if p.suffix in (".svo", ".svo2"))


def svo_to_mp4_entrypoint():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Convert recorded SVO files to MP4 videos.")
    parser.add_argument("path", type=Path, help="SVO file, or directory containing SVO files (e.g. a run output dir).")
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Recurse into subdirectories when path is a directory."
    )
    parser.add_argument("--crf", type=int, default=20, help="ffmpeg CRF (lower = higher quality, larger file).")
    parser.add_argument("--overwrite", action="store_true", help="Re-convert even if the MP4 already exists.")
    args = parser.parse_args()

    if not args.path.exists():
        parser.error(f"Path does not exist: {args.path}")

    svo_files = _find_svo_files(args.path, args.recursive)
    if not svo_files:
        _log.warning(f"No .svo/.svo2 files found under {args.path}")
        return

    _log.info(f"Found {len(svo_files)} SVO file(s) to convert")
    corrupted: list[Path] = []
    for svo_path in svo_files:
        mp4_path = svo_path.with_suffix(".mp4")
        if mp4_path.exists() and not args.overwrite:
            _log.info(f"Skipping {svo_path.name} (MP4 exists; use --overwrite to redo)")
            continue
        # A run killed mid-recording (e.g. a crash) can leave one SVO corrupted.
        # Skip it with a warning so the remaining (good) files still convert.
        try:
            convert_svo_to_mp4(svo_path, mp4_path, crf=args.crf)
        except CorruptSVOError as e:
            _log.warning(f"Skipping corrupted SVO {svo_path.name}: {e}")
            corrupted.append(svo_path)
        except SVOGPUMemoryError as e:
            # The files are fine, the GPU is full -- so every remaining one would fail the same way.
            # Stop rather than march on and report a directory of good recordings as corrupted.
            _log.error(f"{e}")
            _log.error(f"Aborting with {len(svo_files) - svo_files.index(svo_path)} file(s) left to convert.")
            return

    if corrupted:
        _log.warning(
            f"{len(corrupted)} of {len(svo_files)} SVO file(s) were corrupted and skipped: "
            f"{', '.join(p.name for p in corrupted)}"
        )


if __name__ == "__main__":
    svo_to_mp4_entrypoint()

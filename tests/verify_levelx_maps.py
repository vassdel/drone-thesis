"""Verify every data/levelx/map/*.pkl matches the contract contextvae/data.py expects.

Contract (from contextvae/data.py:461-469 and :289-297):
    pickle payload = (semantic_map, H)
    semantic_map : np.ndarray, dtype float32, shape (3, H, W), values in [-1.0, 1.0]
    H            : np.ndarray, dtype float64, shape (3, 3), bottom row [0, 0, 1],
                   2x2 leading block invertible (det != 0)

Also cross-checks that every map referenced by data/levelx/{train,val}/*.info has
a corresponding .pkl on disk.

Run:
    python tests/verify_levelx_maps.py
Exits 0 on full pass, 1 on any failure.
"""

import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
# print(REPO_ROOT)
MAP_DIR = REPO_ROOT / "data" / "levelx" / "map"
TRAIN_DIR = REPO_ROOT / "data" / "levelx" / "train"
VAL_DIR = REPO_ROOT / "data" / "levelx" / "val"


def check_pkl(path: Path) -> list[str]:
    errs: list[str] = []
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception as e:
        return [f"unpickle failed: {e!r}"]

    if not (isinstance(payload, tuple) and len(payload) == 2):
        return [f"payload is not a 2-tuple (got {type(payload).__name__}, len={len(payload) if hasattr(payload, '__len__') else 'n/a'})"]

    sm, H = payload

    # semantic_map checks
    if not isinstance(sm, np.ndarray):
        errs.append(f"semantic_map is not np.ndarray (got {type(sm).__name__})")
    else:
        if sm.dtype != np.float32:
            errs.append(f"semantic_map dtype != float32 (got {sm.dtype})")
        if sm.ndim != 3:
            errs.append(f"semantic_map ndim != 3 (got {sm.ndim}, shape={sm.shape})")
        elif sm.shape[0] != 3:
            errs.append(f"semantic_map shape[0] != 3 (got {sm.shape})")
        if sm.size > 0:
            mn, mx = float(sm.min()), float(sm.max())
            if mn < -1.0 - 1e-6 or mx > 1.0 + 1e-6:
                errs.append(f"semantic_map out of [-1, 1] (min={mn:.4f}, max={mx:.4f})")

    # H checks
    if not isinstance(H, np.ndarray):
        errs.append(f"H is not np.ndarray (got {type(H).__name__})")
    else:
        if H.dtype != np.float64:
            errs.append(f"H dtype != float64 (got {H.dtype})")
        if H.shape != (3, 3):
            errs.append(f"H shape != (3, 3) (got {H.shape})")
        elif not np.all(np.isfinite(H)):
            errs.append("H has non-finite entries")
        else:
            if not np.allclose(H[2], [0.0, 0.0, 1.0]):
                errs.append(f"H bottom row != [0, 0, 1] (got {H[2].tolist()})")
            det2x2 = float(np.linalg.det(H[:2, :2]))
            if abs(det2x2) < 1e-12:
                errs.append(f"H[:2,:2] is singular (det={det2x2:.3e})")

    return errs


def check_info_references(map_stems: set[str]) -> list[str]:
    errs: list[str] = []
    for split_dir in (TRAIN_DIR, VAL_DIR):
        if not split_dir.is_dir():
            continue
        for info in sorted(split_dir.glob("*.info")):
            try:
                line = info.read_text().strip().splitlines()[0]
                tokens = line.split()
                if len(tokens) < 2:
                    errs.append(f"{info.relative_to(REPO_ROOT)}: malformed line '{line}'")
                    continue
                map_name = tokens[1]
                if map_name not in map_stems:
                    errs.append(f"{info.relative_to(REPO_ROOT)}: references missing map '{map_name}.pkl'")
            except Exception as e:
                errs.append(f"{info.relative_to(REPO_ROOT)}: read failed: {e!r}")
    return errs


def main() -> int:
    if not MAP_DIR.is_dir():
        print(f"ERROR: map dir not found: {MAP_DIR}", file=sys.stderr)
        return 1

    pkls = sorted(MAP_DIR.glob("*.pkl"))
    if not pkls:
        print(f"ERROR: no .pkl files in {MAP_DIR}", file=sys.stderr)
        return 1

    failed = 0
    for p in pkls:
        errs = check_pkl(p)
        if errs:
            failed += 1
            print(f"FAIL {p.name}")
            for e in errs:
                print(f"     - {e}")
        else:
            print(f"PASS {p.name}")

    print()
    print(f"Format check: {len(pkls) - failed}/{len(pkls)} files OK")

    map_stems = {p.stem for p in pkls}
    ref_errs = check_info_references(map_stems)
    if ref_errs:
        print(f"Cross-reference: {len(ref_errs)} broken .info -> .pkl link(s)")
        for e in ref_errs:
            print(f"     - {e}")
    else:
        print("Cross-reference: all train/val .info entries resolve to existing maps")

    return 0 if (failed == 0 and not ref_errs) else 1


if __name__ == "__main__":
    sys.exit(main())

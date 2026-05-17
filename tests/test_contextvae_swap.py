"""Lightweight wiring test for LSTM->ContextVAE swap (2026-05-12).

Validates that the swap done in
`uav_guidance/server_code_only/main_program/scene_recog_socket_yolov8.py`
is still in place: the new modules import cleanly, expose the documented
public API and constants, the pixel<->metric shim round-trips, and the
server file is still wired to ContextVAEInferencer.infer (not to the
removed LSTM predictor).

Heavy checks (real ckpt, ADE/FDE, PNG overlay) live next to the swap:
  - uav_guidance/server_code_only/main_program/verify_inference.py
  - uav_guidance/server_code_only/main_program/visual_sanity.py

Run:
    python tests/test_contextvae_swap.py
Exits 0 on full pass, 1 on any failure.
"""

import ast
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SWAP_DIR = REPO_ROOT / "uav_guidance" / "server_code_only" / "main_program"
MAP_PICKLE = REPO_ROOT / "data" / "levelx" / "map" / "inD_01.pkl"
VAL_TXT = REPO_ROOT / "data" / "levelx" / "val" / "inD_01.txt"
SERVER_FILE = SWAP_DIR / "scene_recog_socket_yolov8.py"

# main_program/ has no __init__.py, so the three swap modules can't be
# imported as a package. Inject the directory onto sys.path.
sys.path.insert(0, str(SWAP_DIR))


def check_imports() -> bool:
    print("=== check_imports ===")
    try:
        import contextvae_inference as ci
        import pixel_metric_shim as pms
    except Exception as e:
        print(f"  import failed: {e!r}")
        return False

    missing = []
    if not callable(getattr(ci, "ContextVAEInferencer", None)):
        missing.append("contextvae_inference.ContextVAEInferencer")
    if not callable(getattr(ci.ContextVAEInferencer, "infer", None)):
        missing.append("contextvae_inference.ContextVAEInferencer.infer")
    if not callable(getattr(pms, "PixelMetricShim", None)):
        missing.append("pixel_metric_shim.PixelMetricShim")
    for const in ("OB_HORIZON", "PRED_HORIZON", "PROTOCOL_FUTURE_STEPS", "EXT", "MAP_SIZE"):
        if not hasattr(ci, const):
            missing.append(f"contextvae_inference.{const}")

    if missing:
        for m in missing:
            print(f"  missing: {m}")
        return False
    print("  all expected symbols present")
    return True


def check_constants() -> bool:
    print("=== check_constants ===")
    import contextvae_inference as ci

    expected = {
        "OB_HORIZON": 10,
        "PRED_HORIZON": 25,
        "PROTOCOL_FUTURE_STEPS": 6,
        "EXT": 202,
        "MAP_SIZE": 224,
    }
    bad = []
    for name, want in expected.items():
        got = getattr(ci, name)
        if got != want:
            bad.append(f"{name}: got {got!r}, expected {want!r}")
    if bad:
        for b in bad:
            print(f"  {b}")
        return False
    print(f"  {expected}")
    return True


def check_shim_roundtrip() -> bool:
    print("=== check_shim_roundtrip ===")
    if not MAP_PICKLE.exists():
        print(f"  missing map pickle: {MAP_PICKLE}")
        return False
    if not VAL_TXT.exists():
        print(f"  missing val recording: {VAL_TXT}")
        return False

    from pixel_metric_shim import PixelMetricShim

    shim = PixelMetricShim(str(MAP_PICKLE))

    # Pull a real in-map (x, y) from row 0 of the val recording.
    # File format: fid aid x y heading group
    first_row = next(
        (ln for ln in VAL_TXT.read_text().splitlines() if ln.strip()),
        None,
    )
    if first_row is None:
        print(f"  val recording is empty: {VAL_TXT}")
        return False
    toks = first_row.split()
    if len(toks) < 4:
        print(f"  malformed row: {first_row!r}")
        return False
    xy = np.array([float(toks[2]), float(toks[3])], dtype=np.float64)

    # Single-point round-trip.
    u, v = shim.metric_to_pixel(*xy)
    x2, y2 = shim.pixel_to_metric(u, v)
    err = float(np.hypot(x2 - xy[0], y2 - xy[1]))
    print(f"  single: ({xy[0]:.4f}, {xy[1]:.4f}) m -> ({u:.2f}, {v:.2f}) px "
          f"-> ({x2:.6f}, {y2:.6f}) m  err={err:.2e} m")
    if not np.allclose(xy, [x2, y2], atol=1e-6):
        print(f"  FAIL: single round-trip error {err:.2e} m exceeds 1e-6")
        return False

    # Batch round-trip on three nearby points.
    xy_batch = np.array([xy, xy + [1.0, 0.0], xy + [0.0, 1.0]], dtype=np.float64)
    uv_batch = shim.metrics_to_pixel(xy_batch)
    xy_back = shim.pixels_to_metric(uv_batch)
    err_batch = float(np.linalg.norm(xy_back - xy_batch, axis=-1).max())
    print(f"  batch (N=3): max err {err_batch:.2e} m")
    if not np.allclose(xy_batch, xy_back, atol=1e-6):
        print(f"  FAIL: batch round-trip error {err_batch:.2e} m exceeds 1e-6")
        return False

    return True


def check_call_site_regression() -> bool:
    """Confirm scene_recog_socket_yolov8.py still imports the swap modules
    and calls .infer(...), and does NOT import the removed LSTM module.

    Uses ast so the comment block in scene_recog_socket_yolov8.py that
    documents the LSTM removal (and mentions `tr_pred_lstm` by name)
    does not false-positive."""
    print("=== check_call_site_regression ===")
    if not SERVER_FILE.exists():
        print(f"  missing server file: {SERVER_FILE}")
        return False

    try:
        tree = ast.parse(SERVER_FILE.read_text())
    except SyntaxError as e:
        print(f"  ast.parse failed: {e!r}")
        return False

    from_imports = {}      # module -> set of imported names
    plain_imports = set()  # `import foo` / `import foo.bar`
    infer_calls = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            from_imports.setdefault(mod, set()).update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                plain_imports.add(a.name)
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "infer":
                infer_calls += 1

    ok = True

    ci_names = from_imports.get("contextvae_inference", set())
    if "ContextVAEInferencer" not in ci_names:
        print("  missing: `from contextvae_inference import ContextVAEInferencer`")
        ok = False
    else:
        print(f"  contextvae_inference imports: {sorted(ci_names)}")

    pms_names = from_imports.get("pixel_metric_shim", set())
    if "PixelMetricShim" not in pms_names:
        print("  missing: `from pixel_metric_shim import PixelMetricShim`")
        ok = False
    else:
        print(f"  pixel_metric_shim imports: {sorted(pms_names)}")

    lstm_hits = []
    for mod, names in from_imports.items():
        if "tr_pred_lstm" in mod or "tr_pred_lstm" in names:
            lstm_hits.append(f"from {mod} import {sorted(names)}")
    for mod in plain_imports:
        if "tr_pred_lstm" in mod:
            lstm_hits.append(f"import {mod}")
    if lstm_hits:
        print("  FAIL: removed LSTM module is still imported:")
        for h in lstm_hits:
            print(f"    {h}")
        ok = False
    else:
        print("  no `tr_pred_lstm` import (LSTM removal preserved)")

    if infer_calls < 1:
        print("  FAIL: no `.infer(...)` call found in server file")
        ok = False
    else:
        print(f"  found {infer_calls} `.infer(...)` call(s)")

    return ok


def main() -> int:
    checks = [
        check_imports,
        check_constants,
        check_shim_roundtrip,
        check_call_site_regression,
    ]
    results = [(c.__name__, c()) for c in checks]
    print()
    failures = [name for name, ok in results if not ok]
    if failures:
        print(f"FAIL ({len(failures)}/{len(checks)}): {', '.join(failures)}")
        return 1
    print(f"PASS ({len(checks)}/{len(checks)})")
    print("To visually verify inference output, run: "
          "python uav_guidance/server_code_only/main_program/visual_sanity.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

REQUIRED_KEYS = (
    "state",
    "actions",
    "base_image",
    "wrist_image",
    "base_image_embeddings",
    "wrist_image_embeddings",
    "prompt",
)


def _normalize_prompt(value) -> str | None:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(()).item()
        else:
            return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value else None


def _validate_one(npz_path: Path, expected_embed_dim: int | None) -> dict:
    result = {"file": str(npz_path), "ok": True, "errors": []}
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:
        result["ok"] = False
        result["errors"].append(f"failed_to_load: {exc}")
        return result

    missing = [k for k in REQUIRED_KEYS if k not in data.files]
    if missing:
        result["ok"] = False
        result["errors"].append(f"missing_keys: {missing}")
        return result

    state = data["state"]
    actions = data["actions"]
    base_image = data["base_image"]
    wrist_image = data["wrist_image"]
    base_emb = data["base_image_embeddings"]
    wrist_emb = data["wrist_image_embeddings"]
    prompt = _normalize_prompt(data["prompt"])

    if state.ndim != 2 or state.shape[1] != 8:
        result["ok"] = False
        result["errors"].append(f"state_shape_invalid: {state.shape}")
    if actions.ndim != 2 or actions.shape[1] != 7:
        result["ok"] = False
        result["errors"].append(f"actions_shape_invalid: {actions.shape}")

    if base_image.ndim != 4 or tuple(base_image.shape[1:]) != (224, 224, 3):
        result["ok"] = False
        result["errors"].append(f"base_image_shape_invalid: {base_image.shape}")
    if wrist_image.ndim != 4 or tuple(wrist_image.shape[1:]) != (224, 224, 3):
        result["ok"] = False
        result["errors"].append(f"wrist_image_shape_invalid: {wrist_image.shape}")

    if base_emb.ndim != 2:
        result["ok"] = False
        result["errors"].append(f"base_emb_shape_invalid: {base_emb.shape}")
    if wrist_emb.ndim != 2:
        result["ok"] = False
        result["errors"].append(f"wrist_emb_shape_invalid: {wrist_emb.shape}")

    if base_emb.ndim == 2 and wrist_emb.ndim == 2 and base_emb.shape[1] != wrist_emb.shape[1]:
        result["ok"] = False
        result["errors"].append(f"embedding_dim_mismatch: {base_emb.shape[1]} vs {wrist_emb.shape[1]}")

    if expected_embed_dim is not None and base_emb.ndim == 2 and base_emb.shape[1] != expected_embed_dim:
        result["ok"] = False
        result["errors"].append(f"base_embed_dim_invalid: expected={expected_embed_dim} got={base_emb.shape[1]}")
    if expected_embed_dim is not None and wrist_emb.ndim == 2 and wrist_emb.shape[1] != expected_embed_dim:
        result["ok"] = False
        result["errors"].append(f"wrist_embed_dim_invalid: expected={expected_embed_dim} got={wrist_emb.shape[1]}")

    lengths = []
    for arr in (state, actions, base_image, wrist_image, base_emb, wrist_emb):
        if arr.ndim >= 1:
            lengths.append(arr.shape[0])
    if lengths and len(set(lengths)) != 1:
        result["ok"] = False
        result["errors"].append(f"time_length_mismatch: {lengths}")

    if prompt is None:
        result["ok"] = False
        result["errors"].append("prompt_invalid")

    result["num_steps"] = int(lengths[0]) if lengths else 0
    result["embed_dim"] = int(base_emb.shape[1]) if base_emb.ndim == 2 else None
    return result


def _find_npz_files(root: Path) -> list[Path]:
    if root.is_file() and root.name == "processed_demo.npz":
        return [root]
    return sorted(root.rglob("processed_demo.npz"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LIBERO processed_demo.npz contracts.")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="path to processed_demo.npz or directory containing episodes",
    )
    parser.add_argument("--embed-dim", type=int, default=None, help="optional expected embedding dimension")
    parser.add_argument("--report-json", type=str, default=None, help="optional output report path")
    parser.add_argument("--allow-fail", action="store_true", help="always exit 0 even when invalid files exist")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    files = _find_npz_files(root)
    if not files:
        logger.error("No processed_demo.npz found under: %s", root)
        sys.exit(1)

    logger.info("Found %d processed_demo.npz files under %s", len(files), root)
    results = [_validate_one(path, args.embed_dim) for path in files]
    failed = [r for r in results if not r["ok"]]
    passed = len(results) - len(failed)

    logger.info("Validation summary: passed=%d failed=%d", passed, len(failed))
    for item in failed[:20]:
        logger.error("FAIL: %s -> %s", item["file"], item["errors"])
    if len(failed) > 20:
        logger.error("... and %d more failures", len(failed) - 20)

    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "root": str(root),
            "total": len(results),
            "passed": passed,
            "failed": len(failed),
            "results": results,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Saved report: %s", report_path)

    if failed and not args.allow_fail:
        sys.exit(2)


if __name__ == "__main__":
    main()


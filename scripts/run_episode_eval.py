import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np
from openpi_client import msgpack_numpy
from PIL import Image
import websockets.sync.client

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def _normalize_prompt(value) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(()).item()
        else:
            return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return ""
    return value.strip()


def _resolve_npz_path(episode_path: Path) -> Path:
    if episode_path.is_file() and episode_path.name == "processed_demo.npz":
        return episode_path
    candidate = episode_path / "processed_demo.npz"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"processed_demo.npz not found: {episode_path}")


def _make_grid(images: np.ndarray, max_frames: int, max_cols: int) -> np.ndarray:
    if images.ndim != 4:
        raise ValueError(f"images must be (T,H,W,C), got {images.shape}")
    take = min(len(images), max_frames)
    if take == 0:
        raise ValueError("no frames to render")
    imgs = images[:take]
    h, w, c = imgs.shape[1:]
    if c != 3:
        raise ValueError(f"expected 3 channels, got {c}")
    cols = min(max_cols, take)
    rows = (take + cols - 1) // cols
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        r = idx // cols
        cidx = idx % cols
        canvas[r * h : (r + 1) * h, cidx * w : (cidx + 1) * w] = img
    return canvas


def _save_preview(images: np.ndarray, out_file: Path, max_frames: int, max_cols: int) -> None:
    grid = _make_grid(images, max_frames=max_frames, max_cols=max_cols)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_file)


def _run_policy_inference(
    *,
    npz_data,
    prompt: str,
    prefix: str,
    host: str,
    port: int,
    max_steps: int,
    connect_timeout: float,
) -> dict:
    uri = f"ws://{host}:{port}"
    steps = min(max_steps, int(npz_data["state"].shape[0]))
    results = []
    t0 = time.time()

    with websockets.sync.client.connect(
        uri,
        compression=None,
        max_size=None,
        open_timeout=connect_timeout,
        close_timeout=connect_timeout,
    ) as ws:
        server_metadata = msgpack_numpy.unpackb(ws.recv())
        packer = msgpack_numpy.Packer()
        for step_idx in range(steps):
            obs = {
                "query_base_image": np.asarray(npz_data["base_image"][step_idx]),
                "query_wrist_image": np.asarray(npz_data["wrist_image"][step_idx]),
                "query_state": np.asarray(npz_data["state"][step_idx], dtype=np.float32),
                "query_prompt": prompt,
                "prefix": prefix,
            }
            ws.send(packer.pack(obs))
            resp = ws.recv()
            if isinstance(resp, str):
                raise RuntimeError(f"inference server returned error string: {resp}")
            payload = msgpack_numpy.unpackb(resp)
            actions = np.asarray(payload["query_actions"], dtype=np.float32)
            results.append(
                {
                    "step_idx": step_idx,
                    "query_actions_shape": list(actions.shape),
                    "query_actions_first": actions[0].tolist() if actions.ndim >= 2 and len(actions) > 0 else [],
                }
            )

    elapsed = time.time() - t0
    return {
        "server_uri": uri,
        "steps": steps,
        "elapsed_sec": elapsed,
        "server_metadata": server_metadata,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-episode evaluation helper for review O/X loop.")
    parser.add_argument("--episode", type=str, required=True, help="episode directory or processed_demo.npz path")
    parser.add_argument("--out", type=str, required=True, help="output directory for result artifacts")
    parser.add_argument("--mode", choices=["artifacts", "infer"], default="infer")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-preview-frames", type=int, default=12)
    parser.add_argument("--max-preview-cols", type=int, default=4)
    parser.add_argument("--prefix", type=str, default=None, help="optional policy prefix override")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    episode_path = Path(args.episode).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {out_dir}; use --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = _resolve_npz_path(episode_path)
    npz_data = np.load(npz_path, allow_pickle=True)
    for key in ("state", "base_image", "wrist_image", "prompt"):
        if key not in npz_data.files:
            raise KeyError(f"missing key {key} in {npz_path}")

    prompt = _normalize_prompt(npz_data["prompt"]) or "task"
    prefix = args.prefix or npz_path.parent.name

    base_image = np.asarray(npz_data["base_image"])
    wrist_image = np.asarray(npz_data["wrist_image"])
    state = np.asarray(npz_data["state"])

    _save_preview(
        base_image,
        out_dir / "base_preview.png",
        max_frames=args.max_preview_frames,
        max_cols=args.max_preview_cols,
    )
    _save_preview(
        wrist_image,
        out_dir / "wrist_preview.png",
        max_frames=args.max_preview_frames,
        max_cols=args.max_preview_cols,
    )

    summary = {
        "episode_npz": str(npz_path),
        "mode": args.mode,
        "num_steps": int(state.shape[0]),
        "state_shape": list(state.shape),
        "base_image_shape": list(base_image.shape),
        "wrist_image_shape": list(wrist_image.shape),
        "prompt": prompt,
        "prefix": prefix,
        "artifacts": ["base_preview.png", "wrist_preview.png"],
    }

    if args.mode == "infer":
        infer_payload = _run_policy_inference(
            npz_data=npz_data,
            prompt=prompt,
            prefix=prefix,
            host=args.host,
            port=args.port,
            max_steps=args.max_steps,
            connect_timeout=args.connect_timeout,
        )
        (out_dir / "infer_results.json").write_text(json.dumps(infer_payload, indent=2), encoding="utf-8")
        summary["artifacts"].append("infer_results.json")
        summary["inference_steps"] = infer_payload["steps"]
        summary["server_uri"] = infer_payload["server_uri"]

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("saved artifacts to %s", out_dir)


if __name__ == "__main__":
    main()

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def _find_episode_dirs(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.rglob("processed_demo.npz")})


def _to_prompt(v) -> str:
    if isinstance(v, np.ndarray):
        if v.shape == ():
            v = v.item()
        elif v.size == 1:
            v = v.reshape(()).item()
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")
    if not isinstance(v, str):
        return ""
    return v


def _maybe_build_faiss(embeddings: np.ndarray, out_dir: Path) -> bool:
    try:
        import faiss
    except Exception as exc:
        logger.warning("faiss not available; skip faiss index build: %s", exc)
        return False

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype(np.float32, copy=False))
    faiss.write_index(index, str(out_dir / "index.faiss"))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local vector DB artifacts from approved demos.")
    parser.add_argument("--approved-root", type=str, default="runtime_demos/approved")
    parser.add_argument("--output-root", type=str, default="local_vector_db")
    parser.add_argument("--team-id", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-faiss", action="store_true", help="build index.faiss if faiss is installed")
    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    approved_root = (current_dir / args.approved_root).resolve()
    if not approved_root.exists():
        raise FileNotFoundError(f"approved root does not exist: {approved_root}")

    out_dir = (current_dir / args.output_root / args.team_id).resolve()
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output already exists: {out_dir}. use --overwrite to rebuild")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = _find_episode_dirs(approved_root)
    if not episode_dirs:
        raise RuntimeError(f"no approved episodes found under {approved_root}")
    logger.info("building local vector db from %d episodes", len(episode_dirs))

    all_embeddings: list[np.ndarray] = []
    all_episode_ids: list[np.ndarray] = []
    all_step_indices: list[np.ndarray] = []
    episodes_meta = []

    for ep_id, ep_dir in enumerate(episode_dirs):
        npz_path = ep_dir / "processed_demo.npz"
        data = np.load(npz_path, allow_pickle=True)
        for key in ("base_image_embeddings", "state", "actions", "prompt"):
            if key not in data.files:
                raise KeyError(f"{npz_path} missing required key: {key}")

        emb = np.asarray(data["base_image_embeddings"], dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"invalid embeddings shape in {npz_path}: {emb.shape}")

        num_steps = emb.shape[0]
        all_embeddings.append(emb)
        all_episode_ids.append(np.full((num_steps,), ep_id, dtype=np.int32))
        all_step_indices.append(np.arange(num_steps, dtype=np.int32))

        rel = ep_dir.relative_to(approved_root)
        episodes_meta.append(
            {
                "episode_id": ep_id,
                "relative_path": str(rel),
                "processed_demo_path": str(npz_path.resolve()),
                "num_steps": int(num_steps),
                "state_dim": int(np.asarray(data["state"]).shape[1]),
                "action_dim": int(np.asarray(data["actions"]).shape[1]),
                "prompt": _to_prompt(data["prompt"]),
            }
        )

    embeddings = np.concatenate(all_embeddings, axis=0)
    episode_ids = np.concatenate(all_episode_ids, axis=0)
    step_indices = np.concatenate(all_step_indices, axis=0)

    np.savez_compressed(
        out_dir / "vectors.npz",
        embeddings=embeddings,
        episode_ids=episode_ids,
        step_indices=step_indices,
    )

    (out_dir / "episodes.json").write_text(json.dumps(episodes_meta, indent=2), encoding="utf-8")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "team_id": args.team_id,
        "approved_root": str(approved_root),
        "num_episodes": len(episodes_meta),
        "num_vectors": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "files": ["vectors.npz", "episodes.json"],
    }

    if args.write_faiss:
        faiss_ok = _maybe_build_faiss(embeddings, out_dir)
        summary["faiss_index_built"] = bool(faiss_ok)
        if faiss_ok:
            summary["files"].append("index.faiss")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("done: %s", out_dir)
    logger.info("vectors=%d embedding_dim=%d", embeddings.shape[0], embeddings.shape[1])


if __name__ == "__main__":
    main()

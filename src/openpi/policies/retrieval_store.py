from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from openpi.policies.utils import EMBED_DIM

logger = logging.getLogger(__name__)


def _to_prompt(value) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(()).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return ""
    return value


def _get_action_chunk_libero(actions: np.ndarray, step_idx: int, action_horizon: int) -> np.ndarray:
    num_steps = len(actions)
    chunk = []
    for i in range(action_horizon):
        if step_idx + i < num_steps:
            chunk.append(actions[step_idx + i])
        else:
            chunk.append(np.concatenate([np.zeros(6, dtype=np.float32), actions[-1, -1:]], axis=0))
    chunk = np.stack(chunk, axis=0)
    assert chunk.shape == (action_horizon, 7), f"{chunk.shape=}"
    return chunk


@dataclass(frozen=True)
class RetrievalHit:
    episode_id: int
    step_idx: int
    global_idx: int


class BaseLiberoRetrievalStore:
    def search(self, query_embedding: np.ndarray, k: int) -> list[RetrievalHit]:
        raise NotImplementedError

    def get_context(self, hit: RetrievalHit, action_horizon: int) -> dict[str, Any]:
        raise NotImplementedError

    def get_embedding(self, hit: RetrievalHit) -> np.ndarray:
        raise NotImplementedError


class LegacyLiberoDemoStore(BaseLiberoRetrievalStore):
    """Legacy retrieval store reading directly from demos_dir/*/processed_demo.npz."""

    def __init__(self, demos_dir: str | Path, knn_k: int):
        demos_dir = Path(demos_dir)
        self._demos_dir = demos_dir
        if not demos_dir.exists():
            raise FileNotFoundError(f"demos_dir does not exist: {demos_dir}")

        logger.info("LegacyLiberoDemoStore: loading demos from %s", demos_dir)
        self._demos = {
            demo_idx: np.load(demos_dir / folder / "processed_demo.npz", allow_pickle=True)
            for demo_idx, folder in enumerate(os.listdir(demos_dir))
            if (demos_dir / folder).is_dir()
        }
        if not self._demos:
            raise RuntimeError(f"no processed_demo.npz found under {demos_dir}")

        self._all_indices = np.array(
            [
                (ep_idx, step_idx)
                for ep_idx in list(self._demos.keys())
                for step_idx in range(self._demos[ep_idx]["actions"].shape[0])
            ],
            dtype=np.int32,
        )
        self._all_embeddings = np.concatenate(
            [np.asarray(self._demos[ep_idx]["base_image_embeddings"], dtype=np.float32) for ep_idx in self._demos],
            axis=0,
        )
        assert self._all_embeddings.shape == (len(self._all_indices), EMBED_DIM), f"{self._all_embeddings.shape=}"

        logger.info("LegacyLiberoDemoStore: building ANN index")
        try:
            from autofaiss import build_index
        except Exception as exc:
            raise RuntimeError(
                "LegacyLiberoDemoStore requires autofaiss. "
                "Install autofaiss or use local_vector_db mode."
            ) from exc
        self._knn_index, _ = build_index(
            embeddings=self._all_embeddings,
            save_on_disk=False,
            min_nearest_neighbors_to_retrieve=knn_k + 5,
            max_index_query_time_ms=10,
            max_index_memory_usage="25G",
            current_memory_available="50G",
            metric_type="l2",
            nb_cores=8,
        )

    def search(self, query_embedding: np.ndarray, k: int) -> list[RetrievalHit]:
        _distances, topk_indices = self._knn_index.search(query_embedding, k)
        raw = topk_indices[0]
        hits: list[RetrievalHit] = []
        for idx in raw:
            if int(idx) < 0:
                continue
            ep_idx, step_idx = self._all_indices[int(idx)]
            hits.append(RetrievalHit(episode_id=int(ep_idx), step_idx=int(step_idx), global_idx=int(idx)))
        if len(hits) < k:
            raise RuntimeError(f"retrieval returned only {len(hits)} hits for k={k}")
        return hits[:k]

    def get_context(self, hit: RetrievalHit, action_horizon: int) -> dict[str, Any]:
        demo = self._demos[hit.episode_id]
        return {
            "state": np.asarray(demo["state"][hit.step_idx]),
            "wrist_image": np.asarray(demo["wrist_image"][hit.step_idx]),
            "base_image": np.asarray(demo["base_image"][hit.step_idx]),
            "actions": _get_action_chunk_libero(np.asarray(demo["actions"]), hit.step_idx, action_horizon),
            "prompt": _to_prompt(demo["prompt"]),
        }

    def get_embedding(self, hit: RetrievalHit) -> np.ndarray:
        return np.asarray(self._all_embeddings[hit.global_idx], dtype=np.float32)


class LocalVectorDBStore(BaseLiberoRetrievalStore):
    """Retrieval store backed by preprocessing/local_vector_db/<team_id> artifacts."""

    def __init__(self, local_db_dir: str | Path, demo_cache_size: int = 32):
        self._db_dir = Path(local_db_dir)
        self._demo_cache_size = demo_cache_size
        self._demo_cache: OrderedDict[int, Any] = OrderedDict()

        summary_path = self._db_dir / "summary.json"
        episodes_path = self._db_dir / "episodes.json"
        vectors_path = self._db_dir / "vectors.npz"
        if not (summary_path.exists() and episodes_path.exists() and vectors_path.exists()):
            raise FileNotFoundError(
                f"local vector db artifacts missing under {self._db_dir} "
                f"(need summary.json, episodes.json, vectors.npz)"
            )

        self._summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self._episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
        vectors = np.load(vectors_path, allow_pickle=True)
        self._embeddings = np.asarray(vectors["embeddings"], dtype=np.float32)
        self._episode_ids = np.asarray(vectors["episode_ids"], dtype=np.int32)
        self._step_indices = np.asarray(vectors["step_indices"], dtype=np.int32)

        if self._embeddings.ndim != 2:
            raise ValueError(f"invalid embeddings shape: {self._embeddings.shape}")
        if self._embeddings.shape[1] != EMBED_DIM:
            raise ValueError(f"invalid embedding dim: expected {EMBED_DIM}, got {self._embeddings.shape[1]}")
        if not (
            len(self._embeddings) == len(self._episode_ids)
            and len(self._episode_ids) == len(self._step_indices)
        ):
            raise ValueError("vectors.npz arrays length mismatch")

        approved_root = self._summary.get("approved_root")
        self._approved_root = Path(approved_root) if isinstance(approved_root, str) and approved_root else None
        if self._approved_root is not None and not self._approved_root.exists():
            logger.warning("approved_root from summary does not exist: %s", self._approved_root)

        self._episode_to_npz = self._build_episode_npz_mapping(self._episodes)

        self._faiss_index = None
        faiss_path = self._db_dir / "index.faiss"
        if faiss_path.exists():
            try:
                import faiss  # type: ignore

                self._faiss_index = faiss.read_index(str(faiss_path))
                logger.info("LocalVectorDBStore: loaded FAISS index %s", faiss_path)
            except Exception as exc:
                logger.warning("LocalVectorDBStore: failed to load FAISS index, fallback to numpy search: %s", exc)

    def _build_episode_npz_mapping(self, rows: list[dict]) -> dict[int, Path]:
        mapping: dict[int, Path] = {}
        for row in rows:
            ep_id = int(row["episode_id"])
            npz_path = self._resolve_episode_npz_path(row)
            if not npz_path.exists():
                raise FileNotFoundError(f"episode npz path not found for episode_id={ep_id}: {npz_path}")
            mapping[ep_id] = npz_path
        return mapping

    def _resolve_episode_npz_path(self, row: dict) -> Path:
        explicit = row.get("processed_demo_path")
        if isinstance(explicit, str) and explicit:
            p = Path(explicit)
            if p.exists():
                return p

        relative_path = row.get("relative_path")
        if isinstance(relative_path, str) and relative_path:
            rel = Path(relative_path)
            if rel.is_absolute():
                candidate = rel / "processed_demo.npz" if rel.is_dir() else rel
                if candidate.exists():
                    return candidate
            if self._approved_root is not None:
                candidate = self._approved_root / rel / "processed_demo.npz"
                if candidate.exists():
                    return candidate
            candidate = self._db_dir / rel / "processed_demo.npz"
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"cannot resolve processed_demo.npz for row={row}")

    def _search_with_numpy(self, query_embedding: np.ndarray, k: int) -> np.ndarray:
        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if len(self._embeddings) < k:
            raise RuntimeError(f"not enough vectors: {len(self._embeddings)} < k={k}")
        dists = np.sum((self._embeddings - q) ** 2, axis=1)
        top_idx = np.argpartition(dists, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(dists[top_idx])]
        return np.asarray(top_idx, dtype=np.int64)

    def search(self, query_embedding: np.ndarray, k: int) -> list[RetrievalHit]:
        if self._faiss_index is not None:
            _distances, ids = self._faiss_index.search(np.asarray(query_embedding, dtype=np.float32), k)
            raw = [int(x) for x in ids[0] if int(x) >= 0]
            top_idx = np.asarray(raw, dtype=np.int64)
            if len(top_idx) < k:
                raise RuntimeError(f"faiss returned only {len(top_idx)} hits for k={k}")
        else:
            top_idx = self._search_with_numpy(query_embedding, k)

        hits: list[RetrievalHit] = []
        for gidx in top_idx[:k]:
            ep_idx = int(self._episode_ids[int(gidx)])
            step_idx = int(self._step_indices[int(gidx)])
            hits.append(RetrievalHit(episode_id=ep_idx, step_idx=step_idx, global_idx=int(gidx)))
        return hits

    def _get_demo(self, episode_id: int):
        if episode_id in self._demo_cache:
            demo = self._demo_cache.pop(episode_id)
            self._demo_cache[episode_id] = demo
            return demo
        npz_path = self._episode_to_npz[episode_id]
        demo = np.load(npz_path, allow_pickle=True)
        self._demo_cache[episode_id] = demo
        while len(self._demo_cache) > self._demo_cache_size:
            self._demo_cache.popitem(last=False)
        return demo

    def get_context(self, hit: RetrievalHit, action_horizon: int) -> dict[str, Any]:
        demo = self._get_demo(hit.episode_id)
        return {
            "state": np.asarray(demo["state"][hit.step_idx]),
            "wrist_image": np.asarray(demo["wrist_image"][hit.step_idx]),
            "base_image": np.asarray(demo["base_image"][hit.step_idx]),
            "actions": _get_action_chunk_libero(np.asarray(demo["actions"]), hit.step_idx, action_horizon),
            "prompt": _to_prompt(demo["prompt"]),
        }

    def get_embedding(self, hit: RetrievalHit) -> np.ndarray:
        return np.asarray(self._embeddings[hit.global_idx], dtype=np.float32)


def build_libero_retrieval_store(demos_dir: str | Path, knn_k: int) -> BaseLiberoRetrievalStore:
    demos_dir = Path(demos_dir)
    if (demos_dir / "summary.json").exists() and (demos_dir / "vectors.npz").exists():
        logger.info("build_libero_retrieval_store: using local_vector_db mode (%s)", demos_dir)
        return LocalVectorDBStore(demos_dir)
    logger.info("build_libero_retrieval_store: using legacy demos mode (%s)", demos_dir)
    return LegacyLiberoDemoStore(demos_dir, knn_k=knn_k)

import argparse
import json
import logging
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import torch

from openpi.policies.utils import EMBED_DIM
from openpi.policies.utils import embed_with_batches
from openpi.policies.utils import load_dinov2
from openpi_client.image_tools import resize_with_pad

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def _read_prompt(demo_dir: Path, prompts: list[str] | None) -> str:
    if prompts:
        return str(np.random.choice(prompts))

    meta_path = demo_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            prompt = meta.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                return prompt.strip()
        except Exception as exc:
            logger.warning("failed to read %s: %s", meta_path, exc)

    task_name = demo_dir.parent.name
    parts = task_name.split("_")
    if len(parts) > 1 and parts[0].count("-") == 2:
        return " ".join(parts[1:])
    return task_name.replace("_", " ")


def _find_traj_file(demo_dir: Path) -> Path:
    for name in ("trajectory.h5", "traj.h5"):
        path = demo_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"trajectory file not found in {demo_dir}")


def _prepare_joint7(joint_positions: np.ndarray) -> np.ndarray:
    arr = np.asarray(joint_positions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out = np.zeros((arr.shape[0], 7), dtype=np.float32)
    copy_cols = min(arr.shape[1], 7)
    out[:, :copy_cols] = arr[:, :copy_cols]
    return out


def _prepare_gripper1(gripper_position: np.ndarray) -> np.ndarray:
    arr = np.asarray(gripper_position, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.shape[1] == 1:
        return arr
    return np.mean(arr, axis=1, keepdims=True)


def _prepare_action7(joint_velocity: np.ndarray, gripper_position: np.ndarray) -> np.ndarray:
    joint = np.asarray(joint_velocity, dtype=np.float32)
    if joint.ndim == 1:
        joint = joint.reshape(-1, 1)
    grip = _prepare_gripper1(gripper_position)
    if joint.shape[0] != grip.shape[0]:
        raise ValueError(f"action time mismatch: {joint.shape[0]} vs {grip.shape[0]}")

    # LIBERO action contract is 7-dim: arm(6) + gripper(1).
    arm6 = np.zeros((joint.shape[0], 6), dtype=np.float32)
    copy_cols = min(joint.shape[1], 6)
    arm6[:, :copy_cols] = joint[:, :copy_cols]
    return np.concatenate([arm6, grip[:, :1]], axis=1)


def _load_keep_mask(traj_h5: h5py.File, num_steps: int) -> np.ndarray:
    try:
        skip = traj_h5["observation"]["timestamp"]["skip_action"][:]
        if skip.shape[0] != num_steps:
            logger.warning("skip_action length mismatch. ignoring skip_action.")
            return np.ones(num_steps, dtype=bool)
        return ~skip.astype(bool)
    except Exception:
        return np.ones(num_steps, dtype=bool)


def _list_image_files(frames_dir: Path) -> list[Path]:
    files = [p for p in frames_dir.iterdir() if p.is_file()]
    return sorted(files)


def _ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        pass
    elif np.issubdtype(arr.dtype, np.floating):
        max_val = float(np.max(arr)) if arr.size else 1.0
        if max_val <= 1.5:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"unsupported image shape: {arr.shape}")
    return arr


def _load_camera_frames(
    demo_dir: Path,
    camera_name: str,
    keep_mask: np.ndarray,
    num_steps_after_mask: int,
    rotate_180: bool,
) -> np.ndarray:
    frames_dir = demo_dir / "recordings" / "frames" / camera_name
    if not frames_dir.exists():
        raise FileNotFoundError(f"missing camera directory: {frames_dir}")

    frame_files = _list_image_files(frames_dir)
    if not frame_files:
        raise FileNotFoundError(f"no frames found in {frames_dir}")

    # If frame count is pre-mask length, apply mask to frame file list.
    if len(frame_files) == keep_mask.shape[0]:
        frame_files = [p for p, keep in zip(frame_files, keep_mask) if keep]
    elif len(frame_files) != num_steps_after_mask:
        raise ValueError(
            f"frame count mismatch for {camera_name}: "
            f"{len(frame_files)} vs expected {num_steps_after_mask} or {keep_mask.shape[0]}"
        )

    frames = np.stack([_ensure_uint8_image(np.array(Image.open(p))) for p in frame_files], axis=0)
    if rotate_180:
        frames = np.ascontiguousarray(frames[:, ::-1, ::-1])
    frames = resize_with_pad(frames, 224, 224)
    if frames.shape != (num_steps_after_mask, 224, 224, 3):
        raise ValueError(f"unexpected resized frame shape: {frames.shape}")
    return frames


def process_demo_folder(
    demo_dir: Path,
    *,
    dinov2,
    prompts: list[str] | None,
    wrist_camera: str,
    base_camera: str,
    rotate_180: bool,
    overwrite: bool,
) -> None:
    out_file = demo_dir / "processed_demo.npz"
    if out_file.exists() and not overwrite:
        logger.info("already processed: %s", demo_dir)
        return

    traj_path = _find_traj_file(demo_dir)
    with h5py.File(traj_path, "r") as traj_h5:
        obs_joint = traj_h5["observation"]["robot_state"]["joint_positions"][:]
        obs_grip = traj_h5["observation"]["robot_state"]["gripper_position"][:]
        act_joint = traj_h5["action"]["joint_velocity"][:]
        act_grip = traj_h5["action"]["gripper_position"][:]

        joint7 = _prepare_joint7(obs_joint)
        grip1 = _prepare_gripper1(obs_grip)
        actions7_full = _prepare_action7(act_joint, act_grip)

        num_steps_full = joint7.shape[0]
        if grip1.shape[0] != num_steps_full or actions7_full.shape[0] != num_steps_full:
            raise ValueError(
                f"time length mismatch in {demo_dir}: "
                f"{joint7.shape[0]=}, {grip1.shape[0]=}, {actions7_full.shape[0]=}"
            )

        keep_mask = _load_keep_mask(traj_h5, num_steps_full)
        state = np.concatenate([joint7, grip1], axis=1)[keep_mask].astype(np.float32)
        actions = actions7_full[keep_mask].astype(np.float32)

    num_steps = state.shape[0]
    if num_steps == 0:
        raise ValueError(f"all steps filtered by skip_action in {demo_dir}")
    if state.shape != (num_steps, 8):
        raise ValueError(f"unexpected state shape: {state.shape}")
    if actions.shape != (num_steps, 7):
        raise ValueError(f"unexpected actions shape: {actions.shape}")

    base_image = _load_camera_frames(demo_dir, base_camera, keep_mask, num_steps, rotate_180)
    wrist_image = _load_camera_frames(demo_dir, wrist_camera, keep_mask, num_steps, rotate_180)
    base_emb = embed_with_batches(base_image, dinov2)
    wrist_emb = embed_with_batches(wrist_image, dinov2)
    if base_emb.shape != (num_steps, EMBED_DIM):
        raise ValueError(f"unexpected base embedding shape: {base_emb.shape}")
    if wrist_emb.shape != (num_steps, EMBED_DIM):
        raise ValueError(f"unexpected wrist embedding shape: {wrist_emb.shape}")

    prompt = _read_prompt(demo_dir, prompts)
    processed_demo = {
        "state": state,
        "actions": actions,
        "base_image": base_image,
        "wrist_image": wrist_image,
        "base_image_embeddings": base_emb,
        "wrist_image_embeddings": wrist_emb,
        "prompt": prompt,
    }
    np.savez(out_file, **processed_demo)

    meta = {
        "source": "robot_raw",
        "trajectory_file": traj_path.name,
        "num_steps": num_steps,
        "state_dim": int(state.shape[1]),
        "action_dim": int(actions.shape[1]),
        "base_camera": base_camera,
        "wrist_camera": wrist_camera,
        "rotate_180": bool(rotate_180),
    }
    (demo_dir / "episode_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("saved: %s", out_file)


def _iter_demo_folders(root: Path) -> list[Path]:
    folders = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "recordings").exists() and any((path / x).exists() for x in ("trajectory.h5", "traj.h5")):
            folders.append(path)
    return folders


def process_group_dir(
    group_dir: Path,
    *,
    dinov2,
    prompts: list[str] | None,
    wrist_camera: str,
    base_camera: str,
    rotate_180: bool,
    overwrite: bool,
) -> None:
    demo_folders = _iter_demo_folders(group_dir)
    logger.info("group=%s demos=%d", group_dir, len(demo_folders))
    for demo_dir in demo_folders:
        process_demo_folder(
            demo_dir,
            dinov2=dinov2,
            prompts=prompts,
            wrist_camera=wrist_camera,
            base_camera=base_camera,
            rotate_180=rotate_180,
            overwrite=overwrite,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Source B preprocessor: robot raw logs -> LIBERO contract processed_demo.npz"
    )
    parser.add_argument("--dir", type=str, default=None, help="single group directory (contains demo folders)")
    parser.add_argument("--dir_of_dirs", type=str, default=None, help="directory containing multiple group directories")
    parser.add_argument("--prompts", nargs="+", type=str, default=None, help="optional prompt pool for --dir mode")
    parser.add_argument("--wrist-camera", type=str, default="hand_camera")
    parser.add_argument("--base-camera", type=str, default="varied_camera_1")
    parser.add_argument("--rotate-180", action="store_true", help="rotate images by 180 degrees before resizing")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.dir is None and args.dir_of_dirs is None:
        raise ValueError("Either --dir or --dir_of_dirs must be provided")

    # DINOv2 can take an xformers path that fails on CPU environments.
    # Keep default behavior on GPU, but force safe CPU path automatically.
    if not torch.cuda.is_available() and os.environ.get("XFORMERS_DISABLED") is None:
        os.environ["XFORMERS_DISABLED"] = "1"
        logger.info("CUDA not available; set XFORMERS_DISABLED=1 for CPU compatibility")

    dinov2 = load_dinov2()
    logger.info("loaded DINOv2 for image embedding")

    current_dir = Path(__file__).resolve().parent
    if args.dir is not None:
        group_dir = (current_dir / args.dir).resolve()
        process_group_dir(
            group_dir,
            dinov2=dinov2,
            prompts=args.prompts,
            wrist_camera=args.wrist_camera,
            base_camera=args.base_camera,
            rotate_180=args.rotate_180,
            overwrite=args.overwrite,
        )
    else:
        root = (current_dir / args.dir_of_dirs).resolve()
        for group in sorted([p for p in root.iterdir() if p.is_dir()]):
            inferred_prompts = [" ".join(group.name.split("_")[1:])] if "_" in group.name else None
            process_group_dir(
                group,
                dinov2=dinov2,
                prompts=inferred_prompts,
                wrist_camera=args.wrist_camera,
                base_camera=args.base_camera,
                rotate_180=args.rotate_180,
                overwrite=args.overwrite,
            )

    print("done!")


if __name__ == "__main__":
    main()

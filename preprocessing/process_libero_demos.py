import argparse
import json
import logging
import os

import numpy as np
from openpi_client.image_tools import resize_with_pad

from openpi.policies.utils import EMBED_DIM
from openpi.policies.utils import embed_with_batches
from openpi.policies.utils import load_dinov2

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

REPO_ID = "physical-intelligence/libero"


def process_libero_demos(output_dir: str):
    """Extract all LIBERO benchmark demos from LeRobot HF dataset into processed_demo.npz format.

    Args:
        output_dir: Output directory (e.g., libero_collected_demos_training or libero_collected_demos).
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # Load DINOv2 for image embedding
    dinov2 = load_dinov2()
    logger.info("Loaded DINOv2 for image embedding")

    # Load the LeRobot dataset
    repo_id = REPO_ID
    logger.info(f"Loading LeRobot dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id, local_files_only=False)

    # Get episode info
    total_episodes = dataset.meta.total_episodes
    logger.info(f"Total episodes in dataset: {total_episodes}")

    # Group episodes by task description
    # LeRobot v2 episodes.jsonl uses "tasks" (list of strings), not "task_index"
    episode_indices = list(range(total_episodes))
    episodes_by_task = {}
    for ep_idx in episode_indices:
        task_description = dataset.meta.episodes[ep_idx]["tasks"][0]

        if task_description not in episodes_by_task:
            episodes_by_task[task_description] = []
        episodes_by_task[task_description].append(ep_idx)

    logger.info(f"Found {len(episodes_by_task)} unique tasks")
    for task_desc, ep_list in episodes_by_task.items():
        logger.info(f"  Task: '{task_desc}' -> {len(ep_list)} episodes")

    # Process each task group
    current_dir = os.path.dirname(os.path.abspath(__file__))

    for task_desc, ep_list in episodes_by_task.items():
        # Create task directory name: replace spaces with underscores
        task_dir_name = task_desc.replace(" ", "_")

        task_output_dir = os.path.join(current_dir, output_dir, task_dir_name)

        os.makedirs(task_output_dir, exist_ok=True)

        for ep_count, ep_idx in enumerate(ep_list):
            ep_output_dir = os.path.join(task_output_dir, f"episode_{ep_idx}")

            if os.path.exists(os.path.join(ep_output_dir, "processed_demo.npz")):
                logger.info(f"Episode {ep_idx} already processed, skipping")
                continue

            os.makedirs(ep_output_dir, exist_ok=True)
            logger.info(f"Processing episode {ep_idx} ({ep_count + 1}/{len(ep_list)}) for task: '{task_desc}'")

            # Get episode data from the dataset
            ep_start = dataset.episode_data_index["from"][ep_idx].item()
            ep_end = dataset.episode_data_index["to"][ep_idx].item()
            num_steps = ep_end - ep_start

            # Extract states, actions, and images
            states = []
            actions_list = []
            base_images = []
            wrist_images = []

            for step_idx in range(ep_start, ep_end):
                sample = dataset[step_idx]

                # State: already 8-dim [eef_pos(3), axis_angle(3), gripper(1), pad(1)]
                # LeRobot v2 uses "state" key (not "observation.state")
                state_val = sample["state"]
                if hasattr(state_val, "numpy"):
                    state_val = state_val.numpy()
                state = np.array(state_val, dtype=np.float32)
                states.append(state)

                # Actions: 7-dim [pos_delta(3), rot_delta(3), gripper(1)]
                # LeRobot v2 uses "actions" key (not "action")
                action_val = sample["actions"]
                if hasattr(action_val, "numpy"):
                    action_val = action_val.numpy()
                action = np.array(action_val, dtype=np.float32)
                actions_list.append(action)

                # Images: base camera (agentview) and wrist camera
                # LeRobot v2 uses "image" and "wrist_image" keys
                # May return PIL Image or torch Tensor
                base_img_val = sample["image"]
                if hasattr(base_img_val, "numpy"):
                    base_img = base_img_val.numpy()
                elif hasattr(base_img_val, "convert"):
                    # PIL Image
                    base_img = np.array(base_img_val)
                else:
                    base_img = np.array(base_img_val)
                if np.issubdtype(base_img.dtype, np.floating):
                    base_img = (255 * base_img).astype(np.uint8)
                if base_img.ndim == 3 and base_img.shape[0] == 3:
                    base_img = np.transpose(base_img, (1, 2, 0))
                base_images.append(base_img)

                wrist_img_val = sample["wrist_image"]
                if hasattr(wrist_img_val, "numpy"):
                    wrist_img = wrist_img_val.numpy()
                elif hasattr(wrist_img_val, "convert"):
                    wrist_img = np.array(wrist_img_val)
                else:
                    wrist_img = np.array(wrist_img_val)
                if np.issubdtype(wrist_img.dtype, np.floating):
                    wrist_img = (255 * wrist_img).astype(np.uint8)
                if wrist_img.ndim == 3 and wrist_img.shape[0] == 3:
                    wrist_img = np.transpose(wrist_img, (1, 2, 0))
                wrist_images.append(wrist_img)

            states = np.stack(states, axis=0)
            actions_arr = np.stack(actions_list, axis=0)
            base_images = np.stack(base_images, axis=0)
            wrist_images = np.stack(wrist_images, axis=0)

            assert states.shape == (num_steps, 8), f"{states.shape=}"
            assert actions_arr.shape == (num_steps, 7), f"{actions_arr.shape=}"

            # Rotate images 180 degrees (LIBERO convention) and resize to 224x224
            base_images = np.ascontiguousarray(base_images[:, ::-1, ::-1])
            wrist_images = np.ascontiguousarray(wrist_images[:, ::-1, ::-1])

            base_images = resize_with_pad(base_images, 224, 224)
            wrist_images = resize_with_pad(wrist_images, 224, 224)

            assert base_images.shape == (num_steps, 224, 224, 3) and base_images.dtype == np.uint8
            assert wrist_images.shape == (num_steps, 224, 224, 3) and wrist_images.dtype == np.uint8

            # Compute DINOv2 embeddings
            base_embeddings = embed_with_batches(base_images, dinov2)
            wrist_embeddings = embed_with_batches(wrist_images, dinov2)
            assert base_embeddings.shape == (num_steps, EMBED_DIM), f"{base_embeddings.shape=}"
            assert wrist_embeddings.shape == (num_steps, EMBED_DIM), f"{wrist_embeddings.shape=}"

            # Save processed demo
            processed_demo = {
                "state": states,
                "actions": actions_arr,
                "base_image": base_images,
                "wrist_image": wrist_images,
                "base_image_embeddings": base_embeddings,
                "wrist_image_embeddings": wrist_embeddings,
                "prompt": task_desc,
            }
            np.savez(os.path.join(ep_output_dir, "processed_demo.npz"), **processed_demo)

            # Save episode metadata
            episode_meta = {
                "source_repo": repo_id,
                "source_episode_index": ep_idx,
                "task_description": task_desc,
                "num_steps": num_steps,
                "state_dim": int(states.shape[1]),
                "action_dim": int(actions_arr.shape[1]),
                "image_size": [int(base_images.shape[1]), int(base_images.shape[2])],
            }
            with open(os.path.join(ep_output_dir, "episode_meta.json"), "w") as f:
                json.dump(episode_meta, f, indent=2)

            logger.info(f"Saved processed demo for episode {ep_idx}")

    logger.info("Done processing all LIBERO tasks!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory (e.g., libero_collected_demos_training or libero_collected_demos)",
    )
    args = parser.parse_args()

    process_libero_demos(args.output_dir)

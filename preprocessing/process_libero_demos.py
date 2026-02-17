import numpy as np
import os
import argparse
from openpi.policies.utils import embed_with_batches, load_dinov2, EMBED_DIM
from openpi_client.image_tools import resize_with_pad
import logging

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# LIBERO task suite names and their corresponding HuggingFace dataset repo IDs
TASK_SUITE_TO_REPO = {
    "libero_spatial": "physical-intelligence/libero",
    "libero_object": "physical-intelligence/libero",
    "libero_goal": "physical-intelligence/libero",
    "libero_10": "physical-intelligence/libero",
}


def process_libero_demos(task_suite_name: str, output_dir: str):
    """Extract LIBERO benchmark demos from LeRobot HF dataset into processed_demo.npz format.

    Args:
        task_suite_name: One of libero_spatial, libero_object, libero_goal, libero_10.
        output_dir: Output directory (e.g., libero_collected_demos_training or libero_collected_demos).
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # Load DINOv2 for image embedding
    dinov2 = load_dinov2()
    logger.info("Loaded DINOv2 for image embedding")

    # Load the LeRobot dataset
    repo_id = TASK_SUITE_TO_REPO[task_suite_name]
    logger.info(f"Loading LeRobot dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id, local_files_only=False)

    # Get episode info
    total_episodes = dataset.meta.total_episodes
    logger.info(f"Total episodes in dataset: {total_episodes}")

    # Get task descriptions
    tasks = dataset.meta.tasks
    logger.info(f"Tasks: {tasks}")

    # Filter episodes by task suite name
    # LIBERO dataset in LeRobot format groups episodes by task suite
    # We need to identify which episodes belong to the requested task suite
    episode_indices = list(range(total_episodes))

    # Group episodes by task (language instruction)
    episodes_by_task = {}
    for ep_idx in episode_indices:
        # Get the task index for this episode
        ep_task_index = dataset.meta.episodes[ep_idx]["task_index"]
        task_description = tasks[ep_task_index]

        # Filter by task suite name prefix in the task description
        # LIBERO tasks in the combined dataset are tagged by suite
        if task_description not in episodes_by_task:
            episodes_by_task[task_description] = []
        episodes_by_task[task_description].append(ep_idx)

    logger.info(f"Found {len(episodes_by_task)} unique tasks")
    for task_desc, ep_list in episodes_by_task.items():
        logger.info(f"  Task: '{task_desc}' -> {len(ep_list)} episodes")

    # Filter tasks that belong to the requested suite
    # Use LIBERO benchmark to get the task names for the requested suite
    try:
        from libero.libero import benchmark

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        suite_task_names = set()
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            suite_task_names.add(task.language)
        logger.info(f"Suite '{task_suite_name}' has {len(suite_task_names)} tasks: {suite_task_names}")

        # Filter episodes_by_task to only include tasks in the suite
        filtered_episodes_by_task = {}
        for task_desc, ep_list in episodes_by_task.items():
            if task_desc in suite_task_names:
                filtered_episodes_by_task[task_desc] = ep_list
        episodes_by_task = filtered_episodes_by_task
        logger.info(f"After filtering by suite, {len(episodes_by_task)} tasks remain")
    except ImportError:
        logger.warning("LIBERO benchmark not available, using all tasks from the dataset")

    # Process each task group
    current_dir = os.path.dirname(os.path.abspath(__file__))

    for task_desc, ep_list in episodes_by_task.items():
        # Create task directory name: replace spaces with underscores
        task_dir_name = task_desc.replace(" ", "_")

        if output_dir in ("libero_collected_demos_training",):
            # For training data: group by task suite
            task_output_dir = os.path.join(current_dir, output_dir, task_suite_name)
        else:
            # For test data: use task_suite_name_task_name format
            task_output_dir = os.path.join(current_dir, output_dir, f"{task_suite_name}_{task_dir_name}")

        os.makedirs(task_output_dir, exist_ok=True)

        for ep_count, ep_idx in enumerate(ep_list):
            if output_dir in ("libero_collected_demos_training",):
                ep_output_dir = os.path.join(task_output_dir, f"episode_{ep_idx}")
            else:
                ep_output_dir = os.path.join(task_output_dir, f"demo_{ep_count}")

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
                state = np.array(sample["observation.state"], dtype=np.float32)
                states.append(state)

                # Actions: 7-dim [pos_delta(3), rot_delta(3), gripper(1)]
                action = np.array(sample["action"], dtype=np.float32)
                actions_list.append(action)

                # Images: base camera (agentview) and wrist camera
                # LeRobot stores images as float32 (C, H, W), need to convert to uint8 (H, W, C)
                base_img = np.array(sample["observation.images.image"])
                if np.issubdtype(base_img.dtype, np.floating):
                    base_img = (255 * base_img).astype(np.uint8)
                if base_img.shape[0] == 3:
                    base_img = np.transpose(base_img, (1, 2, 0))
                base_images.append(base_img)

                wrist_img = np.array(sample["observation.images.wrist_image"])
                if np.issubdtype(wrist_img.dtype, np.floating):
                    wrist_img = (255 * wrist_img).astype(np.uint8)
                if wrist_img.shape[0] == 3:
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
            logger.info(f"Saved processed demo for episode {ep_idx}")

    logger.info(f"Done processing {task_suite_name}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_suite_name",
        type=str,
        required=True,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        help="LIBERO task suite name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory (e.g., libero_collected_demos_training or libero_collected_demos)",
    )
    args = parser.parse_args()

    process_libero_demos(args.task_suite_name, args.output_dir)

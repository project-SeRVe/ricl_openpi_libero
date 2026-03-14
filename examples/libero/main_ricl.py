"""RICL evaluation script for LIBERO benchmark.

This script evaluates a RICL policy served via WebSocket on LIBERO tasks.
The key difference from main.py is the use of query_ prefixed keys and
retrieval of query_actions from the RICL policy server.
"""

import collections
import dataclasses
from datetime import datetime
import json
import logging
import math
import pathlib
import re

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_name: str = ""  # Required single-task target; accepts spaces or underscores.
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero_ricl/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero_ricl(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not args.task_name.strip():
        raise ValueError("--task-name must be provided.")

    # Initialize LIBERO task
    benchmark_dict = benchmark.get_benchmark_dict()
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    selected_tasks = [_resolve_single_task(benchmark_dict, args.task_name)]
    logging.info("Evaluating task: %s", selected_tasks[0]["task"].language)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_results = []
    for selected_task in tqdm.tqdm(selected_tasks):
        suite_name = selected_task["suite_name"]
        task_id = selected_task["task_id"]
        task_suite = selected_task["task_suite"]
        task = selected_task["task"]
        initial_states = task_suite.get_task_init_states(task_id)
        max_steps = _max_steps_for_suite(suite_name)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Create task segment for prefix (used by RICL policy for obs logging)
        task_segment = task_description.replace(" ", "_")

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict with RICL query_ prefix
                        element = {
                            "query_base_image": img,
                            "query_wrist_image": wrist_img,
                            "query_state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "query_prompt": str(task_description),
                            "prefix": task_segment,
                        }

                        # Query RICL model to get action
                        action_chunk = client.infer(element)["query_actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        task_success_rate = float(task_successes) / float(task_episodes)
        total_success_rate = float(total_successes) / float(total_episodes)
        logging.info(f"Current task success rate: {task_success_rate}")
        logging.info(f"Current total success rate: {total_success_rate}")
        task_results.append(
            {
                "task_id": task_id,
                "task_description": task_description,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": task_success_rate,
            }
        )

    total_success_rate = float(total_successes) / float(total_episodes)
    logging.info(f"Total success rate: {total_success_rate}")
    logging.info(f"Total episodes: {total_episodes}")

    results = {
        "run_timestamp": run_timestamp,
        "task_name": args.task_name,
        "num_trials_per_task": args.num_trials_per_task,
        "seed": args.seed,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": total_success_rate,
        "tasks": task_results,
    }
    results_path = pathlib.Path(args.video_out_path) / f"results_{run_timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved evaluation results to {results_path}")


def _normalize_task_name(task_name: str) -> str:
    return re.sub(r"[\s_]+", "_", task_name.strip().lower())


def _max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _resolve_single_task(benchmark_dict, task_name: str) -> dict:
    candidate_suites = [
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
        "libero_90",
    ]
    normalized_target = _normalize_task_name(task_name)
    matches = []
    for suite_name in candidate_suites:
        task_suite = benchmark_dict[suite_name]()
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            if _normalize_task_name(task.language) == normalized_target:
                matches.append(
                    {
                        "suite_name": suite_name,
                        "task_id": task_id,
                        "task_suite": task_suite,
                        "task": task,
                    }
                )

    if not matches:
        raise ValueError(f"Task {task_name!r} was not found in LIBERO.")
    if len(matches) > 1:
        raise ValueError(
            f"Task {task_name!r} matched multiple LIBERO tasks: "
            f"{[(match['suite_name'], match['task'].language) for match in matches]}. "
            "Disambiguate by renaming the task target more specifically in the evaluator."
        )
    return matches[0]


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero_ricl)

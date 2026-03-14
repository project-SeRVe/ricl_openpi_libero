import argparse
from datetime import datetime
import json
import pathlib
import shlex
import socket
import subprocess
import sys
import time
import os


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _task_dirs(demos_root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path for path in demos_root.iterdir() if path.is_dir())


def _normalize_task_name(task_name: str) -> str:
    return task_name.strip().lower().replace(" ", "_")


def _build_server_command(
    *,
    policy_config: str,
    checkpoint_dir: pathlib.Path,
    task_dir: pathlib.Path,
    ricl_env: str,
    port: int,
) -> list[str]:
    return [
        "uv",
        "run",
        "--no-sync",
        "scripts/serve_policy_ricl.py",
        f"--port={port}",
        "policy:checkpoint",
        f"--policy.config={policy_config}",
        f"--policy.dir={checkpoint_dir}",
        f"--policy.demos-dir={task_dir}",
        f"--policy.ricl-env={ricl_env}",
    ]


def _build_eval_command(
    *,
    task_name: str,
    host: str,
    port: int,
    num_trials_per_task: int,
    video_out_path: pathlib.Path,
) -> list[str]:
    return [
        "uv",
        "run",
        "--no-sync",
        "examples/libero/main_ricl.py",
        f"--host={host}",
        f"--port={port}",
        f"--task-name={task_name}",
        f"--num-trials-per-task={num_trials_per_task}",
        f"--video-out-path={video_out_path}",
    ]


def _build_eval_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [
        str(REPO_ROOT),
        str(REPO_ROOT / "packages/openpi-client/src"),
        str(REPO_ROOT / "third_party/libero"),
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["LIBERO_CONFIG_PATH"] = str(_ensure_libero_config_dir())
    return env


def _ensure_libero_config_dir() -> pathlib.Path:
    config_dir = REPO_ROOT / ".libero"
    config_dir.mkdir(parents=True, exist_ok=True)

    benchmark_root = REPO_ROOT / "third_party/libero/libero/libero"
    datasets_root = REPO_ROOT / "third_party/libero/libero/datasets"
    config_path = config_dir / "config.yaml"

    config_path.write_text(
        "\n".join(
            [
                f"benchmark_root: {benchmark_root}",
                f"bddl_files: {benchmark_root / 'bddl_files'}",
                f"init_states: {benchmark_root / 'init_files'}",
                f"datasets: {datasets_root}",
                f"assets: {benchmark_root / 'assets'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_dir


def _wait_for_port(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(1.0)
    raise TimeoutError(f"server on {host}:{port} did not become ready within {timeout_s} seconds")


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _run_for_task(
    *,
    task_dir: pathlib.Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    task_name = task_dir.name
    video_out_path = (args.video_out_root / task_name).resolve()
    video_out_path.mkdir(parents=True, exist_ok=True)

    server_cmd = _build_server_command(
        policy_config=args.policy_config,
        checkpoint_dir=args.checkpoint_dir,
        task_dir=task_dir,
        ricl_env=args.ricl_env,
        port=args.port,
    )
    eval_cmd = _build_eval_command(
        task_name=task_name,
        host=args.host,
        port=args.port,
        num_trials_per_task=args.num_trials_per_task,
        video_out_path=video_out_path,
    )

    print(f"[task] {task_name}")
    print(f"[server] {' '.join(shlex.quote(part) for part in server_cmd)}")
    print(f"[eval] {' '.join(shlex.quote(part) for part in eval_cmd)}")

    if args.print_only:
        return {
            "task_name": task_name,
            "server_command": server_cmd,
            "eval_command": eval_cmd,
            "status": "print_only",
        }

    started_at = datetime.now().isoformat()
    server_process = subprocess.Popen(server_cmd, cwd=REPO_ROOT)
    try:
        _wait_for_port(args.host, args.port, args.server_startup_timeout)
        eval_result = subprocess.run(eval_cmd, cwd=REPO_ROOT, env=_build_eval_env(), check=False)
    finally:
        _terminate_process(server_process)

    status = "ok" if eval_result.returncode == 0 else "eval_failed"
    return {
        "task_name": task_name,
        "video_out_path": str(video_out_path),
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(),
        "eval_returncode": eval_result.returncode,
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run RICL LIBERO evaluation sequentially for each task folder under preprocessing/libero_collected_demos."
    )
    parser.add_argument(
        "--demos-root",
        type=pathlib.Path,
        default=pathlib.Path("preprocessing/libero_collected_demos"),
        help="Root directory containing per-task LIBERO retrieval demo folders.",
    )
    parser.add_argument(
        "--policy-config",
        default="pi0_fast_libero_ricl",
        help="Policy config name passed to scripts/serve_policy_ricl.py.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=pathlib.Path,
        default=pathlib.Path("checkpoints/pi0_fast_libero_ricl/priming/3600"),
        help="Checkpoint directory to serve.",
    )
    parser.add_argument(
        "--ricl-env",
        default="libero",
        choices=["libero", "droid"],
        help="RICL environment mode passed to scripts/serve_policy_ricl.py.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host used by the eval client to connect to the server.")
    parser.add_argument("--port", type=int, default=8000, help="Port used by the server and eval client.")
    parser.add_argument(
        "--num-trials-per-task",
        type=int,
        default=50,
        help="Number of rollouts passed to examples/libero/main_ricl.py for each task.",
    )
    parser.add_argument(
        "--video-out-root",
        type=pathlib.Path,
        default=pathlib.Path("examples/libero/data/libero_ricl/batch_eval"),
        help="Root directory where per-task videos and result JSON files will be stored.",
    )
    parser.add_argument(
        "--server-startup-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for the policy server to start before failing a task.",
    )
    parser.add_argument(
        "--task",
        help="If provided, only run the task directory with this exact name.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the per-task server and eval commands without executing them.",
    )
    args = parser.parse_args()

    args.demos_root = args.demos_root.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.video_out_root = args.video_out_root.resolve()

    if not args.demos_root.exists():
        raise FileNotFoundError(f"demos root does not exist: {args.demos_root}")
    if not args.checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint dir does not exist: {args.checkpoint_dir}")

    task_dirs = _task_dirs(args.demos_root)
    if not task_dirs:
        raise ValueError(f"no task directories found under {args.demos_root}")

    if args.task is not None:
        task_dirs = [task_dir for task_dir in task_dirs if task_dir.name == args.task]
        if not task_dirs:
            raise ValueError(f"task {args.task!r} not found under {args.demos_root}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "run_timestamp": run_timestamp,
        "checkpoint_dir": str(args.checkpoint_dir),
        "demos_root": str(args.demos_root),
        "num_trials_per_task": args.num_trials_per_task,
        "tasks": [],
    }

    for task_dir in task_dirs:
        result = _run_for_task(
            task_dir=task_dir,
            args=args,
        )
        summary["tasks"].append(result)

    summary_path = args.video_out_root / f"batch_results_{run_timestamp}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] saved batch results to {summary_path}")

    failed = [task for task in summary["tasks"] if task.get("status") not in {"ok", "print_only"}]
    return 1 if failed and not args.print_only else 0


if __name__ == "__main__":
    sys.exit(main())

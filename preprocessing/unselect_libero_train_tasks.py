import argparse
import random
import shutil
from pathlib import Path


def _list_task_dirs(directory: Path) -> list[Path]:
    return sorted([path for path in directory.iterdir() if path.is_dir()])


def unselect_and_move_train_tasks(
    source_dir: Path,
    target_dir: Path,
    num_tasks: int,
    shuffle: bool,
    seed: int,
    dry_run: bool,
) -> tuple[list[Path], list[Path]]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    source_tasks = _list_task_dirs(source_dir)
    if not source_tasks:
        raise ValueError(f"No task directories found in source: {source_dir}")
    if num_tasks < 1:
        raise ValueError(f"num_tasks must be >= 1, got {num_tasks}")
    if num_tasks > len(source_tasks):
        raise ValueError(
            f"Requested {num_tasks} tasks, but source has only {len(source_tasks)} task directories in {source_dir}"
        )

    selected = list(source_tasks)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected)
    selected = selected[:num_tasks]

    target_dir.mkdir(parents=True, exist_ok=True)
    for src_path in selected:
        dst_path = target_dir / src_path.name
        if dst_path.exists():
            raise FileExistsError(f"Target task directory already exists: {dst_path}")

    if not dry_run:
        for src_path in selected:
            shutil.move(str(src_path), str(target_dir / src_path.name))

    remaining = _list_task_dirs(source_dir)
    return selected, remaining


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reverse selection: move N LIBERO task folders from training back to the original source directory."
        )
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="libero_collected_demos_training",
        help="Source folder under preprocessing that contains selected training task folders.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="libero_collected_demos",
        help="Target folder under preprocessing where task folders are moved back.",
    )
    parser.add_argument("--num_tasks", type=int, required=True, help="Number of task folders to move back.")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle task folder order before selecting --num_tasks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used only when --shuffle is set.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print selected tasks without moving directories.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    source_dir = script_dir / args.source_dir
    target_dir = script_dir / args.target_dir

    selected, remaining = unselect_and_move_train_tasks(
        source_dir=source_dir,
        target_dir=target_dir,
        num_tasks=args.num_tasks,
        shuffle=args.shuffle,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    print(f"source_dir={source_dir}")
    print(f"target_dir={target_dir}")
    print(f"requested_num_tasks={args.num_tasks}")
    print(f"shuffle={args.shuffle}, seed={args.seed}")
    print(f"dry_run={args.dry_run}")
    print(f"selected_tasks={len(selected)}")
    for task in selected:
        print(f"  - {task.name}")
    print(f"remaining_tasks_in_source={len(remaining)}")


if __name__ == "__main__":
    main()

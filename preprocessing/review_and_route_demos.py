import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def _find_episode_dirs(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.rglob("processed_demo.npz")})


def _read_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _move_episode(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"destination already exists: {dst}")
        if not dry_run:
            shutil.rmtree(dst)
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _append_log(log_path: Path, row: dict, dry_run: bool) -> None:
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _list_result_artifacts(result_dir: Path) -> list[Path]:
    if not result_dir.exists():
        return []
    exts = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".gif",
        ".mp4",
        ".webm",
        ".avi",
        ".mov",
        ".json",
    }
    return sorted([p for p in result_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def _run_inference_command(
    template: str,
    *,
    episode_dir: Path,
    rel_path: Path,
    result_dir: Path,
    dry_run: bool,
) -> tuple[int, str]:
    cmd = (
        template.replace("{episode_dir}", str(episode_dir))
        .replace("{rel_path}", str(rel_path))
        .replace("{result_dir}", str(result_dir))
    )
    if dry_run:
        return 0, cmd
    proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return proc.returncode, cmd


def _run_reindex_command(
    *,
    preprocessing_dir: Path,
    approved_root: Path,
    output_root: Path,
    team_id: str,
    write_faiss: bool,
    dry_run: bool,
) -> tuple[int, str]:
    cmd = [
        "uv",
        "run",
        "build_local_vector_db.py",
        "--approved-root",
        str(approved_root),
        "--output-root",
        str(output_root),
        "--team-id",
        team_id,
        "--overwrite",
    ]
    if write_faiss:
        cmd.append("--write-faiss")
    cmd_str = shlex.join(cmd)
    if dry_run:
        return 0, cmd_str
    proc = subprocess.run(cmd, cwd=preprocessing_dir, text=True, capture_output=True)
    if proc.returncode != 0:
        logger.error("reindex failed: rc=%s", proc.returncode)
        if proc.stdout:
            logger.error("reindex stdout:\n%s", proc.stdout)
        if proc.stderr:
            logger.error("reindex stderr:\n%s", proc.stderr)
    return proc.returncode, cmd_str


def _build_row(
    reviewer: str,
    decision: str,
    reason: str,
    source: Path,
    target: Path,
    rel_path: Path,
    episode_meta: dict | None,
    result_dir: Path | None,
    result_artifact_count: int,
    inference_command: str | None,
    inference_returncode: int | None,
    reindex_command: str | None,
    reindex_returncode: int | None,
) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": reviewer,
        "decision": decision,
        "reason": reason,
        "source": str(source),
        "target": str(target),
        "relative_episode_path": str(rel_path),
        "episode_meta": episode_meta,
        "result_dir": str(result_dir) if result_dir is not None else None,
        "result_artifact_count": result_artifact_count,
        "inference_command": inference_command,
        "inference_returncode": inference_returncode,
        "reindex_command": reindex_command,
        "reindex_returncode": reindex_returncode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual O/X review loop with inference result-path support.")
    parser.add_argument("--pending-root", type=str, default="runtime_demos/pending")
    parser.add_argument("--approved-root", type=str, default="runtime_demos/approved")
    parser.add_argument("--rejected-root", type=str, default="runtime_demos/rejected")
    parser.add_argument(
        "--results-root",
        type=str,
        default="runtime_logs/inference",
        help="root directory where inference artifacts are stored by relative episode path",
    )
    parser.add_argument("--log-path", type=str, default="runtime_demos/review_log.jsonl")
    parser.add_argument("--reviewer", type=str, default=os.environ.get("USER", "unknown"))
    parser.add_argument("--reason-required", action="store_true")
    parser.add_argument(
        "--require-results",
        action="store_true",
        help="if set, do not allow O/X when no inference artifacts are found",
    )
    parser.add_argument(
        "--run-inference-cmd",
        type=str,
        default=None,
        help=(
            "optional command template executed per episode before O/X. "
            "supported placeholders: {episode_dir}, {rel_path}, {result_dir}"
        ),
    )
    parser.add_argument("--show-artifacts-limit", type=int, default=8)
    parser.add_argument(
        "--reindex-on-approve",
        action="store_true",
        help="rebuild local vector DB immediately after each approved episode",
    )
    parser.add_argument(
        "--reindex-team-id",
        type=str,
        default=None,
        help="team id for local vector DB output (required if --reindex-on-approve)",
    )
    parser.add_argument(
        "--reindex-output-root",
        type=str,
        default="local_vector_db",
        help="output root for local vector DB artifacts",
    )
    parser.add_argument("--reindex-write-faiss", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="optional max number of pending episodes to review")
    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    pending_root = (current_dir / args.pending_root).resolve()
    approved_root = (current_dir / args.approved_root).resolve()
    rejected_root = (current_dir / args.rejected_root).resolve()
    results_root = (current_dir / args.results_root).resolve()
    log_path = (current_dir / args.log_path).resolve()
    reindex_output_root = (current_dir / args.reindex_output_root).resolve()

    if not pending_root.exists():
        raise FileNotFoundError(f"pending root does not exist: {pending_root}")
    if args.reindex_on_approve and not args.reindex_team_id:
        raise ValueError("--reindex-team-id is required when --reindex-on-approve is set")

    episode_dirs = _find_episode_dirs(pending_root)
    if args.limit is not None:
        episode_dirs = episode_dirs[: args.limit]

    logger.info("pending episodes found: %d", len(episode_dirs))
    if not episode_dirs:
        print("No pending episodes to review.")
        return

    print("Decision keys: [o] approve, [x] reject, [s] skip, [q] quit")
    print(f"reviewer={args.reviewer} dry_run={args.dry_run}")
    if args.run_inference_cmd:
        print("inference mode: enabled (--run-inference-cmd)")
    else:
        print("inference mode: disabled (will only inspect existing --results-root)")

    for idx, episode_dir in enumerate(episode_dirs, start=1):
        rel_path = episode_dir.relative_to(pending_root)
        meta = _read_optional_json(episode_dir / "episode_meta.json")
        num_steps = meta.get("num_steps") if isinstance(meta, dict) else None
        prompt = None
        if isinstance(meta, dict):
            prompt = meta.get("task_description") or meta.get("prompt")

        print()
        print(f"[{idx}/{len(episode_dirs)}] {rel_path}")
        if num_steps is not None:
            print(f"  num_steps={num_steps}")
        if prompt:
            print(f"  prompt={prompt}")

        result_dir = results_root / rel_path
        inference_returncode = None
        inference_command = None
        if args.run_inference_cmd:
            inference_returncode, inference_command = _run_inference_command(
                args.run_inference_cmd,
                episode_dir=episode_dir,
                rel_path=rel_path,
                result_dir=result_dir,
                dry_run=args.dry_run,
            )
            print(f"  inference_cmd={inference_command}")
            print(f"  inference_returncode={inference_returncode}")
            if inference_returncode != 0:
                print("  warning: inference command returned non-zero")

        artifacts = _list_result_artifacts(result_dir)
        print(f"  results_dir={result_dir}")
        print(f"  result_artifacts={len(artifacts)}")
        for item in artifacts[: args.show_artifacts_limit]:
            print(f"    - {item}")
        if len(artifacts) > args.show_artifacts_limit:
            print(f"    ... and {len(artifacts) - args.show_artifacts_limit} more")

        while True:
            raw = input("decision(o/x/s/q): ").strip().lower()
            if raw not in {"o", "x", "s", "q"}:
                print("invalid input. use one of: o, x, s, q")
                continue
            if raw in {"s", "q"}:
                break
            if args.require_results and len(artifacts) == 0:
                print("no inference results found; use [s] skip or provide --run-inference-cmd / --results-root")
                continue
            decision = "approved" if raw == "o" else "rejected"
            reason = input("reason(optional): ").strip()
            if args.reason_required and not reason:
                print("reason is required")
                continue

            target_root = approved_root if decision == "approved" else rejected_root
            target_dir = target_root / rel_path
            _move_episode(episode_dir, target_dir, overwrite=args.overwrite, dry_run=args.dry_run)
            reindex_command = None
            reindex_returncode = None
            if decision == "approved" and args.reindex_on_approve:
                reindex_returncode, reindex_command = _run_reindex_command(
                    preprocessing_dir=current_dir,
                    approved_root=approved_root,
                    output_root=reindex_output_root,
                    team_id=args.reindex_team_id,
                    write_faiss=args.reindex_write_faiss,
                    dry_run=args.dry_run,
                )
                print(f"  reindex_cmd={reindex_command}")
                print(f"  reindex_returncode={reindex_returncode}")
                if reindex_returncode != 0:
                    print("  warning: reindex command returned non-zero")
            row = _build_row(
                reviewer=args.reviewer,
                decision=decision,
                reason=reason,
                source=episode_dir,
                target=target_dir,
                rel_path=rel_path,
                episode_meta=meta,
                result_dir=result_dir,
                result_artifact_count=len(artifacts),
                inference_command=inference_command,
                inference_returncode=inference_returncode,
                reindex_command=reindex_command,
                reindex_returncode=reindex_returncode,
            )
            _append_log(log_path, row, dry_run=args.dry_run)
            print(f"moved -> {decision}: {target_dir}")
            break

        if raw == "q":
            print("quit requested.")
            break

    print("review done.")


if __name__ == "__main__":
    main()

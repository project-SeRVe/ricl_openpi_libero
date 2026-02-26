from __future__ import annotations

import argparse
import base64
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _read_token(token: str | None, token_file: str | None) -> str | None:
    if token:
        return token.strip()
    if token_file:
        return Path(token_file).read_text(encoding="utf-8").strip()
    env_token = os.getenv("SERVE_BEARER_TOKEN") or os.getenv("SYNC_BEARER_TOKEN")
    if env_token:
        return env_token.strip()
    return None


def _json_request(method: str, url: str, token: str | None, payload=None, timeout: float = 20.0):
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            if not raw:
                return None
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc}") from exc


def _chunk_bytes(raw: bytes, chunk_size: int) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    chunks = []
    for idx, start in enumerate(range(0, len(raw), chunk_size)):
        piece = raw[start : start + chunk_size]
        chunks.append({"chunkIndex": idx, "encryptedBlob": base64.b64encode(piece).decode("ascii")})
    return chunks


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json_dict(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid json dict format: {path}")
    return payload


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _upload_manifest_path(state_root: Path, team_id: str) -> Path:
    return state_root / team_id / "upload_manifest.json"


def _pull_state_path(state_root: Path, team_id: str) -> Path:
    return state_root / team_id / "pull_state.json"


def _pull_docs_registry_path(state_root: Path, team_id: str) -> Path:
    return state_root / team_id / "pull_docs.json"


def _load_upload_manifest(path: Path, team_id: str) -> dict[str, Any]:
    payload = _load_json_dict(path, {"team_id": team_id, "updated_at": _now_iso(), "items": {}})
    payload.setdefault("team_id", team_id)
    payload.setdefault("items", {})
    if not isinstance(payload["items"], dict):
        raise ValueError(f"upload manifest.items must be dict: {path}")
    return payload


def _save_upload_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _now_iso()
    _save_json(path, manifest)


def _load_pull_state(path: Path) -> dict[str, Any]:
    return _load_json_dict(path, {"lastVersion": 0})


def _save_pull_state(path: Path, team_id: str, last_version: int) -> None:
    payload = {"teamId": team_id, "lastVersion": last_version, "updatedAt": _now_iso()}
    _save_json(path, payload)


def _load_pull_docs_registry(path: Path, team_id: str) -> dict[str, Any]:
    payload = _load_json_dict(path, {"team_id": team_id, "updated_at": _now_iso(), "documents": {}})
    payload.setdefault("team_id", team_id)
    payload.setdefault("documents", {})
    if not isinstance(payload["documents"], dict):
        raise ValueError(f"pull docs registry.documents must be dict: {path}")
    return payload


def _save_pull_docs_registry(path: Path, registry: dict[str, Any]) -> None:
    registry["updated_at"] = _now_iso()
    _save_json(path, registry)


def _decode_blob(blob) -> bytes:
    if blob is None:
        return b""
    if isinstance(blob, str):
        return base64.b64decode(blob.encode("ascii"))
    if isinstance(blob, list):
        return bytes(blob)
    raise ValueError(f"unsupported blob type: {type(blob)}")


def _apply_sync_items(
    team_id: str,
    items: list[dict[str, Any]],
    out_dir: Path,
    last_version: int,
) -> tuple[int, dict[str, dict[str, Any]]]:
    team_root = out_dir / team_id
    team_root.mkdir(parents=True, exist_ok=True)

    max_version = last_version
    doc_updates: dict[str, dict[str, Any]] = {}

    for item in items:
        version = int(item.get("version", 0))
        max_version = max(max_version, version)
        document_id = item.get("documentId")
        chunk_index = int(item.get("chunkIndex", 0))
        is_deleted = bool(item.get("deleted", item.get("isDeleted", False)))
        if not document_id:
            continue
        document_id = str(document_id)

        update = doc_updates.setdefault(document_id, {})
        update["last_version"] = version
        update["deleted"] = is_deleted
        update["last_seen_at"] = _now_iso()
        if item.get("fileName"):
            update["file_name"] = str(item.get("fileName"))
        if item.get("totalChunks") is not None:
            with contextlib.suppress(Exception):
                update["total_chunks"] = int(item.get("totalChunks"))

        target = team_root / document_id / f"{chunk_index:08d}.bin"
        if is_deleted:
            if target.exists():
                target.unlink()
            continue

        blob = _decode_blob(item.get("encryptedBlob"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)

    return max_version, doc_updates


def _merge_doc_updates(registry: dict[str, Any], updates: dict[str, dict[str, Any]]) -> None:
    docs = registry.setdefault("documents", {})
    for doc_id, upd in updates.items():
        row = docs.get(doc_id, {})
        row.update(upd)
        docs[doc_id] = row


def _entry_key(task_name: str, data_id: str) -> str:
    return f"{task_name}::{data_id}"


def _resolve_processed_demo_path(approved_root: Path, task_name: str, data_id: str) -> Path:
    task_slug = _normalize_name(task_name)
    data_id = data_id.strip()

    direct_candidates = [
        approved_root / task_name / data_id / "processed_demo.npz",
        approved_root / task_name.replace(" ", "_") / data_id / "processed_demo.npz",
        approved_root / task_slug / data_id / "processed_demo.npz",
        approved_root / data_id / "processed_demo.npz",
    ]
    for p in direct_candidates:
        if p.exists():
            return p

    matches = []
    for npz_path in approved_root.rglob("processed_demo.npz"):
        if npz_path.parent.name != data_id:
            continue
        parent_names = [x.name for x in npz_path.parents]
        normalized = {_normalize_name(x) for x in parent_names}
        if task_slug in normalized:
            matches.append(npz_path)

    if not matches:
        raise FileNotFoundError(
            f"no processed_demo.npz found for task={task_name!r}, data_id={data_id!r} under {approved_root}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            "multiple matching demos found. use --file to disambiguate:\n" + "\n".join([str(x) for x in matches[:20]])
        )
    return matches[0]


def _resolve_dry_run_output(output_arg: str | None, task_name: str, data_id: str) -> Path | None:
    if output_arg is None:
        return None
    out = Path(output_arg)
    if out.suffix.lower() == ".json":
        return out
    safe_task = _normalize_name(task_name)
    return out / f"{safe_task}__{data_id}.push_payload.json"


def _build_upload_payload(file_path: Path, remote_file_name: str, chunk_size: int, encrypted_dek: str | None) -> dict:
    raw = file_path.read_bytes()
    if not raw:
        raise ValueError(f"file is empty: {file_path}")
    return {
        "fileName": remote_file_name,
        "chunks": _chunk_bytes(raw, chunk_size),
        "encryptedDEK": encrypted_dek,
    }


def _cmd_upload(args) -> None:
    state_root = Path(args.state_root)
    manifest_file = _upload_manifest_path(state_root, args.team_id)
    manifest = _load_upload_manifest(manifest_file, args.team_id)

    approved_root = Path(args.approved_root)
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            raise FileNotFoundError(f"--file not found: {file_path}")
    else:
        file_path = _resolve_processed_demo_path(approved_root, args.task_name, args.data_id)

    file_path = file_path.resolve()
    size_bytes = file_path.stat().st_size
    sha256 = _sha256_file(file_path)

    key = _entry_key(args.task_name, args.data_id)
    item = manifest["items"].get(key, {})
    prev_sha = item.get("uploaded_sha256")
    if prev_sha == sha256 and not args.force:
        print(
            f"[SKIP] unchanged task={args.task_name} data_id={args.data_id} "
            f"sha256={sha256[:12]} file={file_path}"
        )
        return

    safe_task = _normalize_name(args.task_name)
    remote_file_name = args.remote_file_name or f"{safe_task}__{args.data_id}__{file_path.name}"
    payload = _build_upload_payload(
        file_path=file_path,
        remote_file_name=remote_file_name,
        chunk_size=args.chunk_bytes,
        encrypted_dek=args.encrypted_dek,
    )

    token = _read_token(args.token, args.token_file)
    dry_output = _resolve_dry_run_output(args.dry_run_output, args.task_name, args.data_id)
    if args.dry_run:
        print(
            f"[DRY-RUN UPLOAD] team={args.team_id} task={args.task_name} data_id={args.data_id} "
            f"chunks={len(payload['chunks'])}"
        )
        if dry_output:
            dry_output.parent.mkdir(parents=True, exist_ok=True)
            dry_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[DRY-RUN UPLOAD] payload saved -> {dry_output}")
        return

    if not token:
        raise ValueError("bearer token is required for upload (set --token/--token-file or env)")

    url = f"{args.base_url.rstrip('/')}/api/teams/{args.team_id}/chunks"
    response = _json_request("POST", url=url, token=token, payload=payload, timeout=args.timeout)
    print(
        f"[UPLOAD] team={args.team_id} task={args.task_name} data_id={args.data_id} "
        f"chunks={len(payload['chunks'])} file={file_path.name}"
    )

    manifest["items"][key] = {
        "task_name": args.task_name,
        "data_id": args.data_id,
        "file_path": str(file_path),
        "file_size_bytes": size_bytes,
        "uploaded_sha256": sha256,
        "chunk_bytes": args.chunk_bytes,
        "chunk_count": len(payload["chunks"]),
        "remote_file_name": remote_file_name,
        "uploaded_at": _now_iso(),
        "last_response": response,
    }
    _save_upload_manifest(manifest_file, manifest)
    print(f"[UPLOAD] manifest updated -> {manifest_file}")


def _chunk_index_from_file(path: Path) -> int | None:
    try:
        return int(path.stem)
    except Exception:
        return None


def _is_contiguous(indices: list[int]) -> bool:
    if not indices:
        return False
    start = min(indices)
    end = max(indices)
    return sorted(indices) == list(range(start, end + 1))


def _guess_default_basename(first_chunk_path: Path) -> str:
    header = first_chunk_path.read_bytes()[:4]
    # npz is a zip file
    if header == b"PK\x03\x04":
        return "processed_demo.npz"
    return "payload.bin"


def _resolve_materialized_target(approved_root: Path, doc_id: str, file_name: str | None, first_chunk_path: Path) -> Path:
    if file_name:
        parts = file_name.split("__", 2)
        if len(parts) == 3:
            task_slug, data_id, base = parts
            return approved_root / task_slug / data_id / base
        return approved_root / "_synced_docs" / doc_id / file_name
    return approved_root / "_synced_docs" / doc_id / _guess_default_basename(first_chunk_path)


def _materialize_documents(
    team_id: str,
    out_dir: Path,
    state_root: Path,
    approved_root: Path,
    *,
    allow_incomplete: bool,
) -> dict[str, Any]:
    team_chunks_root = out_dir / team_id
    docs_registry_path = _pull_docs_registry_path(state_root, team_id)
    registry = _load_pull_docs_registry(docs_registry_path, team_id)
    documents = registry.setdefault("documents", {})

    result = {
        "team_id": team_id,
        "materialized": 0,
        "skipped_incomplete": 0,
        "skipped_empty": 0,
        "deleted_removed": 0,
        "errors": [],
    }

    if not team_chunks_root.exists():
        _save_pull_docs_registry(docs_registry_path, registry)
        return result

    for doc_dir in sorted([p for p in team_chunks_root.iterdir() if p.is_dir()]):
        doc_id = doc_dir.name
        row = documents.get(doc_id, {})

        if bool(row.get("deleted", False)):
            mat_path = row.get("materialized_path")
            if mat_path:
                target = Path(mat_path)
                if target.exists():
                    target.unlink()
                    result["deleted_removed"] += 1
            continue

        chunk_files = sorted([p for p in doc_dir.iterdir() if p.is_file()])
        if not chunk_files:
            result["skipped_empty"] += 1
            continue

        indices = []
        for p in chunk_files:
            idx = _chunk_index_from_file(p)
            if idx is None:
                continue
            indices.append(idx)
        if not indices:
            result["skipped_empty"] += 1
            continue

        contiguous = _is_contiguous(indices)
        total_chunks = row.get("total_chunks")
        has_expected_count = isinstance(total_chunks, int) and total_chunks > 0
        count_matches = (len(indices) == total_chunks) if has_expected_count else True
        if not allow_incomplete and (not contiguous or not count_matches):
            result["skipped_incomplete"] += 1
            continue

        first_chunk = chunk_files[0]
        target = _resolve_materialized_target(
            approved_root=approved_root,
            doc_id=doc_id,
            file_name=row.get("file_name"),
            first_chunk_path=first_chunk,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")

        digest = hashlib.sha256()
        ordered_chunks = sorted(
            chunk_files,
            key=lambda p: _chunk_index_from_file(p) if _chunk_index_from_file(p) is not None else 10**12,
        )
        with tmp.open("wb") as wf:
            for ch in ordered_chunks:
                data = ch.read_bytes()
                wf.write(data)
                digest.update(data)
        built_sha = digest.hexdigest()

        if target.exists() and _sha256_file(target) == built_sha:
            tmp.unlink(missing_ok=True)
        else:
            tmp.replace(target)

        row["materialized_path"] = str(target)
        row["materialized_sha256"] = built_sha
        row["materialized_at"] = _now_iso()
        documents[doc_id] = row
        result["materialized"] += 1

    _save_pull_docs_registry(docs_registry_path, registry)
    return result


def _run_rebuild_local_db(
    *,
    repo_root: Path,
    approved_root: Path,
    output_root: Path,
    team_id: str,
    write_faiss: bool,
) -> tuple[int, str]:
    approved_root_abs = approved_root if approved_root.is_absolute() else (repo_root / approved_root).resolve()
    output_root_abs = output_root if output_root.is_absolute() else (repo_root / output_root).resolve()
    cmd = [
        "uv",
        "run",
        "preprocessing/build_local_vector_db.py",
        "--approved-root",
        str(approved_root_abs),
        "--output-root",
        str(output_root_abs),
        "--team-id",
        team_id,
        "--overwrite",
    ]
    if write_faiss:
        cmd.append("--write-faiss")
    cmd_str = shlex.join(cmd)
    proc = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="")
    return proc.returncode, cmd_str


def _cmd_pull(args) -> None:
    token = _read_token(args.token, args.token_file)
    if args.mock_response is None and not token:
        raise ValueError("bearer token is required for real pull")

    state_root = Path(args.state_root)
    state_file = _pull_state_path(state_root, args.team_id)
    docs_registry_file = _pull_docs_registry_path(state_root, args.team_id)
    docs_registry = _load_pull_docs_registry(docs_registry_file, args.team_id)

    state = _load_pull_state(state_file)
    last_version = int(state.get("lastVersion", 0))

    if args.mock_response:
        items = json.loads(Path(args.mock_response).read_text(encoding="utf-8"))
        print(f"[PULL-MOCK] loaded {len(items)} items from {args.mock_response}")
    else:
        qs = urllib.parse.urlencode({"teamId": args.team_id, "lastVersion": last_version})
        url = f"{args.base_url.rstrip('/')}/api/sync/chunks?{qs}"
        items = _json_request("GET", url=url, token=token, timeout=args.timeout) or []
        print(f"[PULL] fetched {len(items)} changed chunks")

    out_dir = Path(args.out_dir)
    new_last_version, doc_updates = _apply_sync_items(
        team_id=args.team_id,
        items=items,
        out_dir=out_dir,
        last_version=last_version,
    )
    _merge_doc_updates(docs_registry, doc_updates)
    _save_pull_docs_registry(docs_registry_file, docs_registry)
    _save_pull_state(state_file, team_id=args.team_id, last_version=new_last_version)
    print(f"[PULL] applied={len(items)} lastVersion {last_version} -> {new_last_version}")
    print(f"[PULL] state updated -> {state_file}")
    print(f"[PULL] docs registry updated -> {docs_registry_file}")

    if args.materialize:
        materialize_result = _materialize_documents(
            team_id=args.team_id,
            out_dir=out_dir,
            state_root=state_root,
            approved_root=Path(args.materialize_approved_root),
            allow_incomplete=args.materialize_allow_incomplete,
        )
        print("[MATERIALIZE]", json.dumps(materialize_result, ensure_ascii=True))

    if args.rebuild_local_db:
        rebuild_team_id = args.rebuild_local_db_team_id or args.team_id
        if args.rebuild_local_db_approved_root:
            approved_root = Path(args.rebuild_local_db_approved_root)
        elif args.materialize:
            approved_root = Path(args.materialize_approved_root)
        else:
            approved_root = Path("preprocessing/runtime_demos/approved_synced")

        repo_root = Path(__file__).resolve().parents[3]
        rc, cmd_str = _run_rebuild_local_db(
            repo_root=repo_root,
            approved_root=approved_root,
            output_root=Path(args.rebuild_local_db_output_root),
            team_id=rebuild_team_id,
            write_faiss=args.rebuild_local_db_write_faiss,
        )
        print(f"[REINDEX] cmd={cmd_str}")
        print(f"[REINDEX] returncode={rc}")
        if rc != 0:
            raise RuntimeError("local vector DB rebuild failed")


def _cmd_materialize(args) -> None:
    result = _materialize_documents(
        team_id=args.team_id,
        out_dir=Path(args.out_dir),
        state_root=Path(args.state_root),
        approved_root=Path(args.approved_root),
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(result, indent=2))


def _cmd_status(args) -> None:
    state_root = Path(args.state_root)
    upload_manifest_file = _upload_manifest_path(state_root, args.team_id)
    upload_manifest = _load_upload_manifest(upload_manifest_file, args.team_id)
    upload_items = upload_manifest.get("items", {})

    docs_registry_file = _pull_docs_registry_path(state_root, args.team_id)
    docs_registry = _load_pull_docs_registry(docs_registry_file, args.team_id)
    pull_docs = docs_registry.get("documents", {})

    if args.json:
        print(
            json.dumps(
                {
                    "team_id": args.team_id,
                    "upload_manifest_file": str(upload_manifest_file),
                    "upload_manifest": upload_manifest,
                    "pull_docs_registry_file": str(docs_registry_file),
                    "pull_docs_registry": docs_registry,
                },
                indent=2,
            )
        )
        return

    print(f"team_id={args.team_id}")
    print(f"upload_manifest={upload_manifest_file}")
    print(f"upload_updated_at={upload_manifest.get('updated_at')}")
    print(f"uploaded_items={len(upload_items)}")
    for key in sorted(upload_items.keys())[: args.limit]:
        row = upload_items[key]
        print(
            f"- upload {key} sha256={str(row.get('uploaded_sha256', ''))[:12]} "
            f"chunks={row.get('chunk_count')} uploaded_at={row.get('uploaded_at')}"
        )

    print(f"pull_docs_registry={docs_registry_file}")
    print(f"pull_docs_updated_at={docs_registry.get('updated_at')}")
    print(f"pulled_documents={len(pull_docs)}")
    for doc_id in sorted(pull_docs.keys())[: args.limit]:
        row = pull_docs[doc_id]
        print(
            f"- pull doc_id={doc_id} file_name={row.get('file_name')} "
            f"deleted={row.get('deleted')} materialized_path={row.get('materialized_path')}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data",
        description="Team demo data sync CLI (incremental upload + pull + materialize).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_upload = sub.add_parser("upload", help="upload approved demo as chunk payload")
    p_upload.add_argument("team_id", help="<team-name or id>")
    p_upload.add_argument("task_name", help="<task-name>")
    p_upload.add_argument("data_id", help="<data-id>")
    p_upload.add_argument("--approved-root", default="preprocessing/runtime_demos/approved")
    p_upload.add_argument("--file", default=None, help="optional explicit processed_demo.npz path")
    p_upload.add_argument("--base-url", default="http://localhost:8000")
    p_upload.add_argument("--token", default=None)
    p_upload.add_argument("--token-file", default=None)
    p_upload.add_argument("--chunk-bytes", type=int, default=262144)
    p_upload.add_argument("--encrypted-dek", default=None)
    p_upload.add_argument("--remote-file-name", default=None)
    p_upload.add_argument("--state-root", default="preprocessing/.sync_state")
    p_upload.add_argument("--force", action="store_true", help="upload even when file hash is unchanged")
    p_upload.add_argument("--dry-run", action="store_true")
    p_upload.add_argument("--dry-run-output", default=None, help="output .json or directory for payload dump")
    p_upload.add_argument("--timeout", type=float, default=20.0)
    p_upload.set_defaults(func=_cmd_upload)

    p_pull = sub.add_parser("pull", help="pull incremental chunk updates")
    p_pull.add_argument("team_id", help="<team-name or id>")
    p_pull.add_argument("--base-url", default="http://localhost:8000")
    p_pull.add_argument("--token", default=None)
    p_pull.add_argument("--token-file", default=None)
    p_pull.add_argument("--state-root", default="preprocessing/.sync_state")
    p_pull.add_argument("--out-dir", default="preprocessing/.synced_chunks")
    p_pull.add_argument("--mock-response", default=None, help="json file for offline test")
    p_pull.add_argument("--materialize", action="store_true", help="rebuild synced files after pull")
    p_pull.add_argument(
        "--materialize-approved-root",
        default="preprocessing/runtime_demos/approved_synced",
        help="target root for reconstructed synced files",
    )
    p_pull.add_argument("--materialize-allow-incomplete", action="store_true")
    p_pull.add_argument(
        "--rebuild-local-db",
        action="store_true",
        help="rebuild local vector DB after pull/materialize",
    )
    p_pull.add_argument(
        "--rebuild-local-db-team-id",
        default=None,
        help="team id for local_vector_db output (default: same as pull team_id)",
    )
    p_pull.add_argument(
        "--rebuild-local-db-approved-root",
        default=None,
        help="approved root for index rebuild (default: materialize target when --materialize is used)",
    )
    p_pull.add_argument(
        "--rebuild-local-db-output-root",
        default="preprocessing/local_vector_db",
        help="output root for local vector DB artifacts",
    )
    p_pull.add_argument("--rebuild-local-db-write-faiss", action="store_true")
    p_pull.add_argument("--timeout", type=float, default=20.0)
    p_pull.set_defaults(func=_cmd_pull)

    p_materialize = sub.add_parser("materialize", help="materialize pulled chunks into files")
    p_materialize.add_argument("team_id", help="<team-name or id>")
    p_materialize.add_argument("--state-root", default="preprocessing/.sync_state")
    p_materialize.add_argument("--out-dir", default="preprocessing/.synced_chunks")
    p_materialize.add_argument("--approved-root", default="preprocessing/runtime_demos/approved_synced")
    p_materialize.add_argument("--allow-incomplete", action="store_true")
    p_materialize.set_defaults(func=_cmd_materialize)

    p_status = sub.add_parser("status", help="show upload and pull status")
    p_status.add_argument("team_id", help="<team-name or id>")
    p_status.add_argument("--state-root", default="preprocessing/.sync_state")
    p_status.add_argument("--json", action="store_true")
    p_status.add_argument("--limit", type=int, default=20)
    p_status.set_defaults(func=_cmd_status)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

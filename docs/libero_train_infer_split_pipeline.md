# LIBERO Train/Infer Split Pipeline (Current Implementation)

## 1) Scope
This document reflects what is implemented now in `ricl_openpi_libero`.

- Training path: keep existing LIBERO RICL flow.
- Inference/operation path: `pending -> review(O/X) -> approved/rejected -> local_vector_db -> serve`.
- Sync path: chunk upload/pull/materialize CLI is implemented.

Not yet implemented is listed in Section 9.

---

## 2) Canonical Data Contract

### 2.1 `processed_demo.npz` required keys
- `state`
- `actions`
- `base_image`
- `wrist_image`
- `base_image_embeddings`
- `wrist_image_embeddings`
- `prompt`

### 2.2 required shapes
- `state`: `(T, 8)`
- `actions`: `(T, 7)`
- `base_image`: `(T, 224, 224, 3)`
- `wrist_image`: `(T, 224, 224, 3)`

Validator code:
- `preprocessing/validate_processed_demo.py`

---

## 3) Source A / Source B Preprocessing

### 3.1 Source A (LeRobot LIBERO)
- Script: `preprocessing/process_libero_demos.py`
- Input: `physical-intelligence/libero`
- Output layout:
  - `preprocessing/<output_dir>/<task_name>/episode_<episode_id>/processed_demo.npz`

Run example:
```bash
cd preprocessing
uv run process_libero_demos.py --output_dir libero_collected_demos_training
```

### 3.2 Source B (robot raw logs)
- Script: `preprocessing/process_robot_demos_to_libero.py`
- Input contract per episode:
  - `trajectory.h5` or `traj.h5`
  - `recordings/frames/varied_camera_1/*`
  - `recordings/frames/hand_camera/*`
  - optional `meta.json` (prompt)
- Output:
  - `<episode_dir>/processed_demo.npz`
  - `<episode_dir>/episode_meta.json`

Run example:
```bash
cd preprocessing
uv run process_robot_demos_to_libero.py --dir runtime_demos/pending/sample_one
```

---

## 4) Validation + Manual O/X Routing

### 4.1 Validate contract
- Script: `preprocessing/validate_processed_demo.py`

Run example:
```bash
cd preprocessing
uv run validate_processed_demo.py \
  --root runtime_demos/pending \
  --report-json runtime_demos/validation_report.json
```

### 4.2 Review and route
- Script: `preprocessing/review_and_route_demos.py`
- Core behavior:
  - scans `pending` for `processed_demo.npz`
  - prints episode metadata and `results_dir`
  - prints image/video/json artifact list under `results_dir`
  - asks human input: `o/x/s/q`
  - moves episode to `approved` or `rejected`
  - appends `review_log.jsonl`

Run example (short, default paths):
```bash
cd preprocessing
uv run review_and_route_demos.py --require-results
```

Default paths:
- `pending-root`: `runtime_demos/pending`
- `approved-root`: `runtime_demos/approved`
- `rejected-root`: `runtime_demos/rejected`
- `results-root`: `runtime_logs/inference`
- `log-path`: `runtime_demos/review_log.jsonl`

Optional:
- `--run-inference-cmd` is a hook command template, not mandatory.
- placeholders: `{episode_dir}`, `{rel_path}`, `{result_dir}`
- built-in evaluator script is available:
  - `scripts/run_episode_eval.py`

Example (run evaluator + O/X):
```bash
cd preprocessing
uv run review_and_route_demos.py \
  --require-results \
  --run-inference-cmd "uv run ../scripts/run_episode_eval.py --episode {episode_dir} --out {result_dir} --mode infer --host 0.0.0.0 --port 8000 --overwrite"
```

---

## 5) Build Local Vector DB

- Script: `preprocessing/build_local_vector_db.py`
- Input:
  - `runtime_demos/approved/**/processed_demo.npz`
- Output:
  - `preprocessing/local_vector_db/<team_id>/summary.json`
  - `preprocessing/local_vector_db/<team_id>/episodes.json`
  - `preprocessing/local_vector_db/<team_id>/vectors.npz`
  - optional `index.faiss` with `--write-faiss`

Run example:
```bash
cd preprocessing
uv run build_local_vector_db.py \
  --approved-root runtime_demos/approved \
  --output-root local_vector_db \
  --team-id team_a \
  --overwrite
```

---

## 6) Sync CLI (Incremental upload/pull/materialize)

- CLI entrypoint:
  - `uv run data ...`
  - defined in `pyproject.toml` -> `openpi.tools.data_sync_cli:main`
- Script: `src/openpi/tools/data_sync_cli.py`

Commands:
- `data upload <team_id> <task_name> <data_id>`
- `data pull <team_id>`
- `data materialize <team_id>`
- `data status <team_id>`

Current behavior:
- Upload:
  - uploads `processed_demo.npz` as chunk payload
  - keeps hash manifest at `preprocessing/.sync_state/<team>/upload_manifest.json`
  - skips unchanged unless `--force`
- Pull:
  - stores chunks under `preprocessing/.synced_chunks/<team>/<documentId>/`
  - updates pull docs registry
- Materialize:
  - reconstructs file from pulled chunks
  - writes to `preprocessing/runtime_demos/approved_synced/...` (by naming convention)
- Optional rebuild:
  - `data pull` can trigger local vector DB rebuild in the same command via `--rebuild-local-db`
  - rebuild target defaults to pulled team id unless overridden

Run example:
```bash
uv run data upload team_a pick_up_book demo_11 --dry-run
uv run data pull team_a --materialize --materialize-approved-root preprocessing/runtime_demos/approved_synced
uv run data pull team_a --materialize --rebuild-local-db
uv run data status team_a
```

---

## 7) Data Locations (Default + Current Workspace)

### 7.1 Default paths used by scripts
- Pending demos:
  - `preprocessing/runtime_demos/pending`
- Approved demos:
  - `preprocessing/runtime_demos/approved`
- Rejected demos:
  - `preprocessing/runtime_demos/rejected`
- Synced/materialized demos:
  - `preprocessing/runtime_demos/approved_synced`
- Review/inference artifacts:
  - `preprocessing/runtime_logs/inference`
- Local vector DB:
  - `preprocessing/local_vector_db/<team_id>`
- Sync state:
  - `preprocessing/.sync_state/<team_id>`
- Synced chunks:
  - `preprocessing/.synced_chunks/<team_id>/<documentId>/`

### 7.2 Current sample data found in this workspace
- Pending sample:
  - `preprocessing/runtime_demos/pending/sample_one/demo_11/processed_demo.npz`
- Synced approved sample:
  - `preprocessing/runtime_demos/approved_synced/pick_up_book/demo_999/processed_demo.npz`
- Local vector DB samples:
  - `preprocessing/local_vector_db/team_sync/{summary.json,episodes.json,vectors.npz}`
  - `preprocessing/local_vector_db/team_demo/{summary.json,episodes.json,vectors.npz}`

Note:
- `preprocessing/runtime_logs/inference` can be empty until evaluation/review creates artifacts.

---

## 8) Serving Retrieval Runtime

### 8.1 Policy-side retrieval adapter
- `src/openpi/policies/retrieval_store.py`
- `src/openpi/policies/policy.py` (`RiclLiberoPolicy`)

Selection logic:
- if `demos_dir` contains `summary.json` + `vectors.npz`:
  - uses `LocalVectorDBStore`
- otherwise:
  - uses legacy demos-folder store (`processed_demo.npz` folders)

### 8.2 Serving command example
```bash
uv run scripts/serve_policy_ricl.py policy:checkpoint \
  --policy.config=pi0_fast_libero_ricl \
  --policy.dir=checkpoints/pi0_fast_libero_ricl/<EXP>/<STEP> \
  --policy.demos_dir=preprocessing/local_vector_db/team_a \
  --policy.ricl_env=libero \
  --port=8000
```

---

## 9) Not Implemented Yet (Important)

1. Real crypto pipeline is not implemented inside `data_sync_cli.py`.
- Chunk payload uses base64 transport and optional `encryptedDEK` field pass-through.
- No local key generation/decryption logic is currently in this CLI.

2. Vector-delta upload is not implemented.
- Current upload unit is file-based (`processed_demo.npz` chunking).

---

## 10) Minimal End-to-End (Current)

```bash
cd preprocessing

# A or B preprocess
uv run process_robot_demos_to_libero.py --dir runtime_demos/pending/sample_one

# validate
uv run validate_processed_demo.py --root runtime_demos/pending

# human review O/X
uv run review_and_route_demos.py --require-results

# optional: human review O/X + immediate reindex
uv run review_and_route_demos.py \
  --require-results \
  --reindex-on-approve \
  --reindex-team-id team_a

# (repo root) serve with local vector db
cd ..
uv run scripts/serve_policy_ricl.py policy:checkpoint \
  --policy.config=pi0_fast_libero_ricl \
  --policy.dir=checkpoints/pi0_fast_libero_ricl/<EXP>/<STEP> \
  --policy.demos_dir=preprocessing/local_vector_db/team_a \
  --policy.ricl_env=libero \
  --port=8000
```

#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEMOS_ROOT="$REPO_ROOT/preprocessing/libero_collected_demos"
POLICY_CONFIG="pi0_fast_libero_ricl"
CHECKPOINT_DIR="$REPO_ROOT/checkpoints/pi0_fast_libero_ricl/priming/3600"
RICL_ENV="libero"
HOST="127.0.0.1"
PORT="8000"
NUM_TRIALS_PER_TASK="50"
VIDEO_OUT_ROOT="$REPO_ROOT/examples/libero/data/libero_ricl/batch_eval"
SERVER_STARTUP_TIMEOUT="300"
TASK_NAME=""
PRINT_ONLY="0"
SERVER_PYTHON="$REPO_ROOT/.venv/bin/python"
EVAL_PYTHON="$REPO_ROOT/examples/libero/.venv/bin/python"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Run LIBERO RICL serving and evaluation with split Python environments.

Options:
  --demos-root PATH
  --policy-config NAME
  --checkpoint-dir PATH
  --ricl-env {libero|droid}
  --host HOST
  --port PORT
  --num-trials-per-task INT
  --video-out-root PATH
  --server-startup-timeout SECONDS
  --task-name NAME
  --task NAME
  --server-python PATH
  --eval-python PATH
  --print-only
  -h, --help
EOF
}

require_option_value() {
  local option_name="$1"
  local option_value="${2:-}"

  if [[ -z "$option_value" ]]; then
    echo "Missing value for $option_name" >&2
    usage
    exit 1
  fi
}

normalize_task_name() {
  local task_name="$1"
  task_name="${task_name// /_}"
  echo "${task_name,,}"
}

ensure_libero_config() {
  local config_dir="$REPO_ROOT/.libero"
  local benchmark_root="$REPO_ROOT/third_party/libero/libero/libero"
  local datasets_root="$REPO_ROOT/third_party/libero/libero/datasets"

  mkdir -p "$config_dir"
  cat > "$config_dir/config.yaml" <<EOF
benchmark_root: $benchmark_root
bddl_files: $benchmark_root/bddl_files
init_states: $benchmark_root/init_files
datasets: $datasets_root
assets: $benchmark_root/assets
EOF
  echo "$config_dir"
}

wait_for_server() {
  local deadline_seconds="$1"
  local server_pid="$2"
  local start_time
  start_time="$(date +%s)"

  while true; do
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
      echo "Server process exited before becoming ready on $HOST:$PORT" >&2
      return 1
    fi

    if SERVER_HOST="$HOST" SERVER_PORT="$PORT" "$SERVER_PYTHON" -c \
      "from websockets.sync.client import connect; import os; connect(f\"ws://{os.environ['SERVER_HOST']}:{os.environ['SERVER_PORT']}\", open_timeout=1, close_timeout=1).close()" \
      >/dev/null 2>&1; then
      local elapsed_seconds
      elapsed_seconds=$(( "$(date +%s)" - start_time ))
      echo "Server became ready on $HOST:$PORT after ${elapsed_seconds}s" >&2
      return 0
    fi

    if (( "$(date +%s)" - start_time >= deadline_seconds )); then
      echo "Timed out waiting for server on $HOST:$PORT" >&2
      return 1
    fi
    sleep 1
  done
}

build_server_pythonpath() {
  local existing_pythonpath="${PYTHONPATH:-}"
  local base="$REPO_ROOT:$REPO_ROOT/packages/openpi-client/src"
  if [[ -n "$existing_pythonpath" ]]; then
    echo "$base:$existing_pythonpath"
  else
    echo "$base"
  fi
}

build_eval_pythonpath() {
  local existing_pythonpath="${PYTHONPATH:-}"
  local base="$REPO_ROOT:$REPO_ROOT/packages/openpi-client/src:$REPO_ROOT/third_party/libero"
  if [[ -n "$existing_pythonpath" ]]; then
    echo "$base:$existing_pythonpath"
  else
    echo "$base"
  fi
}

resolve_repo_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$REPO_ROOT/$path"
  fi
}

run_for_task() {
  local task_dir="$1"
  local task_name
  local video_out_path
  local libero_config_path
  local server_pid
  local server_pythonpath
  local eval_pythonpath

  task_name="$(basename "$task_dir")"
  video_out_path="$VIDEO_OUT_ROOT/$task_name"
  mkdir -p "$video_out_path"

  libero_config_path="$(ensure_libero_config)"
  server_pythonpath="$(build_server_pythonpath)"
  eval_pythonpath="$(build_eval_pythonpath)"

  local server_cmd=(
    "$SERVER_PYTHON" "$REPO_ROOT/scripts/serve_policy_ricl.py"
    "--port=$PORT"
    "policy:checkpoint"
    "--policy.config=$POLICY_CONFIG"
    "--policy.dir=$CHECKPOINT_DIR"
    "--policy.demos-dir=$task_dir"
    "--policy.ricl-env=$RICL_ENV"
  )

  local eval_cmd=(
    "$EVAL_PYTHON" "$REPO_ROOT/examples/libero/main_ricl.py"
    "--args.host=$HOST"
    "--args.port=$PORT"
    "--args.task-name=$task_name"
    "--args.num-trials-per-task=$NUM_TRIALS_PER_TASK"
    "--args.video-out-path=$video_out_path"
  )

  echo "[task] $task_name"
  echo "[server] ${server_cmd[*]}"
  echo "[eval] ${eval_cmd[*]}"

  if [[ "$PRINT_ONLY" == "1" ]]; then
    return 0
  fi

  PYTHONPATH="$server_pythonpath" "$SERVER_PYTHON" "$REPO_ROOT/scripts/serve_policy_ricl.py" \
    "--port=$PORT" \
    "policy:checkpoint" \
    "--policy.config=$POLICY_CONFIG" \
    "--policy.dir=$CHECKPOINT_DIR" \
    "--policy.demos-dir=$task_dir" \
    "--policy.ricl-env=$RICL_ENV" &
  server_pid=$!

  cleanup() {
    if kill -0 "$server_pid" >/dev/null 2>&1; then
      kill "$server_pid" >/dev/null 2>&1 || true
      wait "$server_pid" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup RETURN

  wait_for_server "$SERVER_STARTUP_TIMEOUT" "$server_pid"

  PYTHONPATH="$eval_pythonpath" LIBERO_CONFIG_PATH="$libero_config_path" \
    "$EVAL_PYTHON" "$REPO_ROOT/examples/libero/main_ricl.py" \
    "--args.host=$HOST" \
    "--args.port=$PORT" \
    "--args.task-name=$task_name" \
    "--args.num-trials-per-task=$NUM_TRIALS_PER_TASK" \
    "--args.video-out-path=$video_out_path"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --demos-root=*)
      DEMOS_ROOT="${1#*=}"
      require_option_value "--demos-root" "$DEMOS_ROOT"
      shift
      ;;
    --demos-root)
      require_option_value "--demos-root" "${2:-}"
      DEMOS_ROOT="$2"
      shift 2
      ;;
    --policy-config=*)
      POLICY_CONFIG="${1#*=}"
      require_option_value "--policy-config" "$POLICY_CONFIG"
      shift
      ;;
    --policy-config)
      require_option_value "--policy-config" "${2:-}"
      POLICY_CONFIG="$2"
      shift 2
      ;;
    --checkpoint-dir=*)
      CHECKPOINT_DIR="${1#*=}"
      require_option_value "--checkpoint-dir" "$CHECKPOINT_DIR"
      shift
      ;;
    --checkpoint-dir)
      require_option_value "--checkpoint-dir" "${2:-}"
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --ricl-env=*)
      RICL_ENV="${1#*=}"
      require_option_value "--ricl-env" "$RICL_ENV"
      shift
      ;;
    --ricl-env)
      require_option_value "--ricl-env" "${2:-}"
      RICL_ENV="$2"
      shift 2
      ;;
    --host=*)
      HOST="${1#*=}"
      require_option_value "--host" "$HOST"
      shift
      ;;
    --host)
      require_option_value "--host" "${2:-}"
      HOST="$2"
      shift 2
      ;;
    --port=*)
      PORT="${1#*=}"
      require_option_value "--port" "$PORT"
      shift
      ;;
    --port)
      require_option_value "--port" "${2:-}"
      PORT="$2"
      shift 2
      ;;
    --num-trials-per-task=*)
      NUM_TRIALS_PER_TASK="${1#*=}"
      require_option_value "--num-trials-per-task" "$NUM_TRIALS_PER_TASK"
      shift
      ;;
    --num-trials-per-task)
      require_option_value "--num-trials-per-task" "${2:-}"
      NUM_TRIALS_PER_TASK="$2"
      shift 2
      ;;
    --video-out-root=*)
      VIDEO_OUT_ROOT="${1#*=}"
      require_option_value "--video-out-root" "$VIDEO_OUT_ROOT"
      shift
      ;;
    --video-out-root)
      require_option_value "--video-out-root" "${2:-}"
      VIDEO_OUT_ROOT="$2"
      shift 2
      ;;
    --server-startup-timeout=*)
      SERVER_STARTUP_TIMEOUT="${1#*=}"
      require_option_value "--server-startup-timeout" "$SERVER_STARTUP_TIMEOUT"
      shift
      ;;
    --server-startup-timeout)
      require_option_value "--server-startup-timeout" "${2:-}"
      SERVER_STARTUP_TIMEOUT="$2"
      shift 2
      ;;
    --task-name=*|--task=*)
      TASK_NAME="${1#*=}"
      require_option_value "--task-name" "$TASK_NAME"
      shift
      ;;
    --task-name|--task)
      require_option_value "--task-name" "${2:-}"
      TASK_NAME="$2"
      shift 2
      ;;
    --server-python=*)
      SERVER_PYTHON="${1#*=}"
      require_option_value "--server-python" "$SERVER_PYTHON"
      shift
      ;;
    --server-python)
      require_option_value "--server-python" "${2:-}"
      SERVER_PYTHON="$2"
      shift 2
      ;;
    --eval-python=*)
      EVAL_PYTHON="${1#*=}"
      require_option_value "--eval-python" "$EVAL_PYTHON"
      shift
      ;;
    --eval-python)
      require_option_value "--eval-python" "${2:-}"
      EVAL_PYTHON="$2"
      shift 2
      ;;
    --print-only)
      PRINT_ONLY="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

DEMOS_ROOT="$(realpath "$DEMOS_ROOT")"
CHECKPOINT_DIR="$(realpath "$CHECKPOINT_DIR")"
VIDEO_OUT_ROOT="$(realpath -m "$VIDEO_OUT_ROOT")"
SERVER_PYTHON="$(resolve_repo_path "$SERVER_PYTHON")"
EVAL_PYTHON="$(resolve_repo_path "$EVAL_PYTHON")"

if [[ ! -d "$DEMOS_ROOT" ]]; then
  echo "Demos root does not exist: $DEMOS_ROOT" >&2
  exit 1
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "Checkpoint dir does not exist: $CHECKPOINT_DIR" >&2
  exit 1
fi

if [[ ! -x "$SERVER_PYTHON" ]]; then
  echo "Server Python is not executable: $SERVER_PYTHON" >&2
  exit 1
fi

if [[ ! -x "$EVAL_PYTHON" ]]; then
  echo "Eval Python is not executable: $EVAL_PYTHON" >&2
  exit 1
fi

mapfile -t ALL_TASK_DIRS < <(find "$DEMOS_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ ${#ALL_TASK_DIRS[@]} -eq 0 ]]; then
  echo "No task directories found under $DEMOS_ROOT" >&2
  exit 1
fi

TASK_DIRS=()
if [[ -n "$TASK_NAME" ]]; then
  normalized_target="$(normalize_task_name "$TASK_NAME")"
  for task_dir in "${ALL_TASK_DIRS[@]}"; do
    if [[ "$(normalize_task_name "$(basename "$task_dir")")" == "$normalized_target" ]]; then
      TASK_DIRS+=("$task_dir")
    fi
  done
  if [[ ${#TASK_DIRS[@]} -eq 0 ]]; then
    echo "Task not found under $DEMOS_ROOT: $TASK_NAME" >&2
    exit 1
  fi
else
  TASK_DIRS=("${ALL_TASK_DIRS[@]}")
fi

mkdir -p "$VIDEO_OUT_ROOT"

for task_dir in "${TASK_DIRS[@]}"; do
  run_for_task "$task_dir"
done

import os
from pathlib import Path


CAPSTONE_CACHE_ROOT = Path("/mnt/disks/sdb/capstone")

_CACHE_ENV_DEFAULTS = {
    "OPENPI_DATA_HOME": CAPSTONE_CACHE_ROOT / "openpi_cache",
    "HF_HOME": CAPSTONE_CACHE_ROOT / "hf_cache",
    "HF_DATASETS_CACHE": CAPSTONE_CACHE_ROOT / "hf_cache" / "datasets",
    "TRANSFORMERS_CACHE": CAPSTONE_CACHE_ROOT / "hf_cache",
    "HUGGINGFACE_HUB_CACHE": CAPSTONE_CACHE_ROOT / "hf_cache",
    "TORCH_HOME": CAPSTONE_CACHE_ROOT / "torch_cache",
    "LEROBOT_HOME": CAPSTONE_CACHE_ROOT / "hf_cache" / "lerobot",
    "JAX_COMPILATION_CACHE_DIR": CAPSTONE_CACHE_ROOT / "jax_cache",
}


def _should_replace_with_capstone_path(current_value: str | None) -> bool:
    if not current_value:
        return True

    current_path = Path(current_value).expanduser()
    return str(current_path).startswith("/mnt/disks/sdb/") and not str(current_path).startswith(
        f"{CAPSTONE_CACHE_ROOT}/"
    )


def configure_project_cache_env() -> None:
    """Apply cache defaults for local capstone runs and replace legacy non-capstone paths."""
    for env_name, path in _CACHE_ENV_DEFAULTS.items():
        if _should_replace_with_capstone_path(os.environ.get(env_name)):
            os.environ[env_name] = str(path)
        else:
            os.environ[env_name] = str(Path(os.environ[env_name]).expanduser())
        Path(os.environ[env_name]).mkdir(parents=True, exist_ok=True)

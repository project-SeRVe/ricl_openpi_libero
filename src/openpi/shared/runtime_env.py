import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_ROOT = REPO_ROOT / ".cache" / "capstone"


def _get_cache_root() -> Path:
    cache_root = os.environ.get("OPENPI_PROJECT_CACHE_ROOT")
    if cache_root:
        return Path(cache_root).expanduser().resolve()
    return DEFAULT_CACHE_ROOT


def _cache_env_defaults(cache_root: Path) -> dict[str, Path]:
    return {
        "OPENPI_DATA_HOME": cache_root / "openpi_cache",
        "HF_HOME": cache_root / "hf_cache",
        "HF_DATASETS_CACHE": cache_root / "hf_cache" / "datasets",
        "TRANSFORMERS_CACHE": cache_root / "hf_cache",
        "HUGGINGFACE_HUB_CACHE": cache_root / "hf_cache",
        "TORCH_HOME": cache_root / "torch_cache",
        "LEROBOT_HOME": cache_root / "hf_cache" / "lerobot",
        "JAX_COMPILATION_CACHE_DIR": cache_root / "jax_cache",
    }


def _should_replace_with_project_cache_path(current_value: str | None, cache_root: Path) -> bool:
    if not current_value:
        return True

    current_path = Path(current_value).expanduser()
    current_path_str = str(current_path)
    legacy_prefix = "/mnt/disks/sdb/"
    return current_path_str.startswith(legacy_prefix) and not current_path_str.startswith(f"{cache_root}/")


def configure_project_cache_env() -> None:
    """Apply cache defaults under the repository and replace legacy external paths."""
    cache_root = _get_cache_root()
    for env_name, path in _cache_env_defaults(cache_root).items():
        if _should_replace_with_project_cache_path(os.environ.get(env_name), cache_root):
            os.environ[env_name] = str(path)
        else:
            os.environ[env_name] = str(Path(os.environ[env_name]).expanduser())
        Path(os.environ[env_name]).mkdir(parents=True, exist_ok=True)

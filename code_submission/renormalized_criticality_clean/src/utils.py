from __future__ import annotations

import json
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


def get_git_revision_hash() -> str:

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def _json_default(o: Any) -> Any:
    

    try:
        import numpy as np  
    except Exception:
        np = None 

    if np is not None:

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)

        if isinstance(o, (np.ndarray,)):
            return o.tolist()


    if isinstance(o, Path):
        return str(o)


    return str(o)


def save_run_manifest(
    output_dir: Union[str, Path],
    config: Dict[str, Any],
    seeds: Dict[str, Any],
    output_paths: List[str],
    filename: str = "run_manifest.json",
) -> Path:
    
    output_dir = Path(output_dir)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_hash": get_git_revision_hash(),
        "config": config,
        "seeds": seeds,
        "output_paths": output_paths,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / filename


    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=4, default=_json_default)
    tmp_path.replace(manifest_path)

    return manifest_path


def safe_save_run_manifest(
    output_dir: Union[str, Path],
    config: Dict[str, Any],
    seeds: Dict[str, Any],
    output_paths: List[str],
    filename: str = "run_manifest.json",
) -> Path | None:
    
    try:
        return save_run_manifest(output_dir, config, seeds, output_paths, filename)
    except Exception as e:

        print(f"WARNING: failed to write run manifest to {output_dir}: {e}", flush=True)
        return None


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:

    config_path = Path(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

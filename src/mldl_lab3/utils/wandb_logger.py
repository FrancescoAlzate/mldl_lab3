import os
from typing import Any, Dict, Optional


_wandb = None


def _maybe_import_wandb():
    global _wandb
    if _wandb is None:
        try:
            import wandb  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("wandb is not installed. Please add it to requirements and pip install.") from exc
        _wandb = wandb
    return _wandb


def setup_wandb(cfg: Dict[str, Any]) -> None:
    wb = _maybe_import_wandb()
    wb_mode = cfg.get("wandb", {}).get("mode", "online")
    if wb_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
    project = cfg["wandb"].get("project", "mldl-lab3")
    entity = cfg["wandb"].get("entity")
    run_name: Optional[str] = cfg["wandb"].get("run_name") or cfg["experiment"]["name"]
    wb.init(project=project, entity=entity, name=run_name, config=cfg)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    wb = _maybe_import_wandb()
    wb.log(metrics, step=step)


def watch_model(model: Any) -> None:
    wb = _maybe_import_wandb()
    wb.watch(model, log="gradients", log_freq=100)


def finish() -> None:
    wb = _maybe_import_wandb()
    wb.finish()


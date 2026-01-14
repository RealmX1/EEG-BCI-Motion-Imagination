"""
Weights & Biases integration for EEG-BCI project.

Provides auto-detection, graceful fallback, and unified logging interface.

Usage:
    from src.utils.wandb_logger import WandbLogger, create_wandb_logger

    # Initialize (auto-detects if wandb is available)
    logger = WandbLogger(
        project="eeg-bci",
        config={"model": "eegnet", "task": "binary"},
        name="S01_eegnet_binary",
        tags=["within-subject", "binary"],
    )

    # Log metrics
    logger.log({"train_loss": 0.5, "train_acc": 0.8})

    # Log confusion matrix
    logger.log_confusion_matrix(y_true, y_pred, class_names)

    # Save model artifact
    logger.save_model("best.pt")

    # Finish run
    logger.finish()
"""

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def is_wandb_available() -> bool:
    """Check if wandb is installed and available."""
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        return False


class WandbLogger:
    """
    WandB logger wrapper class.

    Features:
    - Auto-detects if wandb is available
    - Provides no-op fallback when not installed
    - Unified API interface
    - Supports GPU system metrics monitoring
    """

    def __init__(
        self,
        project: str = "eeg-bci",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
        notes: Optional[str] = None,
        enabled: bool = True,
        log_model: bool = True,
        log_system: bool = True,
    ):
        """
        Initialize WandB logger.

        Args:
            project: wandb project name (default: "eeg-bci")
            entity: wandb team/username (optional)
            name: run name (e.g., "S01_eegnet_binary")
            config: config dictionary (model params, hyperparams, etc.)
            tags: tag list (e.g., ["within-subject", "binary"])
            group: group name (for batch runs)
            job_type: job type (e.g., "train", "eval")
            notes: run notes
            enabled: whether to enable wandb (False uses no-op)
            log_model: whether to log model artifacts
            log_system: whether to monitor system metrics (GPU usage, etc.)
        """
        self._enabled = enabled and is_wandb_available()
        self._run = None
        self._log_model = log_model
        self._step = 0
        self._upload_threads: List[threading.Thread] = []  # Track async uploads

        if not self._enabled:
            if enabled and not is_wandb_available():
                logger.info("wandb not installed, using no-op logger")
            return

        import wandb

        # Initialize wandb run
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            tags=tags,
            group=group,
            job_type=job_type,
            notes=notes,
            settings=wandb.Settings(
                _disable_stats=not log_system,
                console="off",
            ),
        )

        logger.info(f"wandb initialized: {self._run.url}")

    @property
    def enabled(self) -> bool:
        """Whether wandb is enabled."""
        return self._enabled

    @property
    def run(self):
        """Get current wandb run object."""
        return self._run

    @property
    def url(self) -> Optional[str]:
        """Get run URL."""
        if self._run:
            return self._run.url
        return None

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """Log metrics."""
        if not self._enabled:
            return

        import wandb

        if step is None:
            step = self._step
            self._step += 1

        wandb.log(metrics, step=step, commit=commit)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        val_majority_acc: Optional[float] = None,
        learning_rate: Optional[float] = None,
        **extra_metrics,
    ) -> None:
        """Log metrics for a single epoch."""
        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
        }

        if val_majority_acc is not None:
            metrics["val/majority_accuracy"] = val_majority_acc

        if learning_rate is not None:
            metrics["train/learning_rate"] = learning_rate

        metrics.update(extra_metrics)
        self.log(metrics, step=epoch)

    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "confusion_matrix",
    ) -> None:
        """Log confusion matrix."""
        if not self._enabled:
            return

        import wandb

        # Convert to lists if numpy arrays
        y_true_list = y_true.tolist() if hasattr(y_true, 'tolist') else list(y_true)
        y_pred_list = y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)

        if class_names is None:
            n_classes = max(max(y_true_list), max(y_pred_list)) + 1
            class_names = [f"Class {i}" for i in range(n_classes)]

        wandb.log({
            title: wandb.plot.confusion_matrix(
                y_true=y_true_list,
                preds=y_pred_list,
                class_names=class_names,
            )
        })

    def save_model(
        self,
        model_path: Union[str, Path],
        artifact_name: Optional[str] = None,
        artifact_type: str = "model",
        metadata: Optional[Dict[str, Any]] = None,
        async_upload: bool = True,
    ) -> None:
        """Save model as wandb artifact.

        Args:
            model_path: Path to the model file
            artifact_name: Name for the artifact (default: filename stem)
            artifact_type: Type of artifact (default: "model")
            metadata: Optional metadata dict
            async_upload: If True, upload in background thread (default: True)
        """
        if not self._enabled or not self._log_model:
            return

        model_path = Path(model_path)
        if not model_path.exists():
            logger.warning(f"Model path does not exist: {model_path}")
            return

        if artifact_name is None:
            artifact_name = model_path.stem

        def _upload():
            import wandb
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                metadata=metadata,
            )
            artifact.add_file(str(model_path))
            self._run.log_artifact(artifact)
            logger.info(f"Model artifact uploaded: {artifact_name}")

        if async_upload:
            # Run upload in background thread
            thread = threading.Thread(target=_upload, daemon=True)
            thread.start()
            self._upload_threads.append(thread)
            logger.debug(f"Model artifact upload started (async): {artifact_name}")
        else:
            _upload()

    def log_summary(self, metrics: Dict[str, Any]) -> None:
        """Log final summary metrics."""
        if not self._enabled:
            return

        for key, value in metrics.items():
            self._run.summary[key] = value

    def watch(self, model, log: str = "gradients", log_freq: int = 100) -> None:
        """Watch model gradients and parameters."""
        if not self._enabled:
            return

        import wandb
        wandb.watch(model, log=log, log_freq=log_freq)

    def wait_for_uploads(self, timeout: Optional[float] = None) -> None:
        """Wait for all pending async uploads to complete.

        Args:
            timeout: Maximum seconds to wait per thread (None = wait forever)
        """
        if not self._upload_threads:
            return

        logger.debug(f"Waiting for {len(self._upload_threads)} pending uploads...")
        for thread in self._upload_threads:
            thread.join(timeout=timeout)

        # Clean up completed threads
        self._upload_threads = [t for t in self._upload_threads if t.is_alive()]
        if self._upload_threads:
            logger.warning(f"{len(self._upload_threads)} uploads did not complete in time")

    def finish(self, exit_code: int = 0, quiet: bool = False) -> None:
        """Finish wandb run, waiting for pending uploads."""
        if not self._enabled:
            return

        # Wait for pending uploads before finishing
        self.wait_for_uploads()

        import wandb
        wandb.finish(exit_code=exit_code, quiet=quiet)
        logger.info("wandb run finished")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        exit_code = 1 if exc_type is not None else 0
        self.finish(exit_code=exit_code)
        return False


class WandbCallback:
    """Callback class for WithinSubjectTrainer wandb integration."""

    def __init__(self, wandb_logger: WandbLogger):
        """
        Initialize callback.

        Args:
            wandb_logger: WandbLogger instance
        """
        self.logger = wandb_logger
        self._best_val_acc = 0.0

    @property
    def enabled(self) -> bool:
        """Whether wandb is enabled."""
        return self.logger.enabled

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        val_majority_acc: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        """Called at the end of each epoch."""
        self.logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            val_majority_acc=val_majority_acc,
            learning_rate=learning_rate,
        )

        if val_acc > self._best_val_acc:
            self._best_val_acc = val_acc
            self.logger.log_summary({"best_val_accuracy": val_acc})

    def on_train_end(
        self,
        best_epoch: int,
        best_val_acc: float,
        test_acc: Optional[float] = None,
        test_majority_acc: Optional[float] = None,
        model_path: Optional[Path] = None,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """Called at the end of training."""
        summary = {
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
        }

        if test_acc is not None:
            summary["test/accuracy"] = test_acc

        if test_majority_acc is not None:
            summary["test/majority_accuracy"] = test_majority_acc

        self.logger.log_summary(summary)

        # Log confusion matrix if provided
        if y_true is not None and y_pred is not None:
            self.logger.log_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                class_names=class_names,
                title="test/confusion_matrix",
            )

        # Save model artifact if provided
        if model_path and Path(model_path).exists():
            self.logger.save_model(
                model_path,
                metadata={
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_acc,
                    "test_accuracy": test_acc,
                    "test_majority_accuracy": test_majority_acc,
                },
            )


def create_wandb_logger(
    subject_id: str,
    model_type: str,
    task: str,
    paradigm: str = "imagery",
    config: Optional[Dict[str, Any]] = None,
    enabled: bool = True,
    project: str = "eeg-bci",
    entity: Optional[str] = None,
    group: Optional[str] = None,
    **kwargs,
) -> WandbLogger:
    """
    Convenience function to create a wandb logger.

    Args:
        subject_id: Subject ID (e.g., "S01")
        model_type: Model type ("eegnet" or "cbramod")
        task: Task type ("binary", "ternary", "quaternary")
        paradigm: Paradigm ("imagery" or "movement")
        config: Additional config dictionary
        enabled: Whether to enable wandb
        project: wandb project name
        entity: wandb entity (team/username)
        group: wandb run group
        **kwargs: Additional arguments passed to WandbLogger

    Returns:
        WandbLogger instance
    """
    paradigm_short = "MI" if paradigm == "imagery" else "ME"
    name = f"{subject_id}_{model_type}_{task}_{paradigm_short}"

    tags = [
        f"subject:{subject_id}",
        f"model:{model_type}",
        f"task:{task}",
        f"paradigm:{paradigm}",
        "within-subject",
    ]

    full_config = {
        "subject_id": subject_id,
        "model_type": model_type,
        "task": task,
        "paradigm": paradigm,
    }
    if config:
        full_config.update(config)

    return WandbLogger(
        project=project,
        entity=entity,
        name=name,
        config=full_config,
        tags=tags,
        group=group,
        job_type="train",
        enabled=enabled,
        **kwargs,
    )

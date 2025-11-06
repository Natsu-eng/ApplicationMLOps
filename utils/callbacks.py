

# ============================================================================
# CALLBACKS DÉCOUPLÉS
# ============================================================================

from datetime import datetime
from pathlib import Path
from typing import Dict

import torch

from src.shared.logging import get_logger

logger = get_logger(__name__)

class TrainingCallback:
    """Interface de callback abstraite"""
    
    def on_train_begin(self, logs: Dict = None):
        pass
    
    def on_train_end(self, logs: Dict = None):
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Dict = None):
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        pass
    
    def on_batch_begin(self, batch: int, logs: Dict = None):
        pass
    
    def on_batch_end(self, batch: int, logs: Dict = None):
        pass


class LoggingCallback(TrainingCallback):
    """Callback de logging"""
    
    def __init__(self, log_every_n_epochs: int = 1):
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        if (epoch + 1) % self.log_every_n_epochs == 0:
            logger.info(
                f"Epoch {epoch + 1}",
                train_loss=logs.get('train_loss'),
                val_loss=logs.get('val_loss'),
                val_acc=logs.get('val_accuracy'),
                lr=logs.get('lr')
            )


class CheckpointCallback(TrainingCallback):
    """Callback de sauvegarde de checkpoints"""
    
    def __init__(self, checkpoint_dir: Path, save_best_only: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.best_val_loss = float('inf')
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        val_loss = logs.get('val_loss', float('inf'))
        
        if not self.save_best_only or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': logs.get('model_state_dict'),
                'optimizer_state_dict': logs.get('optimizer_state_dict'),
                'val_loss': val_loss,
                'timestamp': datetime.now().isoformat()
            }, checkpoint_path)
            
            logger.info(f"Checkpoint sauvegardé: {checkpoint_path}")


class StreamlitCallback(TrainingCallback):
    """Callback pour interface Streamlit (découplé)"""
    
    def __init__(self, progress_bar=None, status_text=None, total_epochs: int = 100):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_epochs = total_epochs
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        if self.progress_bar:
            progress = (epoch + 1) / self.total_epochs
            self.progress_bar.progress(progress)
        
        if self.status_text:
            text = (f"Epoch {epoch + 1}/{self.total_epochs} | "
                   f"Train: {logs.get('train_loss', 0):.4f} | "
                   f"Val: {logs.get('val_loss', 0):.4f}")
            self.status_text.text(text)


# ============================================================================
# CALLBACKS DÉCOUPLÉS
# ============================================================================

from datetime import datetime
from pathlib import Path
from typing import Dict
from typing import Dict, Optional, Literal

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
                f"Epoch {epoch + 1} - "
                f"train_loss: {logs.get('train_loss')}, "
                f"val_loss: {logs.get('val_loss')}, "
                f"val_acc: {logs.get('val_accuracy')}, "
                f"lr: {logs.get('lr')}"
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


class EarlyStoppingCallback(TrainingCallback):
    """
    Callback d'early stopping robuste avec gestion adaptative.
    
    Fonctionnalités :
    - Patience configurable avec delta minimum
    - Support multi-métriques (loss, accuracy, F1, etc.)
    - Sauvegarde automatique du meilleur modèle
    - Restauration du meilleur état à la fin
    - Logging détaillé pour debugging
    - Gestion des NaN et valeurs infinies
    
    Exemple :
        callback = EarlyStoppingCallback(
            patience=15,
            min_delta=1e-4,
            metric='val_loss',
            mode='min',
            restore_best_weights=True
        )
    """
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        metric: str = 'val_loss',
        mode: Literal['min', 'max'] = 'min',
        restore_best_weights: bool = True,
        checkpoint_path: Optional[Path] = None,
        verbose: bool = True
    ):
        """
        Initialise le callback d'early stopping.
        
        Args:
            patience: Nombre d'epochs sans amélioration avant arrêt
            min_delta: Amélioration minimale pour considérer un progrès
            metric: Métrique à surveiller (ex: 'val_loss', 'val_accuracy', 'val_f1')
            mode: 'min' pour minimiser (loss) ou 'max' pour maximiser (accuracy)
            restore_best_weights: Restaurer les poids du meilleur epoch à la fin
            checkpoint_path: Chemin pour sauvegarder le meilleur modèle (optionnel)
            verbose: Activer les logs détaillés
        """
        super().__init__()
        
        # Validation des paramètres
        if patience <= 0:
            raise ValueError(f"patience doit être > 0, reçu: {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta doit être >= 0, reçu: {min_delta}")
        if mode not in ['min', 'max']:
            raise ValueError(f"mode doit être 'min' ou 'max', reçu: {mode}")
        
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.verbose = verbose
        
        # État interne
        self.best_value: Optional[float] = None
        self.best_epoch: int = 0
        self.wait: int = 0
        self.stopped_epoch: int = 0
        self.best_weights: Optional[Dict] = None
        self.should_stop: bool = False
        
        # Initialisation de best_value selon le mode
        if mode == 'min':
            self.best_value = float('inf')
            self.monitor_op = lambda current, best: current < best - self.min_delta
        else:  # mode == 'max'
            self.best_value = float('-inf')
            self.monitor_op = lambda current, best: current > best + self.min_delta
        
        if self.verbose:
            logger.info(
                f"🛑 EarlyStoppingCallback initialisé - "
                f"metric: {metric}, mode: {mode}, patience: {patience}, "
                f"min_delta: {min_delta:.2e}, restore_weights: {restore_best_weights}"
            )
    
    def on_train_begin(self, logs: Dict = None):
        """Réinitialise l'état au début de l'entraînement."""
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
        
        if self.mode == 'min':
            self.best_value = float('inf')
        else:
            self.best_value = float('-inf')
        
        if self.verbose:
            logger.info(f"🚀 Early stopping activé - surveillance de '{self.metric}'")
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """
        Vérifie l'amélioration à la fin de chaque epoch.
        
        Args:
            epoch: Numéro de l'epoch (0-indexé)
            logs: Dictionnaire contenant les métriques (train_loss, val_loss, etc.)
        """
        if logs is None:
            logger.warning(f"⚠️ Epoch {epoch + 1}: logs=None, early stopping ignoré")
            return
        
        # Récupération de la métrique surveillée
        current_value = logs.get(self.metric)
        
        if current_value is None:
            logger.warning(
                f"⚠️ Epoch {epoch + 1}: métrique '{self.metric}' introuvable dans logs. "
                f"Métriques disponibles: {list(logs.keys())}"
            )
            return
        
        # Gestion des valeurs invalides (NaN, inf)
        if not self._is_valid_value(current_value):
            logger.error(
                f"❌ Epoch {epoch + 1}: valeur invalide pour '{self.metric}': {current_value}"
            )
            self.should_stop = True
            self.stopped_epoch = epoch
            return
        
        # Vérification de l'amélioration
        if self.monitor_op(current_value, self.best_value):
            # Amélioration détectée
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
            
            # Sauvegarde des poids du meilleur modèle
            if self.restore_best_weights and 'model_state_dict' in logs:
                self.best_weights = self._deep_copy_state_dict(logs['model_state_dict'])
            
            # Sauvegarde sur disque si demandé
            if self.checkpoint_path:
                self._save_checkpoint(epoch, logs, current_value)
            
            if self.verbose:
                logger.info(
                    f"✅ Epoch {epoch + 1}: Amélioration détectée ! "
                    f"{self.metric}: {self.best_value:.6f} "
                    f"(Δ={abs(current_value - self.best_value):.2e}) - "
                    f"patience reset: {self.wait}/{self.patience}"
                )
        
        else:
            # Pas d'amélioration
            self.wait += 1
            
            if self.verbose and self.wait % 5 == 0:  # Log tous les 5 epochs sans amélioration
                logger.info(
                    f"⏳ Epoch {epoch + 1}: Pas d'amélioration - "
                    f"{self.metric}: {current_value:.6f} (best: {self.best_value:.6f}) - "
                    f"patience: {self.wait}/{self.patience}"
                )
            
            # Déclenchement de l'early stopping
            if self.wait >= self.patience:
                self.should_stop = True
                self.stopped_epoch = epoch
                
                logger.warning(
                    f"🛑 Early stopping déclenché à l'epoch {epoch + 1} ! "
                    f"Pas d'amélioration pendant {self.patience} epochs. "
                    f"Meilleure valeur: {self.best_value:.6f} (epoch {self.best_epoch + 1})"
                )
    
    def on_train_end(self, logs: Dict = None):
        """Restaure les meilleurs poids et affiche un résumé."""
        if self.stopped_epoch > 0:
            logger.info(
                f"📊 Entraînement arrêté prématurément à l'epoch {self.stopped_epoch + 1} "
                f"(patience épuisée après {self.patience} epochs sans amélioration)"
            )
        
        # Restauration des meilleurs poids
        if self.restore_best_weights and self.best_weights is not None:
            if logs and 'model' in logs:
                try:
                    logs['model'].load_state_dict(self.best_weights)
                    logger.info(
                        f"✅ Poids restaurés depuis le meilleur epoch {self.best_epoch + 1} "
                        f"({self.metric}: {self.best_value:.6f})"
                    )
                except Exception as e:
                    logger.error(f"❌ Erreur restauration des poids: {e}")
        
        # Résumé final
        if self.verbose:
            logger.info(
                f"🏁 Early stopping - Résumé final:\n"
                f"   • Meilleur epoch      : {self.best_epoch + 1}\n"
                f"   • Meilleure valeur    : {self.best_value:.6f}\n"
                f"   • Arrêt prématuré     : {'Oui' if self.stopped_epoch > 0 else 'Non'}\n"
                f"   • Poids restaurés     : {'Oui' if self.restore_best_weights else 'Non'}"
            )
    
    def _is_valid_value(self, value: float) -> bool:
        """Vérifie si une valeur est valide (pas NaN ni inf)."""
        import math
        return not (math.isnan(value) or math.isinf(value))
    
    def _deep_copy_state_dict(self, state_dict: Dict) -> Dict:
        """Copie profonde d'un state_dict PyTorch."""
        import copy
        return copy.deepcopy(state_dict)
    
    def _save_checkpoint(self, epoch: int, logs: Dict, metric_value: float):
        """Sauvegarde un checkpoint du meilleur modèle."""
        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'metric': self.metric,
                'metric_value': metric_value,
                'model_state_dict': logs.get('model_state_dict'),
                'optimizer_state_dict': logs.get('optimizer_state_dict'),
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(checkpoint, self.checkpoint_path)
            
            if self.verbose:
                logger.info(f"💾 Checkpoint sauvegardé: {self.checkpoint_path}")
        
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde checkpoint: {e}")
    
    def get_best_info(self) -> Dict:
        """
        Retourne les informations sur le meilleur epoch.
        
        Returns:
            Dict avec 'best_epoch', 'best_value', 'stopped_early'
        """
        return {
            'best_epoch': self.best_epoch,
            'best_value': self.best_value,
            'stopped_early': self.stopped_epoch > 0,
            'stopped_epoch': self.stopped_epoch if self.stopped_epoch > 0 else None,
            'metric': self.metric,
            'mode': self.mode
        }

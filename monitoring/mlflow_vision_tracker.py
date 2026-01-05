"""
MLflow Tracker spécialisé pour Computer Vision
🆕 VERSION 2.0 avec intégration MLflowRunCollector pour UI
"""

import mlflow
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np
from src.shared.logging import get_logger

# 🆕 IMPORT COLLECTOR
from monitoring.mlflow_collector import get_mlflow_collector

logger = get_logger(__name__)


class ComputerVisionMLflowTracker:
    """
    Tracker MLflow optimisé pour Computer Vision.
    
    🆕 VERSION 2.0 Features:
    - Intégration automatique avec MLflowRunCollector
    - Tracking UI temps réel
    - Mode fallback local si PostgreSQL indisponible
    """
    
    def __init__(self):
        self.enabled = os.getenv("MLFLOW_VISION_ENABLED", "false").lower() == "true"
        self.collector = get_mlflow_collector()  # 🆕 Instance collector
        
        if self.enabled:
            self._setup_tracking()
        else:
            logger.warning(
                "⚠️ MLflow Vision DÉSACTIVÉ\n"
                "   Pour activer: export MLFLOW_VISION_ENABLED=true\n"
                "   Les runs seront collectés localement pour UI uniquement"
            )
    
    def _setup_tracking(self):
        """Configure MLflow pour Computer Vision"""
        try:
            # Base de données séparée
            tracking_uri = os.getenv(
                "MLFLOW_VISION_TRACKING_URI",
                "postgresql+psycopg2://postgres:password@localhost:5432/mlflow_vision_db"
            )
            
            # Experiment dédié
            experiment_name = os.getenv(
                "MLFLOW_VISION_EXPERIMENT_NAME",
                "computer_vision_experiments"
            )
            
            from utils.mlflow import configure_mlflow
            configure_mlflow(tracking_uri, experiment_name)
            
            # Artifact store
            artifact_store = os.getenv(
                "MLFLOW_VISION_ARTIFACT_STORE",
                "./artifacts/computer_vision"
            )
            
            # Créer dossier local
            if not artifact_store.startswith("s3://"):
                Path(artifact_store).mkdir(parents=True, exist_ok=True)
            
            logger.info(
                f"✅ MLflow Computer Vision configuré\n"
                f"   • Tracking URI: {tracking_uri}\n"
                f"   • Experiment: {experiment_name}\n"
                f"   • Artifact Store: {artifact_store}"
            )
            
        except Exception as e:
            logger.error(f"❌ Échec config MLflow Vision: {e}")
            logger.warning("⚠️ MLflow tracking désactivé, collecteur local uniquement")
            self.enabled = False
    
    def start_run(self, run_name: str, tags: Dict[str, str] = None) -> Optional[str]:
        """
        Démarre un run MLflow ET l'ajoute au collecteur UI.
        
        🆕 Modifications:
        - Collecte automatique pour UI
        - Mode fallback si MLflow disabled
        """
        run_id = None
        
        try:
            if self.enabled:
                # Démarrage MLflow standard
                mlflow.start_run(run_name=run_name)
                
                # Tags spécifiques Computer Vision
                default_tags = {
                    "mlflow.source.type": "computer_vision",
                    "framework": "pytorch",
                    "task_type": "image_classification"
                }
                
                if tags:
                    default_tags.update(tags)
                
                for key, value in default_tags.items():
                    mlflow.set_tag(key, value)
                
                run_id = mlflow.active_run().info.run_id
                logger.info(f"🚀 Run MLflow Vision démarré: {run_id}")
            
            else:
                # Mode fallback: génération run_id local
                import uuid
                run_id = f"local_{uuid.uuid4().hex[:16]}"
                logger.info(f"🚀 Run LOCAL démarré (MLflow désactivé): {run_id}")
                
                if tags is None:
                    tags = {}
                
                # Stocker tags localement pour collecteur
                if not hasattr(self, '_local_runs'):
                    self._local_runs = {}
                
                self._local_runs[run_id] = {
                    'run_name': run_name,
                    'tags': tags,
                    'start_time': time.time()
                }
            
            # 🆕 AJOUT AU COLLECTEUR POUR UI
            run_data = {
                'run_id': run_id,
                'run_name': run_name,
                'status': 'RUNNING',
                'start_time': time.time(),
                'model_name': run_name.split('_')[0] if '_' in run_name else run_name,
                'tags': tags or {},
                'mlflow_enabled': self.enabled
            }
            
            self.collector.add_run(run_data)
            logger.info(f"✅ Run ajouté au collecteur UI: {run_id[:16]}...")
            
            return run_id
            
        except Exception as e:
            logger.error(f"❌ Échec start_run: {e}", exc_info=True)
            return None
    
    def get_current_run_id(self) -> Optional[str]:
        """Retourne l'ID du run actif"""
        if not self.enabled:
            # Mode local: chercher dans _local_runs
            if hasattr(self, '_local_runs') and self._local_runs:
                # Retourner le dernier run créé
                return list(self._local_runs.keys())[-1]
            return None
        
        try:
            run = mlflow.active_run()
            return run.info.run_id if run else None
        except:
            return None
    
    def log_model_config(self, model_config: Dict[str, Any]):
        """Log la configuration du modèle"""
        if not self.enabled:
            # Mode local: stocker dans _local_runs
            run_id = self.get_current_run_id()
            if run_id and hasattr(self, '_local_runs'):
                if 'model_config' not in self._local_runs[run_id]:
                    self._local_runs[run_id]['model_config'] = {}
                self._local_runs[run_id]['model_config'].update(model_config)
            return
        
        if not mlflow.active_run():
            return
        
        try:
            for key, value in model_config.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(f"model_{key}", value)
                elif hasattr(value, 'value'):
                    mlflow.log_param(f"model_{key}", value.value)
        except Exception as e:
            logger.warning(f"⚠️ Échec log model_config: {e}")
    
    def log_training_config(self, training_config: Dict[str, Any]):
        """Log la configuration d'entraînement"""
        if not self.enabled:
            run_id = self.get_current_run_id()
            if run_id and hasattr(self, '_local_runs'):
                if 'training_config' not in self._local_runs[run_id]:
                    self._local_runs[run_id]['training_config'] = {}
                self._local_runs[run_id]['training_config'].update(training_config)
            return
        
        if not mlflow.active_run():
            return
        
        try:
            for key, value in training_config.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(f"training_{key}", value)
                elif hasattr(value, 'value'):
                    mlflow.log_param(f"training_{key}", value.value)
        except Exception as e:
            logger.warning(f"⚠️ Échec log training_config: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log les métriques avec validation renforcée.
        
        🆕 Collecte aussi pour UI
        """
        # Validation et nettoyage
        clean_metrics = {}
        
        for key, value in metrics.items():
            if value is None:
                continue
            
            try:
                float_value = float(value)
            except (ValueError, TypeError):
                logger.warning(f"⚠️ Métrique '{key}' non numérique ignorée: {value}")
                continue
            
            if np.isnan(float_value) or np.isinf(float_value):
                logger.warning(f"⚠️ Métrique '{key}' invalide ignorée: {float_value}")
                continue
            
            clean_metrics[key] = float_value
        
        # Log MLflow si enabled
        if self.enabled and mlflow.active_run():
            try:
                for key, value in clean_metrics.items():
                    mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.warning(f"⚠️ Échec log metrics MLflow: {e}")
        
        # 🆕 MISE À JOUR COLLECTEUR
        run_id = self.get_current_run_id()
        if run_id:
            # Récupérer run existant
            existing_run = self.collector.get_run(run_id)
            
            if existing_run:
                # Ajouter/mettre à jour métriques
                if 'metrics' not in existing_run:
                    existing_run['metrics'] = {}
                
                existing_run['metrics'].update(clean_metrics)
                existing_run['last_update'] = time.time()
                
                # Re-ajouter au collecteur (écrase ancien)
                self.collector.remove_run(run_id)
                self.collector.add_run(existing_run, trigger_callbacks=False)
    
    def log_model_artifact(
        self, 
        model: torch.nn.Module, 
        filename: str = "model.pt",
        additional_files: Dict[str, Any] = None
    ):
        """
        Sauvegarde le modèle comme artifact.
        
        🆕 Met à jour le collecteur avec chemin artifact
        """
        if not self.enabled and not mlflow.active_run():
            logger.debug("MLflow désactivé, artifacts non sauvegardés")
            return
        
        temp_files = []
        
        try:
            # Sauvegarde temporaire
            temp_path = Path(f"./temp/{filename}")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_files.append(temp_path)
            
            torch.save(model.state_dict(), temp_path)
            
            # Log artifact
            if self.enabled:
                mlflow.log_artifact(str(temp_path), artifact_path="models")
            
            # Fichiers additionnels
            if additional_files:
                for name, obj in additional_files.items():
                    add_path = temp_path.parent / name
                    temp_files.append(add_path)
                    
                    if isinstance(obj, dict):
                        import json
                        with open(add_path, 'w') as f:
                            json.dump(obj, f, indent=2, default=str)
                    else:
                        import joblib
                        joblib.dump(obj, add_path)
                    
                    if self.enabled:
                        mlflow.log_artifact(str(add_path), artifact_path="models")
            
            # 🆕 MISE À JOUR COLLECTEUR
            run_id = self.get_current_run_id()
            if run_id:
                existing_run = self.collector.get_run(run_id)
                
                if existing_run:
                    existing_run['artifact_path'] = filename
                    existing_run['artifacts_count'] = 1 + (len(additional_files) if additional_files else 0)
                    existing_run['model_saved'] = True
                    
                    self.collector.remove_run(run_id)
                    self.collector.add_run(existing_run, trigger_callbacks=False)
            
            logger.info(f"✅ Modèle et {len(temp_files)-1} artifacts sauvegardés: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Échec log model artifact: {e}", exc_info=True)
        
        finally:
            # Nettoyage
            for temp_file in temp_files:
                try:
                    temp_file.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"⚠️ Impossible de supprimer {temp_file}: {e}")
    
    def log_image_artifact(self, image_path: str, artifact_path: str = "images"):
        """Log une image comme artifact"""
        if not self.enabled or not mlflow.active_run():
            return
        
        try:
            mlflow.log_artifact(image_path, artifact_path=artifact_path)
        except Exception as e:
            logger.warning(f"⚠️ Échec log image: {e}")
    
    def log_confusion_matrix(self, cm: np.ndarray, class_names: list = None):
        """Log une matrice de confusion"""
        if not self.enabled or not mlflow.active_run():
            return
        
        temp_path = None
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            
            if class_names:
                ax.set_xticklabels(class_names, rotation=45, ha='right')
                ax.set_yticklabels(class_names, rotation=0)
            
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            
            temp_path = Path(f"./temp/confusion_matrix_{int(time.time())}.png")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            plt.tight_layout()
            plt.savefig(temp_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            mlflow.log_artifact(str(temp_path), artifact_path="metrics")
            
            logger.info("✅ Matrice de confusion sauvegardée")
            
        except Exception as e:
            logger.warning(f"⚠️ Échec log confusion matrix: {e}")
        
        finally:
            if temp_path:
                try:
                    temp_path.unlink(missing_ok=True)
                except:
                    pass
    
    def log_training_curves(self, history: Dict[str, list]):
        """Log les courbes d'entraînement"""
        if not self.enabled or not mlflow.active_run():
            return
        
        temp_path = None
        
        try:
            import matplotlib.pyplot as plt
            
            available_metrics = []
            if 'train_loss' in history and 'val_loss' in history:
                available_metrics.append('loss')
            if 'val_accuracy' in history:
                available_metrics.append('accuracy')
            if 'val_f1' in history:
                available_metrics.append('f1')
            if 'learning_rates' in history:
                available_metrics.append('lr')
            
            if not available_metrics:
                logger.warning("⚠️ Aucune métrique pour courbes")
                return
            
            n_plots = len(available_metrics)
            n_cols = 2
            n_rows = (n_plots + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            plot_idx = 0
            
            # Loss
            if 'loss' in available_metrics:
                axes[plot_idx].plot(history['train_loss'], label='Train', linewidth=2)
                axes[plot_idx].plot(history['val_loss'], label='Validation', linewidth=2)
                axes[plot_idx].set_title('Loss', fontsize=12, fontweight='bold')
                axes[plot_idx].set_xlabel('Epoch')
                axes[plot_idx].set_ylabel('Loss')
                axes[plot_idx].legend()
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
            
            # Accuracy
            if 'accuracy' in available_metrics:
                axes[plot_idx].plot(history['val_accuracy'], label='Validation', linewidth=2, color='green')
                axes[plot_idx].set_title('Accuracy', fontsize=12, fontweight='bold')
                axes[plot_idx].set_xlabel('Epoch')
                axes[plot_idx].set_ylabel('Accuracy')
                axes[plot_idx].legend()
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
            
            # F1 Score
            if 'f1' in available_metrics:
                axes[plot_idx].plot(history['val_f1'], label='Validation', linewidth=2, color='orange')
                axes[plot_idx].set_title('F1 Score', fontsize=12, fontweight='bold')
                axes[plot_idx].set_xlabel('Epoch')
                axes[plot_idx].set_ylabel('F1 Score')
                axes[plot_idx].legend()
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
            
            # Learning Rate
            if 'lr' in available_metrics:
                axes[plot_idx].plot(history['learning_rates'], linewidth=2, color='red')
                axes[plot_idx].set_title('Learning Rate', fontsize=12, fontweight='bold')
                axes[plot_idx].set_xlabel('Epoch')
                axes[plot_idx].set_ylabel('Learning Rate')
                axes[plot_idx].set_yscale('log')
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
            
            # Masquer axes non utilisés
            for idx in range(plot_idx, len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            
            temp_path = Path(f"./temp/training_curves_{int(time.time())}.png")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(temp_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            mlflow.log_artifact(str(temp_path), artifact_path="metrics")
            
            logger.info("✅ Courbes d'entraînement sauvegardées")
            
        except Exception as e:
            logger.warning(f"⚠️ Échec log training curves: {e}", exc_info=True)
        
        finally:
            if temp_path:
                try:
                    temp_path.unlink(missing_ok=True)
                except:
                    pass
    
    def end_run(self, status: str = "FINISHED"):
        """
        Termine le run actif.
        
        🆕 Met à jour le collecteur avec statut final
        """
        run_id = self.get_current_run_id()
        
        if self.enabled and mlflow.active_run():
            try:
                mlflow.end_run(status=status)
                logger.info(f"✅ Run MLflow terminé: {status}")
            except Exception as e:
                logger.error(f"❌ Échec end_run: {e}")
        
        # 🆕 MISE À JOUR COLLECTEUR
        if run_id:
            existing_run = self.collector.get_run(run_id)
            
            if existing_run:
                existing_run['status'] = status
                existing_run['end_time'] = time.time()
                
                if 'start_time' in existing_run:
                    existing_run['duration'] = existing_run['end_time'] - existing_run['start_time']
                
                self.collector.remove_run(run_id)
                self.collector.add_run(existing_run, trigger_callbacks=True)  # Trigger pour UI update
                
                logger.info(f"✅ Run collecteur mis à jour: {status}")


# Instance globale
cv_mlflow_tracker = ComputerVisionMLflowTracker()
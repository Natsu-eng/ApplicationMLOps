"""
DataLab Pro - State Manager Global avec MLflow intégré
Gestion centralisée de l'état de l'application Streamlit.
"""

import os
import time
import threading
from typing import Any, Dict, Optional, Set, Tuple, List
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import streamlit as st
import numpy as np
import pandas as pd

from monitoring.mlflow_collector import get_mlflow_collector

# Import du système de logging centralisé
from src.shared.logging import get_logger
from utils.task_detector import TaskType

# ========================
# LOGGER
# ========================
logger = get_logger(__name__)

# ========================
# ENUMS
# ========================
class DataType(Enum):
    """Types de données supportés"""
    NONE = "none"
    TABULAR = "tabular"
    IMAGES = "images"

class AppPage(Enum):
    """Pages de l'application"""
    HOME = "main.py"
    DASHBOARD = "pages/1_dashboard.py"
    ML_TRAINING = "pages/2_training.py"
    ML_EVALUATION = "pages/3_evaluation.py"
    CV_TRAINING = "pages/4_training_computer.py"
    ANOMALY_EVAL = "pages/5_anomaly_evaluation.py"

class TrainingStep(Enum):
    """Étapes du workflow ML"""
    DATA_ANALYSIS = 0
    TARGET_SELECTION = 1
    IMBALANCE_ANALYSIS = 2
    PREPROCESSING = 3
    MODEL_SELECTION = 4
    TRAINING_LAUNCH = 5

# ========================
# CONFIG & STATS
# ========================
@dataclass
class Config:
    """Configuration globale de l'application"""
    MAX_FILE_MB: int = 500
    SUPPORTED_EXT: Set[str] = field(default_factory=lambda: {
        '.csv', '.xlsx', '.parquet', '.feather', '.json'
    })
    TIMEOUT_MIN: int = 60
    MAX_MESSAGE_SIZE: int = 500  # MB
    MLFLOW_EXPERIMENT_NAME: str = "datalab_experiments"
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"

@dataclass
class CalcStats:
    """Statistiques de calcul"""
    total: int = 0
    failed: int = 0
    last: float = 0.0
    avg: float = 0.0
    active: int = 0
    history: list = field(default_factory=list)
    max_hist: int = 100

# ========================
# STATE CLASSES
# ========================
@dataclass
class NavState:
    """État de navigation"""
    current: AppPage = AppPage.HOME
    authorized: Set[AppPage] = field(default_factory=set)
    last_active: float = field(default_factory=time.time)
    current_step: int = 0  # Pour workflow UI

@dataclass
class DataState:
    """État des données chargées"""
    # Méta
    loaded: bool = False
    type: DataType = DataType.NONE
    name: Optional[str] = None
    loaded_at: Optional[float] = None

    # Données tabulaires (ML classique)
    df: Optional[pd.DataFrame] = None
    df_raw: Optional[pd.DataFrame] = None
    feature_list: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    task_type: Optional[str] = None

    # Données images (Computer Vision)
    X: Optional[np.ndarray] = None
    X_norm: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    X_train: Optional[np.ndarray] = None
    X_val: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    dir: Optional[str] = None
    structure: Optional[dict] = None
    info: Optional[dict] = None
    task: Optional[str] = None
    img_count: Optional[int] = None
    img_shape: Optional[tuple] = None
    n_classes: Optional[int] = None

@dataclass
class TrainingState:
    """
    État de l'entraînement unifié (ML classique + Computer Vision)
    🆕 v5.0: Intégration MLflow complète
    """
    # === WORKFLOW GÉNÉRAL ===
    current_step: int = 0
    workflow_complete: bool = False
    
    # === CONFIGURATION ML CLASSIQUE ===
    selected_models: List[str] = field(default_factory=list)
    test_size: int = 20
    optimize_hyperparams: bool = False
    
    # === CONFIGURATION COMPUTER VISION ===
    selected_model_type: Optional[str] = None
    model_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    # === CONFIGURATION PRÉTRAITEMENT ===
    preprocessing_config: Dict[str, Any] = field(default_factory=lambda: {
        'numeric_imputation': 'mean',
        'categorical_imputation': 'most_frequent',
        'remove_constant_cols': True,
        'remove_identifier_cols': True,
        'scale_features': True,
        'scaling_method': 'standard',
        'encoding_method': 'onehot',
        'pca_preprocessing': False,
        'use_smote': False,
        'smote_k_neighbors': 5
    })
    
    # === CONFIGURATION DÉSÉQUILIBRE ===
    imbalance_config: Dict[str, Any] = field(default_factory=lambda: {
        'imbalance_detected': False,
        'imbalance_ratio': 1.0,
        'use_class_weights': False,
        'use_smote': False,
        'smote_k_neighbors': 5,
        'smote_sampling_strategy': 'auto'
    })
    
    # === EXPÉRIMENTATION ===
    current_experiment: Optional[str] = None
    
    # === RÉSULTATS ===
    training_results: Optional[Any] = None  # MLTrainingResult ou dict
    trained_model: Any = None
    training_history: Optional[Dict[str, Any]] = None
    preprocessor: Any = None
    class_weights: Optional[Dict] = None
    ml_results: Optional[List[Dict]] = None  # Legacy format

    # === 🆕 MLFLOW RUNS (CRITIQUE) ===
    mlflow_runs: List[Dict[str, Any]] = field(default_factory=list)
    mlflow_experiment_id: Optional[str] = None
    mlflow_experiment_name: str = "datalab_experiments"
    
    # === MÉTHODES HELPER MLFLOW ===
    def add_mlflow_run(self, run: Dict[str, Any]) -> bool:
        """
        Ajoute un run MLflow avec déduplication automatique.
        
        Args:
            run: Dict contenant au minimum 'run_id'
            
        Returns:
            True si ajouté, False si dupliqué ou invalide
        """
        if not isinstance(run, dict):
            logger.warning(f"⚠️ Run invalide (type: {type(run)})")
            return False
        
        run_id = run.get('run_id')
        if not run_id:
            logger.warning("⚠️ Run sans run_id ignoré")
            return False
        
        # Déduplication
        existing_ids = {r.get('run_id') for r in self.mlflow_runs if r.get('run_id')}
        
        if run_id in existing_ids:
            logger.debug(f"Run {run_id[:8]} déjà existant")
            return False
        
        self.mlflow_runs.append(run)
        logger.debug(f"✅ Run {run_id[:8]} ajouté (total: {len(self.mlflow_runs)})")
        return True
    
    def get_mlflow_runs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Récupère les runs MLflow avec limite optionnelle.
        
        Args:
            limit: Nombre max de runs à retourner
            
        Returns:
            Liste de runs
        """
        if limit and limit > 0:
            return self.mlflow_runs[:limit]
        return self.mlflow_runs
    
    def clear_mlflow_runs(self) -> int:
        """
        Nettoie tous les runs MLflow.
        
        Returns:
            Nombre de runs supprimés
        """
        count = len(self.mlflow_runs)
        self.mlflow_runs = []
        logger.info(f"🗑️ {count} runs MLflow supprimés")
        return count
    
    def get_mlflow_runs_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Filtre runs par statut.
        
        Args:
            status: 'FINISHED', 'RUNNING', 'FAILED', etc.
            
        Returns:
            Liste de runs filtrés
        """
        return [r for r in self.mlflow_runs if r.get('status') == status]
    
    def get_mlflow_runs_by_model(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Filtre runs par nom de modèle.
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            Liste de runs filtrés
        """
        return [
            r for r in self.mlflow_runs 
            if r.get('model_name') == model_name or 
               r.get('tags', {}).get('mlflow.runName') == model_name
        ]

@dataclass
class MetricsState:
    """Métriques système et application"""
    start: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    errors: int = 0
    success: int = 0
    memory_percent: float = 0.0
    calc: CalcStats = field(default_factory=CalcStats)

# ========================
# GLOBAL STATE MANAGER (Singleton Thread-Safe)
# ========================
class StateManager:
    """
    Gestionnaire d'état global de l'application.
    
    Pattern Singleton avec synchronisation thread-safe.
    Gère navigation, données, entraînement, métriques et MLflow.
    
    🆕 v5.0: Support MLflow complet avec synchronisation multi-niveaux
    """
    
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        """Implémentation Singleton thread-safe"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """Initialisation unique du singleton"""
        if not self._initialized:
            self.config = Config()
            self._setup_session()
            
            # 🆕 Intégration collecteur MLflow
            self.mlflow_collector = get_mlflow_collector()
            
            # Enregistrement callback bidirectionnel
            self.mlflow_collector.register_callback(self._on_mlflow_run_collected)
            
            self._initialized = True
            logger.info("StateManager v1.0 initialisé avec collecteur MLflow intégré")


    def _on_mlflow_run_collected(self, run: Dict[str, Any]):
        """
        Callback: synchronise chaque run collecté vers session_state.
        Appelé automatiquement par le collecteur MLflow.
        """
        try:
            # Ajout dans training state
            added = st.session_state.training.add_mlflow_run(run)
            
            if added:
                logger.debug(f"Callback: Run {run.get('run_id', 'N/A')[:8]} → STATE")
        
        except Exception as e:
            logger.error(f"❌ Callback _on_mlflow_run_collected: {e}")


    def _setup_session(self):
        """
        Initialisation sécurisée de tous les états dans session_state.
        Garantit que tous les états existent avec valeurs par défaut.
        """
        with self._lock:
            defaults = {
                'nav': NavState(),
                'data': DataState(),
                'training': TrainingState(),
                'metrics': MetricsState(),
                'ready': True,
                'time': time.strftime('%H:%M:%S')
            }
            
            for key, default_value in defaults.items():
                if key not in st.session_state:
                    st.session_state[key] = default_value
                    logger.debug(f"Initialized session_state.{key}")
            
            self._update_auth()

    # ========================================
    # GESTION AUTORISATIONS
    # ========================================
    
    def _update_auth(self):
        """
        Met à jour les pages autorisées selon le type de données chargées.
        
        - Aucune donnée: HOME uniquement
        - Données tabulaires: HOME + DASHBOARD + ML_TRAINING + ML_EVALUATION
        - Données images: HOME + DASHBOARD + CV_TRAINING + ANOMALY_EVAL
        """
        data_type = st.session_state.data.type
        authorized = {AppPage.HOME}
        
        if data_type == DataType.TABULAR:
            authorized.update({
                AppPage.DASHBOARD,
                AppPage.ML_TRAINING,
                AppPage.ML_EVALUATION
            })
            logger.debug("Auth: Tabular → ML pages enabled")
        
        elif data_type == DataType.IMAGES:
            authorized.update({
                AppPage.DASHBOARD,
                AppPage.CV_TRAINING,
                AppPage.ANOMALY_EVAL
            })
            logger.debug("Auth: Images → CV pages enabled")
        
        st.session_state.nav.authorized = authorized

    # ========================================
    # NAVIGATION
    # ========================================
    
    def go(self, page: AppPage) -> bool:
        """
        Change la page courante si autorisée.
        
        Args:
            page: Page cible (AppPage enum)
            
        Returns:
            True si navigation autorisée, False sinon
        """
        if page not in st.session_state.nav.authorized:
            logger.warning(f"⚠️ Accès refusé: {page.value}")
            st.warning(f"⚠️ Accès refusé à {page.name}. Veuillez charger des données appropriées.")
            return False
        
        st.session_state.nav.current = page
        st.session_state.nav.last_active = time.time()
        logger.info(f"→ {page.value}")
        return True

    def switch(self, page: AppPage) -> bool:
        """
        Change de page avec transition Streamlit.
        
        Args:
            page: Page cible
            
        Returns:
            True si succès, False si échec
        """
        if not self.go(page):
            return False
        
        try:
            st.switch_page(page.value)
            return True
        except Exception as e:
            logger.error(f"❌ switch_page failed: {e}", exc_info=True)
            st.error(f"Erreur de navigation: {e}")
            return False

    # ========================================
    # CHARGEMENT DONNÉES
    # ========================================
    
    def set_tabular(
        self, 
        df: pd.DataFrame, 
        df_raw: Optional[pd.DataFrame], 
        name: str
    ) -> bool:
        """
        Charge des données tabulaires.
        
        Args:
            df: DataFrame nettoyé
            df_raw: DataFrame brut original (optionnel)
            name: Nom du fichier/dataset
            
        Returns:
            True si succès, False si échec
        """
        try:
            if df is None or df.empty:
                logger.error("❌ DataFrame vide ou None")
                return False
            
            # Reset complet avant nouveau chargement
            self.reset_data()
            
            d = st.session_state.data
            d.df = df.copy()
            d.df_raw = df_raw.copy() if df_raw is not None else df.copy()
            d.name = name
            d.type = DataType.TABULAR
            d.loaded = True
            d.loaded_at = time.time()
            
            self._update_auth()
            
            logger.info(f"✅ Tabular loaded: {name} | {df.shape}")
            return True
        
        except Exception as e:
            logger.error(f"❌ set_tabular error: {e}", exc_info=True)
            st.error(f"Erreur chargement données: {e}")
            return False

    def set_images(
        self, 
        X: np.ndarray, 
        X_norm: np.ndarray, 
        y: np.ndarray, 
        dir_path: str, 
        structure: dict, 
        info: dict,
        y_train: Optional[np.ndarray] = None  # ✅ AJOUT
    ) -> bool:
        """
        Charge des données images avec détection intelligente unsupervised/supervised.
        
        Args:
            y_train: Labels du TRAIN UNIQUEMENT (pour MVTec AD → détection unsupervised)
        """
        try:
            if len(X) == 0 or len(X) != len(y):
                logger.error(f"Images invalides: len(X)={len(X)}, len(y)={len(y)}")
                return False
            
            # Reset avant chargement
            self.reset_data()
            
            d = st.session_state.data
            d.X = X
            d.X_norm = X_norm
            d.y = y
            d.dir = dir_path
            d.structure = structure
            d.info = info
            d.type = DataType.IMAGES
            d.name = os.path.basename(dir_path)
            d.loaded = True
            d.loaded_at = time.time()
            d.img_count = len(X)
            d.img_shape = X.shape[1:] if len(X.shape) > 3 else X.shape[1:]
            
            d.y_train = y_train
            
            # === DÉTECTION INTELLIGENTE DE LA TÂCHE ===
            from utils.task_detector import detect_cv_task
            
            # Si y_train fourni, l'utiliser pour détection
            labels_for_detection = y_train if y_train is not None else y
            
            task_type, task_metadata = detect_cv_task(labels_for_detection)
            
            d.task = task_type.value
            d.n_classes = task_metadata["n_classes"]
            d.task_metadata = task_metadata
            
            self._update_auth()
            
            # Logs détaillés
            task_name = {
                "unsupervised": "🔍 Unsupervised (MVTec AD)",
                "anomaly_detection": "⚠️ Anomaly Supervised",
                "binary_classification": "🎯 Binary Classification",
                "multiclass_classification": "🎯 Multiclass Classification"
            }.get(task_type.value, task_type.value)
            
            logger.info(
                f"✅ Images chargées: {len(X)} images | "
                f"Tâche détectée: {task_name} | "
                f"y_train fourni: {y_train is not None} | "
                f"Classes détectées: {len(np.unique(labels_for_detection))}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ set_images error: {e}", exc_info=True)
            st.error(f"Erreur chargement images: {e}")
            return False

    # ========================================
    # RESET
    # ========================================
    
    def reset_data(self):
        """Reset sécurisé des données et état d'entraînement"""
        with self._lock:
            st.session_state.data = DataState()
            st.session_state.training = TrainingState()
            self._update_auth()
            logger.info("🗑️ Data + Training reset")

    def reset_all(self):
        """Reset complet de tous les états"""
        with self._lock:
            st.session_state.data = DataState()
            st.session_state.nav = NavState()
            st.session_state.training = TrainingState()
            st.session_state.metrics = MetricsState()
            self._update_auth()
            logger.info("🗑️ Full reset (all states)")

    # ========================================
    # VALIDATION
    # ========================================
    
    def validate(self) -> Tuple[bool, str]:
        """
        Valide que des données sont chargées et valides.
        
        Returns:
            (is_valid, message)
        """
        d = st.session_state.data
        
        if not d.loaded:
            return False, "Aucune donnée chargée"
        
        if d.type == DataType.TABULAR:
            if d.df is None or d.df.empty:
                return False, "DataFrame vide"
            return True, f"{len(d.df):,} lignes × {len(d.df.columns)} colonnes"
        
        if d.type == DataType.IMAGES:
            if d.X is None or len(d.X) == 0:
                return False, "Aucune image"
            if len(d.X) != len(d.y):
                return False, f"Incohérence X ({len(d.X)}) ≠ y ({len(d.y)})"
            return True, f"{len(d.X):,} images | {d.n_classes} classes"
        
        return False, "Type de données inconnu"

    # ========================================
    # MÉTRIQUES & MONITORING
    # ========================================
    
    @contextmanager
    def calc(self, name: str = "operation"):
        """
        Context manager pour tracking des opérations.
        
        Usage:
            with STATE.calc("mon_operation"):
                # code...
        
        Args:
            name: Nom de l'opération (pour logs)
        """
        start_time = time.time()
        stats = st.session_state.metrics.calc
        
        with self._lock:
            stats.active += 1
            stats.total += 1
        
        try:
            yield
        
        except Exception as e:
            with self._lock:
                stats.failed += 1
                st.session_state.metrics.errors += 1
            logger.error(f"❌ Calc error [{name}]: {e}", exc_info=True)
            raise
        
        finally:
            duration = time.time() - start_time
            
            with self._lock:
                stats.active -= 1
                stats.last = duration
                stats.history.append(duration)
                
                if len(stats.history) > stats.max_hist:
                    stats.history.pop(0)
                
                stats.avg = sum(stats.history) / len(stats.history) if stats.history else 0
                st.session_state.metrics.success += 1

    def touch(self):
        """Met à jour le timestamp d'activité"""
        st.session_state.metrics.last_active = time.time()
        st.session_state.time = time.strftime('%H:%M:%S')

    def get(self, key: str, default: Any = None) -> Any:
        """
        Accès sécurisé aux données de session.
        
        Args:
            key: Clé de session_state
            default: Valeur par défaut si clé absente
            
        Returns:
            Valeur ou default
        """
        return st.session_state.get(key, default)

    # ========================================
    # 🆕 MÉTHODES MLFLOW PUBLIQUES
    # ========================================
    
    def add_mlflow_run(self, run: Dict[str, Any]) -> bool:
        """
        Ajoute un run avec collecteur centralisé.
        Délègue au collecteur pour cohérence globale.
        """
        # Délégation au collecteur (source unique de vérité)
        collected = self.mlflow_collector.add_run(run)
        
        if collected:
            # Ajout local (déjà fait par callback mais double sécurité)
            return st.session_state.training.add_mlflow_run(run)
        
        return False
    
    def get_mlflow_runs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Récupère runs depuis collecteur (source unique).
        """
        return self.mlflow_collector.get_runs(limit)
    
    def clear_mlflow_runs(self) -> int:
        """
        Nettoie runs partout (collecteur + states).
        """
        # Collecteur
        count_collector = self.mlflow_collector.clear()
        
        # Training state
        count_training = st.session_state.training.clear_mlflow_runs()
        
        # Session state
        if hasattr(st.session_state, 'mlflow_runs'):
            count_session = len(st.session_state.mlflow_runs)
            st.session_state.mlflow_runs = []
        else:
            count_session = 0
        
        logger.info(
            f"🗑️ MLflow runs nettoyés:\n"
            f"   • Collecteur: {count_collector}\n"
            f"   • Training: {count_training}\n"
            f"   • Session: {count_session}"
        )
        
        return count_collector
    
    def sync_mlflow_runs(self, runs: List[Dict[str, Any]]) -> int:
        """
        Synchronise une liste de runs MLflow.
        
        Args:
            runs: Liste de runs à synchroniser
            
        Returns:
            Nombre de nouveaux runs ajoutés
        """
        count = 0
        for run in runs:
            if self.add_mlflow_run(run):
                count += 1
        
        if count > 0:
            logger.info(f"✅ {count} nouveaux runs MLflow synchronisés")
        
        return count

    # ========================================
    # PROPERTIES (Accès unifié et sécurisé)
    # ========================================
    
    # --- Navigation ---
    @property
    def page(self) -> AppPage:
        """Page courante"""
        return st.session_state.nav.current
    
    # --- Type de données ---
    @property
    def dtype(self) -> DataType:
        """Type de données chargées"""
        return st.session_state.data.type
    
    @property
    def loaded(self) -> bool:
        """Données chargées ?"""
        return st.session_state.data.loaded
    
    @property
    def tabular(self) -> bool:
        """Données tabulaires chargées ?"""
        return self.loaded and self.dtype == DataType.TABULAR
    
    @property
    def images(self) -> bool:
        """Images chargées ?"""
        return self.loaded and self.dtype == DataType.IMAGES
    
    # --- États ---
    @property
    def data(self) -> DataState:
        """État des données"""
        return st.session_state.data
    
    @property
    def nav(self) -> NavState:
        """État de navigation"""
        return st.session_state.nav
    
    @property
    def metrics(self) -> MetricsState:
        """Métriques système"""
        return st.session_state.metrics
    
    @property
    def training(self) -> TrainingState:
        """État d'entraînement"""
        return st.session_state.training
    
    # --- Workflow ---
    @property
    def current_step(self) -> int:
        """Étape courante du workflow"""
        return st.session_state.training.current_step
    
    @current_step.setter
    def current_step(self, value: int):
        st.session_state.training.current_step = value
    
    @property
    def workflow_complete(self) -> bool:
        """Workflow terminé ?"""
        return st.session_state.training.workflow_complete
    
    @workflow_complete.setter
    def workflow_complete(self, value: bool):
        st.session_state.training.workflow_complete = value
    
    # --- Modèles ---
    @property
    def selected_models(self) -> List[str]:
        """Modèles sélectionnés (ML classique)"""
        return st.session_state.training.selected_models
    
    @selected_models.setter
    def selected_models(self, value: List[str]):
        st.session_state.training.selected_models = value
    
    @property
    def selected_model_type(self) -> Optional[str]:
        """Type de modèle sélectionné (CV)"""
        return st.session_state.training.selected_model_type
    
    @selected_model_type.setter
    def selected_model_type(self, value: Optional[str]):
        st.session_state.training.selected_model_type = value
    
    # --- Configuration ---
    @property
    def test_size(self) -> int:
        """Taille du jeu de test (%)"""
        return st.session_state.training.test_size
    
    @test_size.setter
    def test_size(self, value: int):
        st.session_state.training.test_size = value
    
    @property
    def optimize_hyperparams(self) -> bool:
        """Optimisation hyperparamètres activée ?"""
        return st.session_state.training.optimize_hyperparams
    
    @optimize_hyperparams.setter
    def optimize_hyperparams(self, value: bool):
        st.session_state.training.optimize_hyperparams = value
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Configuration du modèle"""
        return st.session_state.training.model_config
    
    @model_config.setter
    def model_config(self, value: Dict[str, Any]):
        st.session_state.training.model_config = value
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """Configuration d'entraînement"""
        return st.session_state.training.training_config
    
    @training_config.setter
    def training_config(self, value: Dict[str, Any]):
        st.session_state.training.training_config = value
    
    @property
    def preprocessing_config(self) -> Dict[str, Any]:
        """Configuration prétraitement"""
        return st.session_state.training.preprocessing_config
    
    @preprocessing_config.setter
    def preprocessing_config(self, value: Dict[str, Any]):
        st.session_state.training.preprocessing_config = value
    
    @property
    def imbalance_config(self) -> Dict[str, Any]:
        """Configuration déséquilibre"""
        return st.session_state.training.imbalance_config
    
    @imbalance_config.setter
    def imbalance_config(self, value: Dict[str, Any]):
        st.session_state.training.imbalance_config = value
    
    # --- Features & Target ---
    @property
    def feature_list(self) -> List[str]:
        """Liste des features"""
        return st.session_state.data.feature_list
    
    @feature_list.setter
    def feature_list(self, value: List[str]):
        st.session_state.data.feature_list = value
    
    @property
    def target_column(self) -> Optional[str]:
        """Colonne cible"""
        return st.session_state.data.target_column
    
    @target_column.setter
    def target_column(self, value: Optional[str]):
        st.session_state.data.target_column = value
    
    @property
    def task_type(self) -> Optional[str]:
        """Type de tâche ML"""
        return st.session_state.data.task_type
    
    @task_type.setter
    def task_type(self, value: Optional[str]):
        st.session_state.data.task_type = value
    
    # --- Résultats ---
    @property
    def training_results(self) -> Optional[Any]:
        """Résultats d'entraînement (MLTrainingResult)"""
        return st.session_state.training.training_results
    
    @training_results.setter
    def training_results(self, value: Any):
        st.session_state.training.training_results = value
    
    @property
    def trained_model(self) -> Any:
        """Modèle entraîné"""
        return st.session_state.training.trained_model
    
    @trained_model.setter
    def trained_model(self, value: Any):
        st.session_state.training.trained_model = value
    
    @property
    def training_history(self) -> Optional[Dict[str, Any]]:
        """Historique d'entraînement"""
        return st.session_state.training.training_history
    
    @training_history.setter
    def training_history(self, value: Optional[Dict[str, Any]]):
        st.session_state.training.training_history = value
    
    @property
    def preprocessor(self) -> Any:
        """Préprocesseur"""
        return st.session_state.training.preprocessor
    
    @preprocessor.setter
    def preprocessor(self, value: Any):
        st.session_state.training.preprocessor = value
    
    @property
    def class_weights(self) -> Optional[Dict]:
        """Poids de classes"""
        return st.session_state.training.class_weights
    
    @class_weights.setter
    def class_weights(self, value: Optional[Dict]):
        st.session_state.training.class_weights = value
    
    @property
    def ml_results(self) -> Optional[List[Dict]]:
        """Résultats ML (format legacy)"""
        return st.session_state.training.ml_results
    
    @ml_results.setter
    def ml_results(self, value: Optional[List[Dict]]):
        st.session_state.training.ml_results = value
    
    @property
    def current_experiment(self) -> Optional[str]:
        """Nom de l'expérience courante"""
        return st.session_state.training.current_experiment
    
    @current_experiment.setter
    def current_experiment(self, value: Optional[str]):
        st.session_state.training.current_experiment = value
    
    # --- 🆕 MLflow (v5.0) ---
    @property
    def mlflow_runs(self) -> List[Dict[str, Any]]:
        """Liste des runs MLflow"""
        return st.session_state.training.mlflow_runs
    
    @mlflow_runs.setter
    def mlflow_runs(self, value: List[Dict[str, Any]]):
        """
        Setter avec validation.
        Préférer add_mlflow_run() pour ajout individuel.
        """
        if not isinstance(value, list):
            logger.warning(f"⚠️ mlflow_runs doit être une liste, reçu: {type(value)}")
            value = []
        st.session_state.training.mlflow_runs = value
    
    @property
    def mlflow_experiment_id(self) -> Optional[str]:
        """ID de l'expérience MLflow courante"""
        return st.session_state.training.mlflow_experiment_id
    
    @mlflow_experiment_id.setter
    def mlflow_experiment_id(self, value: Optional[str]):
        st.session_state.training.mlflow_experiment_id = value
    
    @property
    def mlflow_experiment_name(self) -> str:
        """Nom de l'expérience MLflow"""
        return st.session_state.training.mlflow_experiment_name
    
    @mlflow_experiment_name.setter
    def mlflow_experiment_name(self, value: str):
        st.session_state.training.mlflow_experiment_name = value

# ========================
# 🆕 FONCTION UTILITAIRE GLOBALE DE SYNCHRONISATION
# ========================

def sync_mlflow_runs_all_sources(mlflow_runs: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Synchronise les runs MLflow sur TOUTES les sources disponibles.
    
    Cette fonction DOIT être appelée après chaque entraînement pour garantir
    que les runs sont accessibles partout dans l'application.
    
    Args:
        mlflow_runs: Liste de runs MLflow à synchroniser
        
    Returns:
        Dict avec compteurs par source:
        {
            'session_state': nb_runs_ajoutés,
            'STATE': nb_runs_ajoutés,
            'STATE.training': nb_runs_ajoutés,
            'total_synchronized': total
        }
    
    Usage:
        # Dans orchestrateur ou training.py après entraînement:
        from monitoring.state_managers import sync_mlflow_runs_all_sources
        
        mlflow_runs = [...]  # runs collectés
        sync_mlflow_runs_all_sources(mlflow_runs)
    """
    
    if not mlflow_runs:
        logger.warning("⚠️ Aucun run MLflow à synchroniser")
        return {
            'session_state': 0,
            'STATE': 0,
            'STATE.training': 0,
            'total_synchronized': 0
        }
    
    logger.info(f"🔄 Synchronisation de {len(mlflow_runs)} runs MLflow...")
    
    counters = {
        'session_state': 0,
        'STATE': 0,
        'STATE.training': 0,
        'total_synchronized': 0
    }
    
    # === SOURCE 1: st.session_state.mlflow_runs ===
    try:
        if not hasattr(st.session_state, 'mlflow_runs'):
            st.session_state.mlflow_runs = []
        
        existing_ids = {
            r.get('run_id') 
            for r in st.session_state.mlflow_runs 
            if r.get('run_id')
        }
        
        new_runs = [
            r for r in mlflow_runs 
            if r.get('run_id') and r.get('run_id') not in existing_ids
        ]
        
        if new_runs:
            st.session_state.mlflow_runs.extend(new_runs)
            counters['session_state'] = len(new_runs)
            logger.info(
                f"✅ {len(new_runs)} runs → session_state.mlflow_runs "
                f"(total: {len(st.session_state.mlflow_runs)})"
            )
    
    except Exception as e:
        logger.error(f"❌ Erreur sync session_state: {e}")
    
    # === SOURCE 2: STATE.mlflow_runs (via property) ===
    try:
        # Utilise le StateManager singleton global
        existing_ids = {
            r.get('run_id') 
            for r in STATE.mlflow_runs 
            if r.get('run_id')
        }
        
        new_runs = [
            r for r in mlflow_runs 
            if r.get('run_id') and r.get('run_id') not in existing_ids
        ]
        
        if new_runs:
            # Ajout via méthode pour déduplication automatique
            for run in new_runs:
                STATE.add_mlflow_run(run)
            
            counters['STATE'] = len(new_runs)
            logger.info(
                f"✅ {len(new_runs)} runs → STATE.mlflow_runs "
                f"(total: {len(STATE.mlflow_runs)})"
            )
    
    except Exception as e:
        logger.error(f"❌ Erreur sync STATE: {e}")
    
    # === SOURCE 3: STATE.training.mlflow_runs (direct) ===
    try:
        existing_ids = {
            r.get('run_id') 
            for r in st.session_state.training.mlflow_runs 
            if r.get('run_id')
        }
        
        new_runs = [
            r for r in mlflow_runs 
            if r.get('run_id') and r.get('run_id') not in existing_ids
        ]
        
        if new_runs:
            for run in new_runs:
                st.session_state.training.add_mlflow_run(run)
            
            counters['STATE.training'] = len(new_runs)
            logger.info(
                f"✅ {len(new_runs)} runs → STATE.training.mlflow_runs "
                f"(total: {len(st.session_state.training.mlflow_runs)})"
            )
    
    except Exception as e:
        logger.error(f"❌ Erreur sync STATE.training: {e}")
    
    # === CALCUL TOTAL ===
    counters['total_synchronized'] = sum([
        counters['session_state'],
        counters['STATE'],
        counters['STATE.training']
    ])
    
    if counters['total_synchronized'] > 0:
        logger.info(
            f"✅ Synchronisation MLflow terminée: "
            f"{counters['total_synchronized']} runs ajoutés au total"
        )
    else:
        logger.info("ℹ️ Aucun nouveau run à synchroniser (tous déjà présents)")
    
    return counters

# ========================
# INITIALIZER FUNCTION
# ========================

def init() -> StateManager:
    """
    Initialise et retourne le StateManager global.
    
    Cette fonction DOIT être appelée au début de chaque page Streamlit.
    
    Gère:
    - Initialisation singleton
    - Vérification timeout session
    - Mise à jour métriques système
    - Touch timestamp
    
    Returns:
        Instance StateManager (singleton)
    
    Usage:
        from monitoring.state_managers import init, STATE
        
        STATE = init()  # en début de page
        
        # Ou utiliser directement STATE global:
        from monitoring.state_managers import STATE
    """
    
    # Récupération/création singleton
    sm = StateManager()
    now = time.time()

    # Vérification que 'metrics' existe
    if "metrics" not in st.session_state:
        logger.warning("⚠️ metrics absent → réinitialisation complète")
        sm.reset_all()
    else:
        # Vérification timeout session
        last_active = st.session_state.metrics.last_active
        timeout_seconds = sm.config.TIMEOUT_MIN * 60
        
        if now - last_active > timeout_seconds:
            logger.warning(
                f"⏱️ Session expirée "
                f"({(now - last_active) / 60:.1f} min > {sm.config.TIMEOUT_MIN} min) "
                f"→ reset complet"
            )
            sm.reset_all()

    # Touch timestamp
    sm.touch()

    # Mise à jour métriques système
    try:
        import psutil # type: ignore
        st.session_state.metrics.memory_percent = psutil.virtual_memory().percent
    except ImportError:
        logger.debug("psutil non disponible (métriques mémoire désactivées)")
    except Exception as e:
        logger.warning(f"⚠️ Erreur lecture métriques système: {e}")

    return sm

# ========================
# GLOBAL INSTANCE (Singleton)
# ========================

STATE = StateManager()

# ========================
# EXPORTS
# ========================

__all__ = [
    # Classes principales
    'StateManager',
    'DataType',
    'AppPage',
    'TrainingStep',
    
    # États
    'DataState',
    'TrainingState',
    'NavState',
    'MetricsState',
    
    # Instance globale
    'STATE',
    
    # Fonctions utilitaires
    'init',
    'sync_mlflow_runs_all_sources',
]

# ========================
# 🆕 HELPERS DEBUGGING (développement)
# ========================

def debug_state_info() -> Dict[str, Any]:
    """
    Retourne un snapshot de l'état actuel pour debugging.
    
    Returns:
        Dict avec informations de debug
    """
    try:
        return {
            'loaded': STATE.loaded,
            'data_type': STATE.dtype.value if STATE.dtype else None,
            'data_name': STATE.data.name,
            'current_page': STATE.page.value if STATE.page else None,
            'current_step': STATE.current_step,
            'workflow_complete': STATE.workflow_complete,
            'selected_models': STATE.selected_models,
            'mlflow_runs_count': len(STATE.mlflow_runs),
            'training_results_present': STATE.training_results is not None,
            'session_age_seconds': time.time() - STATE.metrics.start,
            'memory_percent': STATE.metrics.memory_percent,
            'calc_stats': {
                'total': STATE.metrics.calc.total,
                'failed': STATE.metrics.calc.failed,
                'avg_duration': STATE.metrics.calc.avg,
                'active': STATE.metrics.calc.active
            }
        }
    except Exception as e:
        logger.error(f"❌ Erreur debug_state_info: {e}")
        return {'error': str(e)}

def validate_mlflow_consistency() -> Dict[str, Any]:
    """
    Valide la cohérence des runs MLflow entre les sources.
    
    Returns:
        Dict avec résultat de validation
    """
    try:
        sources = {}
        
        # Session state
        if hasattr(st.session_state, 'mlflow_runs'):
            sources['session_state'] = len(st.session_state.mlflow_runs)
        else:
            sources['session_state'] = 0
        
        # STATE global
        try:
            sources['STATE'] = len(STATE.mlflow_runs)
        except:
            sources['STATE'] = 0
        
        # STATE.training
        try:
            sources['STATE.training'] = len(STATE.training.mlflow_runs)
        except:
            sources['STATE.training'] = 0
        
        # Analyse
        max_count = max(sources.values())
        min_count = min(sources.values())
        
        is_consistent = (max_count == min_count)
        
        return {
            'is_consistent': is_consistent,
            'sources': sources,
            'max_count': max_count,
            'min_count': min_count,
            'recommendation': (
                "✅ Toutes les sources sont synchronisées" 
                if is_consistent 
                else f"⚠️ Incohérence détectée: appeler sync_mlflow_runs_all_sources()"
            )
        }
    
    except Exception as e:
        logger.error(f"❌ Erreur validate_mlflow_consistency: {e}")
        return {'error': str(e)}

# ========================
# 🆕 DECORATORS UTILITAIRES
# ========================

def require_data(func):
    """
    Décorateur: vérifie que des données sont chargées.
    
    Usage:
        @require_data
        def ma_fonction():
            # code qui nécessite des données
    """
    def wrapper(*args, **kwargs):
        is_valid, message = STATE.validate()
        if not is_valid:
            st.error(f"❌ Données requises: {message}")
            logger.warning(f"require_data failed: {message}")
            return None
        return func(*args, **kwargs)
    return wrapper

def require_tabular(func):
    """
    Décorateur: vérifie que des données tabulaires sont chargées.
    
    Usage:
        @require_tabular
        def analyse_ml():
            # code ML classique
    """
    def wrapper(*args, **kwargs):
        if not STATE.tabular:
            st.error("❌ Cette fonction nécessite des données tabulaires")
            logger.warning("require_tabular failed")
            return None
        return func(*args, **kwargs)
    return wrapper

def require_images(func):
    """
    Décorateur: vérifie que des images sont chargées.
    
    Usage:
        @require_images
        def train_cnn():
            # code Computer Vision
    """
    def wrapper(*args, **kwargs):
        if not STATE.images:
            st.error("❌ Cette fonction nécessite des images")
            logger.warning("require_images failed")
            return None
        return func(*args, **kwargs)
    return wrapper

# ========================
# LOGGING STARTUP
# ========================

logger.info(
    "StateManager v1.0.0 chargé | "
    f"Features: MLflow multi-sources, Thread-safe, "
    f"Debugging helpers"
)
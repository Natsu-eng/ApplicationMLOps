"""
DataLab Pro - State Manager Global
Version: 4.0 | Production-Ready | Thread-Safe | Fixed
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

# Import du système de logging centralisé
from src.shared.logging import get_logger

# ========================
# LOGGER
# ========================
logger = get_logger(__name__)

# ========================
# ENUMS
# ========================
class DataType(Enum):
    NONE = "none"
    TABULAR = "tabular"
    IMAGES = "images"

class AppPage(Enum):
    HOME = "main.py"
    DASHBOARD = "pages/1_dashboard.py"
    ML_TRAINING = "pages/2_training.py"
    ML_EVALUATION = "pages/3_evaluation.py"
    CV_TRAINING = "pages/4_training_computer.py"
    ANOMALY_EVAL = "pages/5_anomaly_evaluation.py"

class TrainingStep(Enum):
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
    MAX_FILE_MB: int = 500
    SUPPORTED_EXT: Set[str] = field(default_factory=lambda: {'.csv', '.xlsx', '.parquet', '.feather', '.json'})
    TIMEOUT_MIN: int = 60
    MAX_MESSAGE_SIZE: int = 500  # MB

@dataclass
class CalcStats:
    total: int = 0
    failed: int = 0
    last: float = 0.0
    avg: float = 0.0
    active: int = 0
    history: list = field(default_factory=list)
    max_hist: int = 100

# ========================
# STATE CLASSES - COMPLÈTES
# ========================
@dataclass
class NavState:
    current: AppPage = AppPage.HOME
    authorized: Set[AppPage] = field(default_factory=set)
    last_active: float = field(default_factory=time.time)
    current_step: int = 0  # AJOUT CRITIQUE

@dataclass
class DataState:
    loaded: bool = False
    type: DataType = DataType.NONE
    name: Optional[str] = None
    loaded_at: Optional[float] = None

    # Tabulaires
    df: Optional[pd.DataFrame] = None
    df_raw: Optional[pd.DataFrame] = None
    feature_list: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    task_type: Optional[str] = None

    # Images
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
    # Configuration commune
    current_step: int = 0
    workflow_complete: bool = False
    selected_models: List[str] = field(default_factory=list)
    test_size: int = 20
    optimize_hyperparams: bool = False
    
    # Configuration Computer Vision
    selected_model_type: Optional[str] = None
    model_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)
    imbalance_config: Dict[str, Any] = field(default_factory=dict)
    current_experiment: Optional[str] = None
    
    # Résultats
    training_results: Optional[Dict[str, Any]] = None
    trained_model: Any = None
    training_history: Optional[Dict[str, Any]] = None
    preprocessor: Any = None
    class_weights: Optional[Dict] = None
    ml_results: Optional[Dict] = None

    # Stockage des runs MLflow
    mlflow_runs: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class MetricsState:
    start: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    errors: int = 0
    success: int = 0
    memory_percent: float = 0.0
    calc: CalcStats = field(default_factory=CalcStats)

# ========================
# GLOBAL STATE MANAGER (Singleton Renforcé)
# ========================
class StateManager:
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if not self._initialized:
            self.config = Config()
            self._setup_session()
            self._initialized = True
            logger.info("StateManager initialisé")

    def _setup_session(self):
        """Initialisation sécurisée de tous les états"""
        with self._lock:
            defaults = {
                'nav': NavState(),
                'data': DataState(),
                'training': TrainingState(),
                'metrics': MetricsState(),
                'ready': True,
                'time': time.strftime('%H:%M:%S')
            }
            
            for k, v in defaults.items():
                if k not in st.session_state:
                    st.session_state[k] = v
            
            self._update_auth()

    # --- AUTH ---
    def _update_auth(self):
        t = st.session_state.data.type
        auth = {AppPage.HOME}
        if t == DataType.TABULAR:
            auth.update({AppPage.DASHBOARD, AppPage.ML_TRAINING, AppPage.ML_EVALUATION})
        elif t == DataType.IMAGES:
            auth.update({AppPage.DASHBOARD, AppPage.CV_TRAINING, AppPage.ANOMALY_EVAL})
        st.session_state.nav.authorized = auth

    # --- NAVIGATION ---
    def go(self, page: AppPage) -> bool:
        if page not in st.session_state.nav.authorized:
            logger.warning(f"Accès refusé: {page.value}")
            st.warning(f"⚠️ Accès refusé à {page.name}")
            return False
        st.session_state.nav.current = page
        st.session_state.nav.last_active = time.time()
        logger.info(f"→ {page.value}")
        return True

    def switch(self, page: AppPage) -> bool:
        if not self.go(page):
            return False
        try:
            st.switch_page(page.value)
            return True
        except Exception as e:
            logger.error(f"switch_page failed: {e}")
            st.error(f"Erreur de navigation: {e}")
            return False

    # --- DATA SETTERS ---
    def set_tabular(self, df: pd.DataFrame, df_raw: pd.DataFrame, name: str) -> bool:
        try:
            if df is None or df.empty:
                return False
            self.reset_data()
            d = st.session_state.data
            d.df = df.copy()
            d.df_raw = df_raw.copy() if df_raw is not None else df.copy()
            d.name = name
            d.type = DataType.TABULAR
            d.loaded = True
            d.loaded_at = time.time()
            self._update_auth()
            logger.info(f"Tabular loaded: {name} | {df.shape}")
            return True
        except Exception as e:
            logger.error(f"set_tabular: {e}")
            st.error(f"Erreur chargement: {e}")
            return False

    def set_images(self, X, X_norm, y, dir_path, structure, info) -> bool:
        try:
            if len(X) == 0 or len(X) != len(y):
                return False
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
            d.img_shape = X.shape
            d.n_classes = len(np.unique(y))
            d.task = "anomaly_detection" if len(np.unique(y)) == 2 and set(np.unique(y)) == {0,1} else "classification"
            self._update_auth()
            logger.info(f"Images loaded: {len(X)} | {d.n_classes} classes")
            return True
        except Exception as e:
            logger.error(f"set_images: {e}")
            st.error(f"Erreur chargement images: {e}")
            return False

    def reset_data(self):
        """Reset sécurisé des données"""
        st.session_state.data = DataState()
        st.session_state.training = TrainingState()  # Reset training aussi
        self._update_auth()
        logger.info("Data reset")

    def reset_all(self):
        """Reset complet sécurisé"""
        st.session_state.data = DataState()
        st.session_state.nav = NavState()
        st.session_state.training = TrainingState()
        st.session_state.metrics = MetricsState()
        self._update_auth()
        logger.info("Full reset")

    # --- VALIDATION ---
    def validate(self) -> Tuple[bool, str]:
        d = st.session_state.data
        if not d.loaded:
            return False, "Aucune donnée"
        if d.type == DataType.TABULAR:
            if d.df is None or d.df.empty:
                return False, "DataFrame vide"
            return True, f"{len(d.df):,} lignes"
        if d.type == DataType.IMAGES:
            if d.X is None or len(d.X) == 0:
                return False, "Aucune image"
            if len(d.X) != len(d.y):
                return False, "X ≠ y"
            return True, f"{len(d.X):,} images"
        return False, "Type inconnu"

    # --- CALCULATIONS ---
    @contextmanager
    def calc(self, name: str = "op"):
        start = time.time()
        s = st.session_state.metrics.calc
        with self._lock:
            s.active += 1
            s.total += 1
        try:
            yield
        except Exception as e:
            with self._lock:
                s.failed += 1
                st.session_state.metrics.errors += 1
            logger.error(f"Calc error [{name}]: {e}")
            raise
        finally:
            dur = time.time() - start
            with self._lock:
                s.active -= 1
                s.last = dur
                s.history.append(dur)
                if len(s.history) > s.max_hist:
                    s.history.pop(0)
                s.avg = sum(s.history) / len(s.history) if s.history else 0
                st.session_state.metrics.success += 1

    # --- ACTIVITY ---
    def touch(self):
        st.session_state.metrics.last_active = time.time()
        st.session_state.time = time.strftime('%H:%M:%S')

    def get(self, key: str, default: Any = None) -> Any:
        """Accès sécurisé aux données de session"""
        return st.session_state.get(key, default)

    # --- PROPS (Accès unifié et sécurisé) ---
    @property
    def page(self): 
        return st.session_state.nav.current
    
    @property
    def dtype(self): 
        return st.session_state.data.type
    
    @property
    def loaded(self): 
        return st.session_state.data.loaded
    
    @property
    def tabular(self): 
        return self.loaded and self.dtype == DataType.TABULAR
    
    @property
    def images(self): 
        return self.loaded and self.dtype == DataType.IMAGES
    
    @property
    def data(self):
        return st.session_state.data
    
    @property
    def nav(self):
        return st.session_state.nav
    
    @property
    def metrics(self):
        return st.session_state.metrics
    
    @property
    def training(self):
        return st.session_state.training
    
    # --- PROPS SPÉCIFIQUES POUR COMPATIBILITÉ ---
    @property
    def current_step(self):
        """Accès unifié au current_step depuis training state"""
        return st.session_state.training.current_step
    
    @current_step.setter
    def current_step(self, value):
        st.session_state.training.current_step = value
    
    @property
    def selected_model_type(self):
        return st.session_state.training.selected_model_type
    
    @selected_model_type.setter
    def selected_model_type(self, value):
        st.session_state.training.selected_model_type = value
    
    @property
    def model_config(self):
        return st.session_state.training.model_config
    
    @model_config.setter
    def model_config(self, value):
        st.session_state.training.model_config = value
    
    @property
    def training_config(self):
        return st.session_state.training.training_config
    
    @training_config.setter
    def training_config(self, value):
        st.session_state.training.training_config = value
    
    @property
    def preprocessing_config(self):
        return st.session_state.training.preprocessing_config
    
    @preprocessing_config.setter
    def preprocessing_config(self, value):
        st.session_state.training.preprocessing_config = value
    
    @property
    def imbalance_config(self):
        return st.session_state.training.imbalance_config
    
    @imbalance_config.setter
    def imbalance_config(self, value):
        st.session_state.training.imbalance_config = value
    
    @property
    def current_experiment(self):
        return st.session_state.training.current_experiment
    
    @current_experiment.setter
    def current_experiment(self, value):
        st.session_state.training.current_experiment = value
    
    @property
    def training_results(self):
        return st.session_state.training.training_results
    
    @training_results.setter
    def training_results(self, value):
        st.session_state.training.training_results = value
    
    @property
    def trained_model(self):
        return st.session_state.training.trained_model
    
    @trained_model.setter
    def trained_model(self, value):
        st.session_state.training.trained_model = value
    
    @property
    def training_history(self):
        return st.session_state.training.training_history
    
    @training_history.setter
    def training_history(self, value):
        st.session_state.training.training_history = value
    
    @property
    def preprocessor(self):
        return st.session_state.training.preprocessor
    
    @preprocessor.setter
    def preprocessor(self, value):
        st.session_state.training.preprocessor = value
    
    @property
    def class_weights(self):
        return st.session_state.training.class_weights
    
    @class_weights.setter
    def class_weights(self, value):
        st.session_state.training.class_weights = value
    
    @property
    def ml_results(self):
        return st.session_state.training.ml_results
    
    @ml_results.setter
    def ml_results(self, value):
        st.session_state.training.ml_results = value
    
    @property
    def workflow_complete(self):
        return st.session_state.training.workflow_complete
    
    @workflow_complete.setter
    def workflow_complete(self, value):
        st.session_state.training.workflow_complete = value
    
    @property
    def selected_models(self):
        return st.session_state.training.selected_models
    
    @selected_models.setter
    def selected_models(self, value):
        st.session_state.training.selected_models = value
    
    @property
    def test_size(self):
        return st.session_state.training.test_size
    
    @test_size.setter
    def test_size(self, value):
        st.session_state.training.test_size = value
    
    @property
    def optimize_hyperparams(self):
        return st.session_state.training.optimize_hyperparams
    
    @optimize_hyperparams.setter
    def optimize_hyperparams(self, value):
        st.session_state.training.optimize_hyperparams = value
    
    @property
    def feature_list(self):
        return st.session_state.data.feature_list
    
    @feature_list.setter
    def feature_list(self, value):
        st.session_state.data.feature_list = value
    
    @property
    def target_column(self):
        return st.session_state.data.target_column
    
    @target_column.setter
    def target_column(self, value):
        st.session_state.data.target_column = value
    
    @property
    def task_type(self):
        return st.session_state.data.task_type
    
    @task_type.setter
    def task_type(self, value):
        st.session_state.data.task_type = value


    @property
    def mlflow_runs(self):
        return st.session_state.training.mlflow_runs
    
    @mlflow_runs.setter
    def mlflow_runs(self, value):
        st.session_state.training.mlflow_runs = value

# ========================
# INITIALIZER
# ========================
def init() -> StateManager:
    sm = StateManager()
    now = time.time()

    # ✅ Vérifie que 'metrics' existe avant de l'utiliser
    if "metrics" not in st.session_state:
        logger.warning("⚠️ metrics absent du session_state → réinitialisation complète")
        sm.reset_all()
    else:
        # Vérifie si la session est expirée
        if now - st.session_state.metrics.last_active > sm.config.TIMEOUT_MIN * 60:
            logger.warning("⏱️ Session expirée → reset complet")
            sm.reset_all()

    sm.touch()

    # Mise à jour des métriques mémoire
    try:
        import psutil
        st.session_state.metrics.memory_percent = psutil.virtual_memory().percent
    except Exception as e:
        logger.warning(f"psutil non disponible: {e}")

    return sm

# GLOBAL INSTANCE
STATE = StateManager()
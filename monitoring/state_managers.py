"""
Gestionnaires d'état pour différentes parties de l'application.
"""
import threading
import time
from typing import Dict, Any
from contextlib import contextmanager

class MetricsStateManager:
    """Gestionnaire d'état pour les calculs de métriques."""
    
    def __init__(self):
        self._calculation_lock = threading.RLock()
        self._active_calculations = 0
        self._calculation_stats = {
            "total_calculations": 0,
            "failed_calculations": 0,
            "last_calculation_time": None
        }
    
    @contextmanager
    def calculation_context(self):
        """Context manager pour suivre les calculs."""
        with self._calculation_lock:
            self._active_calculations += 1
            self._calculation_stats["total_calculations"] += 1
            start_time = time.time()
            
            try:
                yield
            except Exception:
                self._calculation_stats["failed_calculations"] += 1
                raise
            finally:
                self._active_calculations -= 1
                self._calculation_stats["last_calculation_time"] = time.time() - start_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de calcul."""
        with self._calculation_lock:
            return self._calculation_stats.copy()

# Instance globale
METRICS_STATE = MetricsStateManager()


import streamlit as st
from typing import Dict, Any, List
class DashboardStateManager:
    """Gestion centralisée de l'état du dashboard""" 
    REQUIRED_KEYS = {
        'column_types': (dict, type(None)),
        'rename_list': list,
        'columns_to_drop': list,
        'useless_candidates': list,
        'dataset_hash': (str, type(None)),
        'last_memory_check': (int, float),
        'dashboard_version': int,
        'selected_univar_col': (str, type(None)),
        'selected_bivar_col1': (str, type(None)),
        'selected_bivar_col2': (str, type(None))
    }
    
    @classmethod
    def initialize(cls):
        """Initialise l'état avec validation"""
        defaults = {
            'column_types': None,
            'rename_list': [],
            'columns_to_drop': [],
            'useless_candidates': [],
            'dataset_hash': None,
            'last_memory_check': 0,
            'dashboard_version': 1,
            'selected_univar_col': None,
            'selected_bivar_col1': None,
            'selected_bivar_col2': None
        }
        
        for key, expected_type in cls.REQUIRED_KEYS.items():
            if key not in st.session_state:
                st.session_state[key] = defaults[key]
            elif not isinstance(st.session_state[key], expected_type):
                print(f"Invalid type for {key}, resetting")
                st.session_state[key] = defaults[key]
    
    @classmethod
    def reset_selections(cls):
        """Reset les sélections pour éviter les erreurs"""
        selection_keys = ['selected_univar_col', 'selected_bivar_col1', 'selected_bivar_col2']
        for key in selection_keys:
            st.session_state[key] = None

class MLStateManager:
    """Gestion de l'état pour la configuration ML"""
    
    @staticmethod
    def initialize_ml_config():
        """Initialise l'état de configuration ML"""
        defaults = {
            'target_column_for_ml_config': None,
            'feature_list_for_ml_config': [],
            'preprocessing_choices': {
                'numeric_imputation': 'mean',
                'categorical_imputation': 'most_frequent',
                'use_smote': False,
                'remove_constant_cols': True,
                'remove_identifier_cols': True,
                'scale_features': True,
                'pca_preprocessing': False
            },
            'selected_models_for_training': [],
            'test_split_for_ml_config': 20,
            'optimize_hp_for_ml_config': False,
            'task_type': 'classification',
            'ml_training_in_progress': False,
            'ml_last_training_time': None
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
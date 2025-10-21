"""
Page Streamlit: Entraînement Computer Vision
"""

from typing import Any, Dict, Optional, Tuple, Union
import streamlit as st
import numpy as np
import time
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import sys
import os
import torch
import torch.nn as nn

# Ajout du chemin pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Imports cohérents avec computer_vision_training.py
from src.models.computer_vision_training import (
    ComputerVisionTrainer,
    AnomalyAwareTrainer,
    ModelConfig,
    TrainingConfig,
    ModelType,
    OptimizerType,
    SchedulerType,
    DataAugmenter,
    MLflowIntegration
)
from src.data.computer_vision_preprocessing import DataPreprocessor, DataValidator
from src.shared.logging import StructuredLogger
from utils.callbacks import LoggingCallback, StreamlitCallback
from utils.device_manager import DeviceManager

logger = StructuredLogger(__name__)

# Configuration Streamlit
st.set_page_config(
    page_title="CV Training | DataLab Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS moderne
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .step-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .imbalance-warning {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .imbalance-moderate {
        background: linear-gradient(135deg, #ff9ff3 0%, #f368e0 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .imbalance-good {
        background: linear-gradient(135deg, #00d2d3 0%, #54a0ff 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .config-error {
        background: linear-gradient(135deg, #ff7979 0%, #eb4d4b 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Setup navigation (optionnel, avec fallback)
try:
    from helpers.navigation_manager import setup_navigation
    setup_navigation()
except ImportError:
    logger.warning("Navigation manager non disponible, passage au mode par défaut")
    pass

# Titre principal
st.title("🚀 Entraînement Computer Vision")
st.markdown("**Wizard guidé pour entraîner votre modèle de détection d'anomalies**")


# ============================================================================
# FONCTIONS HELPERS POUR TRAININGCONFIG
# ============================================================================

def training_config_to_dict(config: Union[TrainingConfig, Dict]) -> Dict:
    """
    Convertit un TrainingConfig en dictionnaire pour affichage.
    
    Args:
        config: TrainingConfig ou Dict
        
    Returns:
        Dict avec tous les paramètres
    """
    if isinstance(config, TrainingConfig):
        return {
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'gradient_clip': config.gradient_clip,
            'optimizer': config.optimizer.value if hasattr(config.optimizer, 'value') else str(config.optimizer),
            'scheduler': config.scheduler.value if hasattr(config.scheduler, 'value') else str(config.scheduler),
            'early_stopping_patience': config.early_stopping_patience,
            'reduce_lr_patience': config.reduce_lr_patience,
            'use_class_weights': config.use_class_weights,
            'deterministic': config.deterministic,
            'seed': config.seed,
            'num_workers': config.num_workers,
            'pin_memory': config.pin_memory,
            'use_mixed_precision': config.use_mixed_precision
        }
    return config if isinstance(config, dict) else {}


def get_training_config_value(config: Union[TrainingConfig, Dict], key: str, default=None):
    """
    Accès sécurisé aux valeurs de config (compatible dict et dataclass).
    
    Args:
        config: TrainingConfig ou Dict
        key: Clé à récupérer
        default: Valeur par défaut
        
    Returns:
        Valeur demandée ou default
    """
    if isinstance(config, TrainingConfig):
        return getattr(config, key, default)
    elif isinstance(config, dict):
        return config.get(key, default)
    return default


# ============================================================================
# FONCTIONS SPÉCIFIQUES À L'UI
# ============================================================================

def plot_class_distribution(label_counts, percentages):
    """Crée un graphique de la distribution des classes."""
    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    
    # Labels intelligents
    if len(labels) == 2 and set(labels) == {0, 1}:
        label_names = ['Normal', 'Anomalie']
    else:
        label_names = [f"Classe {label}" for label in labels]
    
    colors = ['#2ecc71', '#e74c3c'] if len(labels) == 2 else px.colors.qualitative.Set3[:len(labels)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=label_names,
            y=counts,
            text=[f"{count}<br>({perc:.1f}%)" for count, perc in zip(counts, percentages)],
            textposition='auto',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{text}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Distribution des Classes",
        xaxis_title="Classes",
        yaxis_title="Nombre d'images",
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    return fig


def analyze_class_imbalance(y):
    """Analyse le déséquilibre des classes."""
    result = DataValidator.check_class_imbalance(y)
    label_counts = Counter(y)
    
    return {
        'label_counts': label_counts,
        'percentages': [label_counts[k] / len(y) * 100 for k in sorted(label_counts.keys())],
        'imbalance_ratio': result['ratio'],
        'imbalance_level': result['severity']
    }


def compute_automatic_class_weights(y):
    """Calcule les poids des classes pour déséquilibre."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return {
        "dict": dict(zip(classes, weights)), 
        "tensor": torch.tensor(weights, dtype=torch.float32)
    }


def validate_training_configuration():
    """Valide la configuration complète avant lancement."""
    errors = []
    warnings = []
    
    # Vérification des données
    required_data = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
    for data_key in required_data:
        if data_key not in st.session_state:
            errors.append(f"❌ Données manquantes: {data_key}")
    
    if 'model_config' not in st.session_state or not st.session_state.model_config:
        errors.append("❌ Configuration du modèle manquante")
    
    if 'training_config' not in st.session_state or not st.session_state.training_config:
        errors.append("❌ Configuration d'entraînement manquante")
    
    # ✅ CORRECTION: Récupération sécurisée des valeurs
    training_config = st.session_state.get('training_config')
    
    if training_config:
        epochs = get_training_config_value(training_config, 'epochs', 100)
        batch_size = get_training_config_value(training_config, 'batch_size', 32)
        
        # Vérifications spécifiques
        if epochs > 100:
            warnings.append("⚠️ Nombre d'époques élevé - entraînement potentiellement long")
        if batch_size > 64:
            warnings.append("⚠️ Batch size élevé - risque de mémoire insuffisante")
    
    # Validation des données avec DataValidator
    for dataset_key, name in [('X_train', 'train'), ('X_val', 'val'), ('X_test', 'test')]:
        dataset = st.session_state.get(dataset_key)
        labels = st.session_state.get(f'y_{name}')
        
        if dataset is not None and labels is not None:
            val_result = DataValidator.validate_input_data(dataset, labels, name)
            if not val_result.success:
                errors.append(f"❌ Validation {name}: {val_result.error}")
    
    return errors, warnings


def get_default_model_params(model_type):
    """Retourne les paramètres par défaut selon le type de modèle."""
    if model_type == ModelType.CONV_AUTOENCODER.value:
        return {"latent_dim": 256, "base_filters": 32, "num_stages": 4}
    elif model_type == ModelType.SIMPLE_CNN.value:
        return {"dropout_rate": 0.5, "base_filters": 32}
    elif model_type == ModelType.TRANSFER_LEARNING.value:
        return {"pretrained": True, "freeze_layers": 0, "dropout_rate": 0.5}
    else:
        return {}


def validate_anomaly_trainer_config():
    """Valide la configuration pour AnomalyAwareTrainer"""
    try:
        # Vérifier que training_config est bien un objet TrainingConfig
        training_config = st.session_state.get('training_config')
        if not isinstance(training_config, TrainingConfig):
            st.error("❌ training_config doit être une instance de TrainingConfig")
            return False
            
        # Vérifier model_config
        model_config = st.session_state.get('model_config', {})
        if not model_config:
            st.error("❌ model_config manquant")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False

# ============================================================================
# WRAPPER POUR L'ENTRAÎNEMENT
# ============================================================================

def train_computer_vision_model_production(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: Union[str, ModelType] = "simple_cnn",
    model_params: Dict[str, Any] = None,
    training_config: Union[Dict[str, Any], TrainingConfig] = None,
    streamlit_components: Dict = None,
    imbalance_config: Dict[str, Any] = None,
    anomaly_type: str = None
) -> Tuple[Optional[nn.Module], Dict]:
    """
    Fonction wrapper pour compatibilité avec l'ancien code.   
    Utilise le nouveau pipeline mais retourne le format attendu.
    """
    try:
        model_params = model_params or {}
        imbalance_config = imbalance_config or {}

        # Gestion de training_config
        if isinstance(training_config, TrainingConfig):
            train_config = training_config
        elif isinstance(training_config, dict):
            train_config = TrainingConfig(
                epochs=training_config.get('epochs', 100),
                batch_size=training_config.get('batch_size', 32),
                learning_rate=training_config.get('learning_rate', 1e-4),
                weight_decay=training_config.get('weight_decay', 0.01),
                gradient_clip=training_config.get('gradient_clip', 1.0),
                optimizer=OptimizerType(training_config.get('optimizer', 'adamw')),
                scheduler=SchedulerType(training_config.get('scheduler', 'reduce_on_plateau')),
                early_stopping_patience=training_config.get('early_stopping_patience', 15),
                reduce_lr_patience=training_config.get('reduce_lr_patience', 8),
                use_class_weights=imbalance_config.get('use_class_weights', False),
                deterministic=training_config.get('deterministic', True),
                seed=training_config.get('seed', 42)
            )
        else:
            train_config = TrainingConfig(
                use_class_weights=imbalance_config.get('use_class_weights', False)
            )

        # ===========================
        # Gestion robuste de model_type
        # ===========================
        if isinstance(model_type, dict):
            # Cas où model_type est en réalité le model_config de session_state
            model_config_dict = model_type
            model_type = model_config_dict.get('model_type')
            model_params = model_config_dict.get('model_params', {})

        # Conversion automatique string -> Enum
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        elif not isinstance(model_type, ModelType):
            raise ValueError(f"Type de modèle invalide: {model_type}")

        # Filtrer les paramètres valides
        allowed_params = [
            'num_classes', 'input_channels', 'dropout_rate', 
            'base_filters', 'latent_dim', 'num_stages', 
            'pretrained', 'freeze_layers'
        ]
        valid_model_params = {k: v for k, v in model_params.items() if k in allowed_params}

        # Création de ModelConfig
        model_config = ModelConfig(
            model_type=model_type,
            num_classes=valid_model_params.get('num_classes', len(np.unique(y_train))),
            input_channels=valid_model_params.get('input_channels', X_train.shape[-1] if len(X_train.shape) > 3 else 3),
            dropout_rate=valid_model_params.get('dropout_rate', 0.5),
            base_filters=valid_model_params.get('base_filters', 32),
            latent_dim=valid_model_params.get('latent_dim', 256),
            num_stages=valid_model_params.get('num_stages', 4)
        )

        # ===========================
        # Callbacks
        # ===========================
        callbacks = []
        if streamlit_components:
            callbacks.append(StreamlitCallback(
                progress_bar=streamlit_components.get('progress_bar'),
                status_text=streamlit_components.get('status_text'),
                total_epochs=train_config.epochs
            ))
        callbacks.append(LoggingCallback(log_every_n_epochs=5))

        # ===========================
        # Entraînement
        # ===========================
        if anomaly_type:
            trainer = AnomalyAwareTrainer(
                anomaly_type=anomaly_type,
                model_config=model_config,
                training_config=train_config,
                taxonomy_config=None,
                callbacks=callbacks
            )
            result = trainer.train(X_train, y_train, X_val, y_val)
        else:
            trainer = ComputerVisionTrainer(
                model_config=model_config,
                training_config=train_config,
                callbacks=callbacks
            )
            result = trainer.fit(X_train, y_train, X_val, y_val)

        if not result.success:
            logger.error(f"Entraînement échoué: {result.error}")
            return None, {
                'success': False, 
                'error': result.error, 
                'train_loss': [], 
                'val_loss': []
            }

        # ===========================
        # Historique et retour
        # ===========================
        training_config_dict = training_config_to_dict(train_config)
        history = {
            'success': True,
            'train_loss': trainer.history['train_loss'],
            'val_loss': trainer.history['val_loss'],
            'val_accuracy': trainer.history.get('val_accuracy', []),
            'val_f1': trainer.history.get('val_f1', []),
            'learning_rates': trainer.history['learning_rates'],
            'best_epoch': result.metadata.get('best_epoch', 0),
            'best_val_loss': min(trainer.history['val_loss']) if trainer.history['val_loss'] else float('inf'),
            'training_time': result.metadata.get('training_time', 0),
            'total_epochs_trained': len(trainer.history['train_loss']),
            'early_stopping_triggered': len(trainer.history['train_loss']) < train_config.epochs,
            'model_type': model_type.value,  # TOUJOURS une string
            'input_shape': X_train.shape,
            'training_config': training_config_dict,
            'anomaly_type': anomaly_type
        }

        return trainer.model, history

    except Exception as e:
        logger.error(f"Erreur wrapper production: {e}", exc_info=True)
        return None, {
            'success': False, 
            'error': str(e), 
            'train_loss': [], 
            'val_loss': []
        }


# ============================================================================
# INITIALISATION SESSION_STATE
# ============================================================================

if 'X' not in st.session_state or 'y' not in st.session_state:
    st.error("📷 Aucun dataset d'images chargé")
    st.info("Retournez au dashboard pour charger un dataset.")
    if st.button("📊 Aller au Dashboard", type="primary"):
        st.switch_page("pages/1_dashboard.py")
    st.stop()

X = st.session_state["X"]
y = st.session_state["y"]

# Initialisation session_state avec valeurs par défaut
defaults = {
    'training_wizard_step': 0,
    'preprocessing_config': {
        "strategy": "standardize",
        "augmentation_enabled": False,
        "augmentation_factor": 2,
        "methods": ['flip', 'rotate']
    },
    'training_config': TrainingConfig(
        epochs=50,
        batch_size=32,
        learning_rate=1e-4,
        early_stopping_patience=10,
        reduce_lr_patience=5,
        optimizer=OptimizerType.ADAMW,
        scheduler=SchedulerType.REDUCE_ON_PLATEAU
    ),
    'model_config': {
        "model_type": ModelType.CONV_AUTOENCODER,  # Instance de ModelType
        "model_params": get_default_model_params(ModelType.CONV_AUTOENCODER.value)
    },
    'imbalance_config': {
        "use_class_weights": False,
        "use_targeted_augmentation": False,
        "augmentation_factor": 2,
        "strategy": "standardize"
    },
    'selected_model_type': ModelType.CONV_AUTOENCODER  # Instance de ModelType
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================================
# WIZARD STEPS
# ============================================================================

WIZARD_STEPS = [
    {"name": "📊 Données", "description": "Split et Aperçu"},
    {"name": "⚖️ Déséquilibre", "description": "Analyse et Correction"},
    {"name": "🎨 Prétraitement", "description": "Normalisation et Augmentation"},
    {"name": "🤖 Modèle", "description": "Architecture et Paramètres"},
    {"name": "⚙️ Entraînement", "description": "Configuration"},
    {"name": "🚀 Lancement", "description": "Démarrage"}
]

current_step = st.session_state.training_wizard_step

# Header avec progression
st.markdown("---")
col_progress, col_info = st.columns([3, 1])

with col_progress:
    st.subheader("Progression")
    progress_percentage = (current_step / len(WIZARD_STEPS)) * 100
    st.progress(current_step / len(WIZARD_STEPS))
    st.caption(f"Étape {current_step + 1} sur {len(WIZARD_STEPS)}")

with col_info:
    st.metric("📷 Images", f"{len(X):,}")
    st.metric("🎯 Classes", len(np.unique(y)))

# Indicateurs d'étapes
st.markdown("### Étapes du Workflow")
cols = st.columns(len(WIZARD_STEPS))

for idx, (col, step) in enumerate(zip(cols, WIZARD_STEPS)):
    with col:
        status_icon = "✅" if idx < current_step else "🔵" if idx == current_step else "⬜"
        status_color = "green" if idx < current_step else "blue" if idx == current_step else "gray"
        st.markdown(
            f"<div style='text-align: center; color: {status_color};'>"
            f"<h3>{status_icon}</h3>"
            f"<p><b>{step['name']}</b></p>"
            f"<small>{step['description']}</small>"
            f"</div>",
            unsafe_allow_html=True
        )

st.markdown("---")


# ============================================================================
# CONTENU DES ÉTAPES
# ============================================================================

# ÉTAPE 0: Split des données
if current_step == 0:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.header("📊 Étape 1: Préparation des Données")
    
    st.markdown("""
    **Configuration du split Train/Validation/Test**
    Cette étape divise les données en trois ensembles sans fuite (stratification).
    """)
    
    col_split1, col_split2 = st.columns(2)
    
    with col_split1:
        test_size = st.slider(
            "Taille du Test Set (%)",
            min_value=10,
            max_value=40,
            value=20,
            step=5
        )
    
    with col_split2:
        val_size = st.slider(
            "Taille du Validation Set (%)",
            min_value=10,
            max_value=30,
            value=20,
            step=5
        )
    
    # Calcul des tailles
    test_ratio = test_size / 100
    val_ratio = val_size / 100
    n_test = int(len(X) * test_ratio)
    n_train_val = len(X) - n_test
    n_val = int(n_train_val * val_ratio)
    n_train = n_train_val - n_val
    
    # Affichage des statistiques
    st.markdown("### 📈 Répartition des Données")
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<h2>{n_train:,}</h2>"
            f"<p>Images Train</p>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    with col_stat2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<h2>{n_val:,}</h2>"
            f"<p>Images Validation</p>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    with col_stat3:
        st.markdown(
            f"<div class='metric-card'>"
            f"<h2>{n_test:,}</h2>"
            f"<p>Images Test</p>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    # Effectuer le split
    if st.button("🔄 Effectuer le Split", type="primary", use_container_width=True):
        with st.spinner("Séparation des données..."):
            try:
                X_train_val, X_test, y_train_val, y_test = train_test_split(
                    X, y, test_size=test_ratio, stratify=y, random_state=42
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val, y_train_val, 
                    test_size=val_ratio / (1 - test_ratio), 
                    stratify=y_train_val, 
                    random_state=42
                )
                
                st.session_state.update({
                    "X_train": X_train,
                    "X_val": X_val,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_val": y_val,
                    "y_test": y_test,
                    "split_config": {"test_size": test_size, "val_size": val_size}
                })
                
                st.success("✅ Split effectué")
                st.session_state.training_wizard_step = 1
                st.rerun()
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")
                logger.error(f"Split error: {e}", exc_info=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


# ÉTAPE 1: Gestion du déséquilibre
elif current_step == 1:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.header("⚖️ Étape 2: Gestion du Déséquilibre")
    
    if 'y_train' not in st.session_state:
        st.error("❌ Données d'entraînement non disponibles")
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.training_wizard_step = 0
            st.rerun()
        st.stop()
    
    y_train = st.session_state.y_train
    
    with st.spinner("🔍 Analyse du déséquilibre..."):
        imbalance_stats = analyze_class_imbalance(y_train)
    
    st.markdown("### 📊 Analyse de la Distribution des Classes")
    fig = plot_class_distribution(
        imbalance_stats['label_counts'], 
        imbalance_stats['percentages']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Afficher le statut du déséquilibre
    imbalance_ratio = imbalance_stats["imbalance_ratio"]
    imbalance_level = imbalance_stats["imbalance_level"]
    
    if imbalance_level == 'critical':
        st.markdown(
            f"<div class='imbalance-warning'>"
            f"<h3>🚨 Déséquilibre Critique</h3>"
            f"<p><b>Ratio:</b> {imbalance_ratio:.1f}:1</p>"
            f"<p>Risque d'ignorer les classes minoritaires</p>"
            f"</div>",
            unsafe_allow_html=True
        )
    elif imbalance_level == 'high':
        st.markdown(
            f"<div class='imbalance-warning'>"
            f"<h3>⚠️ Déséquilibre Élevé</h3>"
            f"<p><b>Ratio:</b> {imbalance_ratio:.1f}:1</p>"
            f"<p>Correction fortement recommandée</p>"
            f"</div>",
            unsafe_allow_html=True
        )
    elif imbalance_level == 'moderate':
        st.markdown(
            f"<div class='imbalance-moderate'>"
            f"<h3>ℹ️ Déséquilibre Modéré</h3>"
            f"<p><b>Ratio:</b> {imbalance_ratio:.1f}:1</p>"
            f"<p>Correction recommandée</p>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='imbalance-good'>"
            f"<h3>✅ Dataset Équilibré</h3>"
            f"<p><b>Ratio:</b> {imbalance_ratio:.1f}:1</p>"
            f"<p>Aucune correction nécessaire</p>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    st.subheader("🎯 Stratégies de Correction")
    
    col_strat1, col_strat2 = st.columns(2)
    
    with col_strat1:
        st.markdown("#### ⚖️ Poids de Classe")
        use_class_weights = st.checkbox(
            "Utiliser les poids de classe",
            value=imbalance_ratio > 2,
            help="Ajuste la perte pour compenser le déséquilibre"
        )
        if use_class_weights:
            weights_result = compute_automatic_class_weights(y_train)
            for cls, weight in weights_result["dict"].items():
                cls_name = "Normal" if cls == 0 else "Anomalie" if len(imbalance_stats["label_counts"]) == 2 else f"Classe {cls}"
                st.write(f"- {cls_name}: {weight:.2f}")
    
    with col_strat2:
        st.markdown("#### 🎭 Augmentation Ciblée")
        use_targeted_augmentation = st.checkbox(
            "Augmenter les classes minoritaires",
            value=imbalance_ratio > 3
        )
        augmentation_factor = st.slider(
            "Facteur d'augmentation",
            min_value=2,
            max_value=5,
            value=min(4, int(imbalance_ratio)),
            disabled=not use_targeted_augmentation
        )
    
    # Boutons de navigation
    st.markdown("---")
    col_back, col_next = st.columns(2)
    
    with col_back:
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.training_wizard_step = 0
            st.rerun()
    
    with col_next:
        if st.button("Suivant ➡️", type="primary", use_container_width=True):
            st.session_state.imbalance_config = {
                "use_class_weights": use_class_weights,
                "use_targeted_augmentation": use_targeted_augmentation,
                "augmentation_factor": augmentation_factor,
                "strategy": "standardize"
            }
            st.success("✅ Configuration du déséquilibre sauvegardée")
            st.session_state.training_wizard_step = 2
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


# ÉTAPE 2: Prétraitement
elif current_step == 2:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.header("🎨 Étape 3: Prétraitement des Images")
    
    st.markdown("""
    **Configuration du prétraitement**
    Optimisez vos images pour l'entraînement.
    """)
    
    col_preprocess1, col_preprocess2 = st.columns(2)
    
    with col_preprocess1:
        st.subheader("🔧 Normalisation")
        normalization_method = st.selectbox(
            "Méthode de normalisation",
            options=["standardize", "normalize", "none"],
            index=0,
            help="Standardize: (x-mean)/std, Normalize: min-max [0,1]"
        )
    
    with col_preprocess2:
        st.subheader("📐 Redimensionnement")
        resize_option = st.selectbox(
            "Taille cible",
            options=["Conserver original", "128×128", "224×224", "256×256"],
            index=0
        )
        current_size = X.shape[1:3] if len(X.shape) > 2 else (0, 0)
        st.info(f"**Taille actuelle:** {current_size[0]}×{current_size[1]}")
    
    st.markdown("---")
    st.subheader("🎭 Augmentation de Données")
    
    augmentation_enabled = st.checkbox(
        "Activer l'augmentation",
        value=st.session_state.preprocessing_config.get("augmentation_enabled", False)
    )
    
    augmentation_factor = st.session_state.preprocessing_config.get("augmentation_factor", 2)
    methods = ['flip', 'rotate']
    
    if augmentation_enabled:
        col_aug1, col_aug2 = st.columns(2)
        
        with col_aug1:
            augmentation_factor = st.slider(
                "Facteur de multiplication",
                min_value=1,
                max_value=5,
                value=augmentation_factor
            )
        
        with col_aug2:
            st.markdown("**Méthodes:**")
            methods = []
            if st.checkbox("Flip horizontal", value=True):
                methods.append('flip')
            if st.checkbox("Rotation", value=True):
                methods.append('rotate')
            if st.checkbox("Ajout de bruit", value=False):
                methods.append('noise')
    
    # Boutons de navigation
    st.markdown("---")
    col_back, col_next = st.columns(2)
    
    with col_back:
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.training_wizard_step = 1
            st.rerun()
    
    with col_next:
        if st.button("Suivant ➡️", type="primary", use_container_width=True):
            st.session_state.preprocessing_config = {
                "strategy": normalization_method,
                "augmentation_enabled": augmentation_enabled,
                "augmentation_factor": augmentation_factor,
                "methods": methods
            }
            st.success("✅ Configuration de prétraitement sauvegardée")
            st.session_state.training_wizard_step = 3
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


# ÉTAPE 3: Configuration du modèle
elif current_step == 3:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.header("🤖 Étape 4: Configuration du Modèle")
    
    st.markdown("""
    **Choisissez l'architecture de votre modèle**
    Options adaptées à la détection d'anomalies.
    """)
    
    col_model1, col_model2, col_model3 = st.columns(3)
    
    with col_model1:
        st.markdown("""
        <div style='border: 2px solid #667eea; border-radius: 10px; padding: 1rem; height: 250px;'>
        <h3>🔄 AutoEncoder</h3>
        <p><b>Non supervisé</b></p>
        <ul>
            <li>Reconstruit les images normales</li>
            <li>Anomalies via erreur de reconstruction</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Choisir AutoEncoder", key="select_autoencoder", use_container_width=True):
            st.session_state.selected_model_type = ModelType.CONV_AUTOENCODER  # Instance de ModelType
            st.session_state.model_config = {
                "model_type": ModelType.CONV_AUTOENCODER,  # Instance de ModelType
                "model_params": get_default_model_params(ModelType.CONV_AUTOENCODER.value)
            }
            st.rerun()
    
    with col_model2:
        st.markdown("""
        <div style='border: 2px solid #764ba2; border-radius: 10px; padding: 1rem; height: 250px;'>
        <h3>🧠 CNN Classifier</h3>
        <p><b>Supervisé</b></p>
        <ul>
            <li>Classification Normal/Anomalie</li>
            <li>Labels requis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Choisir CNN Classifier", key="select_cnn", use_container_width=True):
            st.session_state.selected_model_type = ModelType.SIMPLE_CNN  # Instance de ModelType
            st.session_state.model_config = {
                "model_type": ModelType.SIMPLE_CNN,  # Instance de ModelType
                "model_params": get_default_model_params(ModelType.SIMPLE_CNN.value)
            }
            st.rerun()
    
    with col_model3:
        st.markdown("""
        <div style='border: 2px solid #f39c12; border-radius: 10px; padding: 1rem; height: 250px;'>
        <h3>🎯 Transfer Learning</h3>
        <p><b>Pré-entraîné</b></p>
        <ul>
            <li>ResNet, EfficientNet</li>
            <li>Fine-tuning rapide</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Choisir Transfer Learning", key="select_transfer", use_container_width=True):
            st.session_state.selected_model_type = ModelType.TRANSFER_LEARNING  # Instance de ModelType
            st.session_state.model_config = {
                "model_type": ModelType.TRANSFER_LEARNING,  # Instance de ModelType
                "model_params": get_default_model_params(ModelType.TRANSFER_LEARNING.value)
            }
            st.rerun()
    
    # Configuration spécifique
    if 'model_config' in st.session_state and st.session_state.model_config:
        st.markdown("---")
        st.subheader(f"⚙️ Paramètres - {st.session_state.model_config['model_type'].upper()}")
        
        model_type = st.session_state.model_config["model_type"]
        model_params = st.session_state.model_config["model_params"]
        
        if model_type == ModelType.CONV_AUTOENCODER.value:
            col_param1, col_param2 = st.columns(2)
            with col_param1:
                latent_dim = st.slider(
                    "Dimension Latente",
                    min_value=32,
                    max_value=512,
                    value=model_params.get("latent_dim", 256),
                    step=32,
                    help="Taille de l'espace latent compressé"
                )
            with col_param2:
                base_filters = st.slider(
                    "Filtres de Base",
                    min_value=16,
                    max_value=128,
                    value=model_params.get("base_filters", 32),
                    step=16,
                    help="Nombre de filtres dans la première couche"
                )
            
            num_stages = st.slider(
                "Nombre de Stages",
                min_value=2,
                max_value=5,
                value=model_params.get("num_stages", 4),
                help="Profondeur du réseau"
            )
            
            st.session_state.model_config["model_params"].update({
                "latent_dim": latent_dim,
                "base_filters": base_filters,
                "num_stages": num_stages
            })
        
        elif model_type == ModelType.SIMPLE_CNN.value:
            col_param1, col_param2 = st.columns(2)
            with col_param1:
                base_filters = st.slider(
                    "Filtres de Base",
                    min_value=16,
                    max_value=128,
                    value=model_params.get("base_filters", 32),
                    step=16
                )
            with col_param2:
                dropout_rate = st.slider(
                    "Taux de Dropout",
                    min_value=0.0,
                    max_value=0.7,
                    value=model_params.get("dropout_rate", 0.5),
                    step=0.1,
                    help="Régularisation pour éviter l'overfitting"
                )
            st.session_state.model_config["model_params"].update({
                "base_filters": base_filters,
                "dropout_rate": dropout_rate
            })
        
        elif model_type == ModelType.TRANSFER_LEARNING.value:
            col_param1, col_param2 = st.columns(2)
            with col_param1:
                pretrained = st.checkbox(
                    "Utiliser modèle pré-entraîné",
                    value=model_params.get("pretrained", True),
                    help="Utilise les poids ImageNet"
                )
            with col_param2:
                freeze_layers = st.slider(
                    "Couches gelées",
                    min_value=0,
                    max_value=200,
                    value=model_params.get("freeze_layers", 0),
                    help="Nombre de couches à ne pas entraîner"
                )
            
            dropout_rate = st.slider(
                "Taux de Dropout",
                min_value=0.0,
                max_value=0.7,
                value=model_params.get("dropout_rate", 0.5),
                step=0.1
            )
            
            st.session_state.model_config["model_params"].update({
                "pretrained": pretrained,
                "freeze_layers": freeze_layers,
                "dropout_rate": dropout_rate
            })
    
    # Boutons de navigation
    st.markdown("---")
    col_back, col_next = st.columns(2)
    
    with col_back:
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.training_wizard_step = 2
            st.rerun()
    
    with col_next:
        if st.button("Suivant ➡️", type="primary", use_container_width=True):
            st.success("✅ Configuration du modèle sauvegardée")
            st.session_state.training_wizard_step = 4
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


# ÉTAPE 4: Configuration de l'entraînement
elif current_step == 4:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.header("⚙️ Étape 5: Configuration de l'Entraînement")
    
    st.markdown("**Hyperparamètres d'entraînement**")
    
    col_hyper1, col_hyper2, col_hyper3 = st.columns(3)
    
    with col_hyper1:
        # ✅ CORRECTION: Utilisation de get_training_config_value
        epochs = st.slider(
            "Nombre d'Époques",
            min_value=5,
            max_value=200,
            value=get_training_config_value(st.session_state.training_config, 'epochs', 50),
            step=5,
            help="Nombre de passages complets sur le dataset"
        )
    
    with col_hyper2:
        # ✅ CORRECTION: Utilisation de get_training_config_value
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
            value=get_training_config_value(st.session_state.training_config, 'learning_rate', 1e-4),
            format_func=lambda x: f"{x:.0e}",
            help="Taux d'apprentissage (plus petit = plus stable)"
        )
    
    with col_hyper3:
        batch_size = st.selectbox(
            "Batch Size",
            options=[8, 16, 32, 64, 128],
            index=2,
            help="Nombre d'images par batch"
        )
    
    st.markdown("---")
    st.subheader("🎯 Optimiseur et Scheduler")
    
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        optimizer = st.selectbox(
            "Optimiseur",
            options=[opt.value for opt in OptimizerType],
            index=0,
            help="AdamW recommandé pour la plupart des cas"
        )
    
    with col_opt2:
        scheduler = st.selectbox(
            "Learning Rate Scheduler",
            options=[sched.value for sched in SchedulerType],
            index=0,
            help="ReduceLROnPlateau réduit le LR si pas d'amélioration"
        )
    
    st.markdown("---")
    st.subheader("🛑 Early Stopping & Régularisation")
    
    col_callback1, col_callback2, col_callback3 = st.columns(3)
    
    with col_callback1:
        # ✅ CORRECTION: Utilisation de get_training_config_value
        early_stopping_patience = st.slider(
            "Early Stopping Patience",
            min_value=3,
            max_value=30,
            value=get_training_config_value(st.session_state.training_config, 'early_stopping_patience', 10),
            help="Arrêt si pas d'amélioration pendant N époques"
        )
    
    with col_callback2:
        # ✅ CORRECTION: Utilisation de get_training_config_value
        reduce_lr_patience = st.slider(
            "Reduce LR Patience",
            min_value=2,
            max_value=15,
            value=get_training_config_value(st.session_state.training_config, 'reduce_lr_patience', 5),
            help="Réduction du LR si pas d'amélioration pendant N époques"
        )
    
    with col_callback3:
        weight_decay = st.select_slider(
            "Weight Decay",
            options=[0.0, 0.001, 0.01, 0.1],
            value=0.01,
            help="Régularisation L2"
        )
    
    # Options avancées
    with st.expander("🔧 Options Avancées"):
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            gradient_clip = st.slider(
                "Gradient Clipping",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.5,
                help="Limite l'amplitude des gradients"
            )
            
            deterministic = st.checkbox(
                "Mode Déterministe",
                value=True,
                help="Résultats reproductibles (seed fixé)"
            )
        
        with col_adv2:
            use_mixed_precision = st.checkbox(
                "Mixed Precision (FP16)",
                value=True,
                help="Accélère l'entraînement sur GPU récents"
            )
            
            num_workers = st.slider(
                "DataLoader Workers",
                min_value=0,
                max_value=8,
                value=4,
                help="Nombre de processus pour charger les données"
            )
    
    # Boutons de navigation
    st.markdown("---")
    col_back, col_next = st.columns(2)
    
    with col_back:
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.training_wizard_step = 3
            st.rerun()
    
    with col_next:
        if st.button("Suivant ➡️", type="primary", use_container_width=True):
            # ✅ CORRECTION: Création d'une instance de TrainingConfig
            st.session_state.training_config = TrainingConfig(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                gradient_clip=gradient_clip,
                optimizer=OptimizerType(optimizer),
                scheduler=SchedulerType(scheduler),
                early_stopping_patience=early_stopping_patience,
                reduce_lr_patience=reduce_lr_patience,
                use_class_weights=st.session_state.imbalance_config.get('use_class_weights', False),
                deterministic=deterministic,
                use_mixed_precision=use_mixed_precision,
                num_workers=num_workers,
                seed=42
            )
            st.success("✅ Configuration d'entraînement sauvegardée")
            st.session_state.training_wizard_step = 5
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


# ÉTAPE 5: Lancement
elif current_step == 5:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.header("🚀 Étape 6: Lancement de l'Entraînement")
    
    st.markdown("**Récapitulatif de la Configuration**")
    
    col_summary1, col_summary2 = st.columns(2)
    
    with col_summary1:
        st.subheader("📊 Données")
        st.json(st.session_state.get("split_config", {}))
        
        st.subheader("⚖️ Déséquilibre")
        st.json(st.session_state.imbalance_config)
        
        st.subheader("🎨 Prétraitement")
        st.json(st.session_state.preprocessing_config)
    
    with col_summary2:
        st.subheader("🤖 Modèle")
        st.json(st.session_state.model_config)
        
        st.subheader("⚙️ Entraînement")
        # ✅ CORRECTION: Utilisation de training_config_to_dict
        st.json(training_config_to_dict(st.session_state.training_config))
    
    st.markdown("---")
    st.subheader("🔍 Validation de la Configuration")
    
    errors, warnings = validate_training_configuration()
    
    if errors:
        for error in errors:
            st.markdown(f'<div class="config-error">{error}</div>', unsafe_allow_html=True)
        launch_disabled = True
    else:
        if warnings:
            for warning in warnings:
                st.warning(warning)
        st.success("✅ Configuration valide")
        launch_disabled = False
    
    st.markdown("---")
    st.subheader("🎯 Informations de Lancement")
    
    col_launch1, col_launch2, col_launch3 = st.columns(3)
    
    with col_launch1:
        total_train_images = len(st.session_state.get('X_train', []))
        if st.session_state.preprocessing_config.get("augmentation_enabled", False):
            total_train_images *= st.session_state.preprocessing_config.get("augmentation_factor", 1)
        st.metric("📷 Images Train", f"{total_train_images:,}")
    
    with col_launch2:
        epochs = get_training_config_value(st.session_state.training_config, 'epochs', 50)
        batch_size = get_training_config_value(st.session_state.training_config, 'batch_size', 32)
        training_time_estimate = (total_train_images * epochs) / (batch_size * 100)  # Estimation rough
        st.metric("⏱️ Temps estimé", f"{max(1, int(training_time_estimate))} min")
    
    with col_launch3:
        use_weights = st.session_state.imbalance_config.get("use_class_weights", False)
        st.metric("⚖️ Gestion déséquilibre", "Activée" if use_weights else "Désactivée")
    
    # Informations système
    st.markdown("---")
    st.subheader("💻 Informations Système")
    
    col_sys1, col_sys2, col_sys3 = st.columns(3)
    
    with col_sys1:
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        st.info(f"**Device:** {device}")
        if torch.cuda.is_available():
            st.success(f"GPU: {torch.cuda.get_device_name(0)}")
    
    with col_sys2:
        mixed_precision = get_training_config_value(st.session_state.training_config, 'use_mixed_precision', True)
        st.info(f"**Mixed Precision:** {'Activée' if mixed_precision else 'Désactivée'}")
    
    with col_sys3:
        deterministic = get_training_config_value(st.session_state.training_config, 'deterministic', True)
        st.info(f"**Mode Déterministe:** {'Activé' if deterministic else 'Désactivé'}")
    
    # Bouton de lancement
    st.markdown("---")
    
    if st.button("🚀 Démarrer l'Entraînement", type="primary", use_container_width=True, disabled=launch_disabled):
        training_container = st.container()
        with training_container:
            st.markdown("### 📈 Entraînement en Cours...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.empty()
            results_placeholder = st.empty()
            
            streamlit_components = {
                "progress_bar": progress_bar,
                "status_text": status_text,
                "metrics_placeholder": metrics_placeholder
            }
            
            try:
                # Détermination du type d'anomalie
                model_type = st.session_state.model_config["model_type"]
                anomaly_type = None
                if model_type == ModelType.CONV_AUTOENCODER.value:
                    anomaly_type = "structural"
                
                # Lancement de l'entraînement
                model, history = train_computer_vision_model_production(
                    X_train=st.session_state.X_train,
                    y_train=st.session_state.y_train,
                    X_val=st.session_state.X_val,
                    y_val=st.session_state.y_val,
                    model_type=model_type,
                    model_params=st.session_state.model_config["model_params"],
                    training_config=st.session_state.training_config,
                    streamlit_components=streamlit_components,
                    imbalance_config=st.session_state.imbalance_config,
                    anomaly_type=anomaly_type
                )
                
                if model is not None and history.get("success", False):
                    # Sauvegarde des résultats
                    st.session_state.trained_model = model
                    st.session_state.training_history = history
                    st.session_state.training_results = {
                        "model": model,
                        "history": history,
                        "training_config": st.session_state.training_config,
                        "model_config": st.session_state.model_config,
                        "preprocessing_config": st.session_state.preprocessing_config,
                        "imbalance_config": st.session_state.imbalance_config,
                        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    with results_placeholder.container():
                        st.success("✅ Entraînement terminé avec succès!")
                        
                        col_result1, col_result2, col_result3, col_result4 = st.columns(4)
                        
                        with col_result1:
                            st.metric(
                                "Meilleure Loss Val", 
                                f"{history.get('best_val_loss', 0):.4f}"
                            )
                        
                        with col_result2:
                            st.metric(
                                "Époques Effectuées", 
                                history.get('total_epochs_trained', 0)
                            )
                        
                        with col_result3:
                            st.metric(
                                "Temps Total", 
                                f"{history.get('training_time', 0):.1f}s"
                            )
                        
                        with col_result4:
                            early_stopped = "Oui" if history.get('early_stopping_triggered', False) else "Non"
                            st.metric("Early Stopping", early_stopped)
                        
                        # Graphiques des courbes
                        st.markdown("### 📊 Courbes d'Entraînement")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=history['train_loss'],
                            mode='lines',
                            name='Train Loss',
                            line=dict(color='#667eea', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            y=history['val_loss'],
                            mode='lines',
                            name='Val Loss',
                            line=dict(color='#764ba2', width=2)
                        ))
                        fig.update_layout(
                            title="Loss au fil des Époques",
                            xaxis_title="Époque",
                            yaxis_title="Loss",
                            template="plotly_white",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Métriques supplémentaires si disponibles
                        if history.get('val_accuracy'):
                            col_met1, col_met2 = st.columns(2)
                            
                            with col_met1:
                                fig_acc = go.Figure()
                                fig_acc.add_trace(go.Scatter(
                                    y=history['val_accuracy'],
                                    mode='lines+markers',
                                    name='Accuracy',
                                    line=dict(color='#2ecc71', width=2)
                                ))
                                fig_acc.update_layout(
                                    title="Accuracy Validation",
                                    xaxis_title="Époque",
                                    yaxis_title="Accuracy",
                                    template="plotly_white"
                                )
                                st.plotly_chart(fig_acc, use_container_width=True)
                            
                            with col_met2:
                                if history.get('val_f1'):
                                    fig_f1 = go.Figure()
                                    fig_f1.add_trace(go.Scatter(
                                        y=history['val_f1'],
                                        mode='lines+markers',
                                        name='F1 Score',
                                        line=dict(color='#e74c3c', width=2)
                                    ))
                                    fig_f1.update_layout(
                                        title="F1 Score Validation",
                                        xaxis_title="Époque",
                                        yaxis_title="F1 Score",
                                        template="plotly_white"
                                    )
                                    st.plotly_chart(fig_f1, use_container_width=True)
                    
                    st.markdown("---")
                    
                    col_action1, col_action2 = st.columns(2)
                    
                    with col_action1:
                        if st.button("📊 Aller à l'Évaluation", type="primary", use_container_width=True):
                            st.switch_page("pages/5_anomaly_evaluation.py")
                    
                    with col_action2:
                        if st.button("🔄 Nouvel Entraînement", use_container_width=True):
                            st.session_state.training_wizard_step = 0
                            st.rerun()
                
                else:
                    st.error("❌ L'entraînement a échoué")
                    if "error" in history:
                        st.error(f"Erreur: {history['error']}")
                    
                    # Logs de débogage
                    with st.expander("🔍 Détails de l'erreur"):
                        st.json(history)
                    
                    if st.button("🔙 Retour à la configuration", use_container_width=True):
                        st.session_state.training_wizard_step = 4
                        st.rerun()
            
            except Exception as e:
                st.error(f"❌ Erreur lors de l'entraînement: {str(e)}")
                logger.error(f"Training launch error: {e}", exc_info=True)
                
                # Affichage de l'erreur complète pour débogage
                with st.expander("🔍 Stack trace complète"):
                    st.code(str(e))
                    import traceback
                    st.code(traceback.format_exc())
    
    # Bouton retour
    st.markdown("---")
    col_back, _ = st.columns(2)
    with col_back:
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.training_wizard_step = 4
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# FOOTER ET INFORMATIONS
# ============================================================================

st.markdown("---")

# Informations sur l'état actuel
with st.expander("ℹ️ Informations sur la session"):
    st.markdown("### État de la Configuration")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("**Données:**")
        st.write(f"- Images totales: {len(X):,}")
        st.write(f"- Classes: {len(np.unique(y))}")
        
        if 'X_train' in st.session_state:
            st.write(f"- Train: {len(st.session_state.X_train):,}")
            st.write(f"- Validation: {len(st.session_state.X_val):,}")
            st.write(f"- Test: {len(st.session_state.X_test):,}")
    
    with col_info2:
        st.markdown("**Configuration:**")
        st.write(f"- Étape actuelle: {current_step + 1}/{len(WIZARD_STEPS)}")
        
        if 'model_config' in st.session_state:
            st.write(f"- Modèle: {st.session_state.model_config.get('model_type', 'Non défini')}")
        
        if 'training_config' in st.session_state:
            epochs = get_training_config_value(st.session_state.training_config, 'epochs', 'N/A')
            batch_size = get_training_config_value(st.session_state.training_config, 'batch_size', 'N/A')
            st.write(f"- Époques: {epochs}")
            st.write(f"- Batch size: {batch_size}")

# Aide et documentation
with st.expander("❓ Aide et Recommandations"):
    st.markdown("""
    ### 📖 Guide Rapide
    
    **1. Split des données**
    - Recommandé: 60% train, 20% val, 20% test
    - La stratification préserve la distribution des classes
    
    **2. Gestion du déséquilibre**
    - Ratio > 3:1 → Activer les poids de classe
    - Ratio > 5:1 → Ajouter l'augmentation ciblée
    
    **3. Prétraitement**
    - `standardize`: Normalisation Z-score (recommandé)
    - `normalize`: Min-max [0,1]
    - Augmentation: x2-x4 pour petits datasets
    
    **4. Choix du modèle**
    - **AutoEncoder**: Dataset non étiqueté ou peu étiqueté
    - **CNN Classifier**: Dataset bien étiqueté
    - **Transfer Learning**: Petit dataset (<1000 images)
    
    **5. Hyperparamètres**
    - Learning rate: 1e-4 (début recommandé)
    - Batch size: 32 (compromis vitesse/mémoire)
    - Early stopping: 10-15 époques patience
    
    **6. Optimisation**
    - AdamW: Meilleur pour la plupart des cas
    - SGD + Momentum: Si vous avez beaucoup de données
    - ReduceLROnPlateau: Réduit automatiquement le LR
    
    ### ⚡ Astuces Performance
    
    - **GPU disponible**: Activez Mixed Precision
    - **Petit dataset**: Utilisez Transfer Learning
    - **Déséquilibre**: Combinez poids + augmentation
    - **Overfitting**: Augmentez dropout et weight decay
    - **Underfitting**: Augmentez la complexité du modèle
    
    ### 🐛 Résolution de Problèmes
    
    **"Out of Memory"**
    - Réduisez le batch size
    - Réduisez la taille des images
    - Réduisez num_workers
    
    **"Loss ne diminue pas"**
    - Augmentez le learning rate
    - Vérifiez la normalisation des données
    - Vérifiez les poids de classe
    
    **"Overfitting"**
    - Augmentez dropout
    - Ajoutez de l'augmentation de données
    - Réduisez la complexité du modèle
    - Activez weight decay
    
    **"Entraînement très lent"**
    - Réduisez la taille des images
    - Augmentez batch size (si mémoire suffisante)
    - Activez Mixed Precision
    - Réduisez num_workers si CPU limité
    """)

# Boutons de réinitialisation et navigation
st.markdown("---")

col_nav1, col_nav2, col_nav3 = st.columns(3)

with col_nav1:
    if st.button("🏠 Retour au Dashboard", use_container_width=True):
        st.switch_page("pages/1_dashboard.py")

with col_nav2:
    if st.button("🔄 Réinitialiser le Wizard", use_container_width=True):
        # Réinitialisation de l'état du wizard
        st.session_state.training_wizard_step = 0
        
        # Réinitialisation des configurations (optionnel)
        if st.session_state.get('reset_configs', False):
            for key in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test',
                       'split_config', 'model_config', 'preprocessing_config']:
                if key in st.session_state:
                    del st.session_state[key]
        
        st.rerun()

with col_nav3:
    if 'trained_model' in st.session_state:
        if st.button("📊 Voir les Résultats", type="primary", use_container_width=True):
            st.switch_page("pages/5_anomaly_evaluation.py")

# Affichage des avertissements système si nécessaire
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if gpu_memory < 4:
        st.warning(f"⚠️ GPU avec mémoire limitée ({gpu_memory:.1f} GB). Considérez batch_size ≤ 16")
else:
    st.info("ℹ️ Entraînement sur CPU détecté. Pour de meilleures performances, utilisez un GPU.")

# Debug info (à retirer en production)
if st.session_state.get('debug_mode', False):
    with st.expander("🔧 Debug Info"):
        st.write("**Session State Keys:**")
        st.write(list(st.session_state.keys()))
        
        st.write("**Training Config Type:**")
        st.write(type(st.session_state.get('training_config')))
        
        if 'training_config' in st.session_state:
            st.write("**Training Config Content:**")
            st.json(training_config_to_dict(st.session_state.training_config))
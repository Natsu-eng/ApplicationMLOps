"""
🚀 UI Components - Training Computer Vision
Composants UI réutilisables pour le workflow d'entraînement CV
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter

from monitoring.state_managers import STATE
from src.shared.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# STYLES CSS GLOBAUX
# ============================================================================

def inject_training_vision_css():
    """Injection CSS pour pages training vision"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .workflow-step-card {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border: 1px solid #e0e0e0;
            margin-bottom: 1.5rem;
        }
        .model-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border: 2px solid transparent;
            transition: all 0.3s ease;
            cursor: pointer;
            height: 100%;
        }
        .model-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        .model-card.selected {
            border-color: #667eea;
            background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
        }
        .status-badge {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            margin: 0.2rem;
        }
        .badge-success { background: #28a745; color: white; }
        .badge-warning { background: #ffc107; color: black; }
        .badge-danger { background: #dc3545; color: white; }
        .badge-info { background: #17a2b8; color: white; }
        .param-section {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid #28a745;
        }
        .mode-badge-supervised {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 600;
            display: inline-block;
        }
        .mode-badge-unsupervised {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 600;
            display: inline-block;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# DÉTECTION MODE & VALIDATION
# ============================================================================

def detect_training_mode(y: Optional[np.ndarray]) -> Tuple[str, Dict]:
    """
    Détection ROBUSTE du mode pour MVTec AD et datasets classiques.

    - Validation explicite de y_train via STATE
    - Fallback sécurisé si y_train absent
    - Logging détaillé pour debugging
    - Gestion des cas limites (y vide, None, etc.)
    
    Priorité absolue : si y_train ne contient QUE des 0 → UNSUPERVISED
    ✅ Support y=None pour non supervisé
    
    Args:
        y: Labels complets (potentiellement train+val+test) ou None pour non supervisé
        
    Returns:
        Tuple (mode, metadata)
        - mode: "supervised" | "unsupervised"
        - metadata: Dict avec détails de la tâche
        
    Raises:
        ValueError: Si les données sont invalides
    """
    from monitoring.state_managers import STATE
    from utils.task_detector import detect_cv_task, TaskType
    
    logger.debug("🔍 Début détection mode training")
    
    # === VALIDATION ENTRÉE ===
    # ✅ CORRECTION: Gérer y=None explicitement pour non supervisé
    if y is None:
        logger.info("✅ y=None détecté → Mode UNSUPERVISED (non supervisé)")
        return "unsupervised", {
            "n_classes": 0,
            "task": "unsupervised",
            "labels": [],
            "description": "Unsupervised Learning - No labels provided",
            "detection_source": "y_parameter_none",
            "y_train_available": False
        }
    
    if len(y) == 0:
        raise ValueError("Labels y vides fournis à detect_training_mode")
    
    # === PRIORITÉ 1 : y_train depuis STATE ===
    y_to_check = None
    source = "unknown"
    
    if hasattr(STATE.data, 'y_train') and STATE.data.y_train is not None:
        y_train = STATE.data.y_train
        
        # Validation supplémentaire
        if isinstance(y_train, np.ndarray) and len(y_train) > 0:
            y_to_check = y_train
            source = "STATE.data.y_train"
            # CORRECTION : Format string au lieu d'arguments nommés
            logger.info(
                f"✅ Détection via y_train depuis STATE - "
                f"n_samples: {len(y_train)}, "
                f"unique_labels: {len(np.unique(y_train))}"
            )
        else:
            # CORRECTION : Format string
            logger.warning(
                f"⚠️ y_train dans STATE mais invalide - "
                f"type: {type(y_train)}, "
                f"length: {len(y_train) if hasattr(y_train, '__len__') else 'N/A'}"
            )
    
    # === FALLBACK : Utiliser y fourni en paramètre ===
    if y_to_check is None:
        y_to_check = y
        source = "y_parameter"
        # CORRECTION : Format string
        logger.warning(
            f"⚠️ Fallback sur y paramètre (y_train absent/invalide) - "
            f"n_samples: {len(y)}, "
            f"unique_labels: {len(np.unique(y))}"
        )
    
    # === DÉTECTION VIA TASK DETECTOR ===
    try:
        task_type, metadata = detect_cv_task(y_to_check)
        
        # CORRECTION : Format string
        logger.info(
            f"🎯 Tâche détectée - "
            f"task_type: {task_type.value}, "
            f"n_classes: {metadata.get('n_classes')}, "
            f"source: {source}, "
            f"is_binary: {metadata.get('is_binary', False)}"
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur task detector: {e}")
        raise ValueError(f"Impossible de détecter la tâche: {str(e)}") from e
    
    # === MAPPING TASK → MODE ===
    if task_type == TaskType.UNSUPERVISED:
        mode = "unsupervised"
        logger.info("✅ Mode UNSUPERVISED confirmé (MVTec AD)")
    
    elif task_type in [
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
        TaskType.ANOMALY_DETECTION
    ]:
        mode = "supervised"
        logger.info(f"✅ Mode SUPERVISED confirmé ({task_type.value})")
    
    else:
        # Cas non prévu (sécurité)
        logger.error(f"❌ TaskType non géré: {task_type}")
        raise ValueError(f"TaskType non supporté: {task_type}")
    
    # === ENRICHISSEMENT METADATA ===
    metadata['detection_source'] = source
    metadata['y_train_available'] = hasattr(STATE.data, 'y_train') and STATE.data.y_train is not None
    
    if source == "y_parameter":
        metadata['warning'] = (
            "Détection basée sur y complet (y_train absent). "
            "Peut être imprécis pour MVTec AD."
        )
    
    # CORRECTION : Format string
    logger.info(
        f"✅ Détection mode terminée - "
        f"mode: {mode}, "
        f"task: {metadata.get('task')}, "
        f"n_classes: {metadata.get('n_classes')}, "
        f"source: {source}"
    )
    
    return mode, metadata

# === FONCTION HELPER : VALIDATION EXPLICITE y_train ===
def validate_y_train_for_mvtec(y_train: Optional[np.ndarray]) -> Tuple[bool, str]:
    if y_train is None:
        return False, "y_train=None (non fourni)"   
    if not isinstance(y_train, np.ndarray):
        return False, f"y_train type invalide: {type(y_train)}"    
    if len(y_train) == 0:
        return False, "y_train vide"   
    unique_labels = np.unique(y_train)  
    # MVTec AD: doit contenir UNIQUEMENT des 0
    if len(unique_labels) == 1 and unique_labels[0] == 0:
        return True, f"✅ MVTec AD valide ({len(y_train)} images normales)"   
    # Pas MVTec AD
    return True, f"⚠️ Pas MVTec AD ({len(unique_labels)} classes: {unique_labels.tolist()})"


def perform_stratified_split(
    X: np.ndarray,
    y: Optional[np.ndarray],
    test_size: float = 0.2,
    val_size: float = 0.2,
    mode: str = "supervised"
) -> Dict[str, Any]:
    """
    Split stratifié ou contrôlé selon le mode.

    Pour le mode non supervisé (MVTec AD) :
    - Train : uniquement les images normales (y=0)
    - Validation : uniquement les images normales (y=0)
    - Test : toutes les anomalies (y=1) et le reste des normales (y=0)
    
    ✅ Support y=None pour non supervisé complet

    Args:
        X: Données
        y: Labels (peut être None pour non supervisé)
        test_size: Ratio test (par rapport à l'ensemble total)
        val_size: Ratio validation (par rapport à l'ensemble total)
        mode: "supervised" ou "unsupervised"

    Returns:
        Dict avec X_train, X_val, X_test, y_train, y_val, y_test, split_info
    """
    from sklearn.model_selection import train_test_split

    logger.info(f"🔄 Split {mode} - test:{test_size:.0%} val:{val_size:.0%}")

    # ✅ CORRECTION: Gérer y=None pour non supervisé
    if y is None:
        if mode != "unsupervised":
            logger.warning("⚠️ y=None mais mode='supervised', passage en 'unsupervised'")
            mode = "unsupervised"
        
        # Split simple sans stratification
        X_train_val, X_test = train_test_split(
            X, test_size=test_size, random_state=42, shuffle=True
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val = train_test_split(
            X_train_val, test_size=val_ratio, random_state=42, shuffle=True
        )
        
        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": None,
            "y_val": None,
            "y_test": None,
            "split_info": {
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "mode": "unsupervised",
                "test_size": test_size,
                "val_size": val_size,
                "y_provided": False
            }
        }

    if mode == "supervised":
        # Split stratifié standard
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # Split validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_ratio,
            stratify=y_train_val,
            random_state=42
        )
    else:
        # Mode non supervisé (MVTec AD)
        # Séparer les indices normaux et anomalies
        normal_idx = np.where(y == 0)[0]
        anomaly_idx = np.where(y == 1)[0]

        # Calcul des tailles pour le test et la validation
        n_test_normal = int(len(normal_idx) * test_size)
        n_val_normal = int(len(normal_idx) * val_size)

        # Mélanger les indices normaux
        np.random.shuffle(normal_idx)

        # Split des indices normaux : train, val, test
        train_normal_idx = normal_idx[:len(normal_idx) - n_test_normal - n_val_normal]
        val_normal_idx = normal_idx[len(normal_idx) - n_test_normal - n_val_normal: len(normal_idx) - n_test_normal]
        test_normal_idx = normal_idx[len(normal_idx) - n_test_normal:]

        # Test set : anomalies + normales (pour évaluation)
        test_idx = np.concatenate([test_normal_idx, anomaly_idx])

        # Construction des datasets
        X_train = X[train_normal_idx]
        y_train = y[train_normal_idx]

        X_val = X[val_normal_idx]
        y_val = y[val_normal_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        logger.info(f"✅ Split MVTec AD - Train: {len(X_train)} normales, Val: {len(X_val)} normales, Test: {len(X_test)} (dont {len(anomaly_idx)} anomalies)")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "split_info": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "mode": mode,
            "test_size": test_size,
            "val_size": val_size
        }
    }


def validate_split_quality(split_data: Dict, mode: str, metadata: Dict) -> Tuple[bool, List[str]]:
    """
    Valide la qualité du split selon le mode.
    
    Returns:
        (is_valid, warnings)
    """
    warnings = []
    
    # Tailles minimales
    if len(split_data["X_train"]) < 100:
        warnings.append("⚠️ Training set < 100 échantillons (risque overfitting)")
    
    if len(split_data["X_val"]) < 20:
        warnings.append("⚠️ Validation set < 20 échantillons (validation peu fiable)")
    
    if len(split_data["X_test"]) < 10:
        warnings.append("⚠️ Test set < 10 échantillons")
    
    if mode == "unsupervised":
        # Anomalies dans test
        n_anomalies_test = np.sum(split_data["y_test"] == 1)
        if n_anomalies_test == 0:
            warnings.append("❌ CRITIQUE: Aucune anomalie dans test set")
            return False, warnings
        
        # Train majoritairement normal
        n_normal_train = np.sum(split_data["y_train"] == 0)
        normal_ratio = n_normal_train / len(split_data["y_train"])
        
        if normal_ratio < 0.8:
            warnings.append(f"⚠️ Train contient {(1-normal_ratio)*100:.0f}% anomalies (>20%)")
        
        # Validation avec anomalies
        n_anomalies_val = np.sum(split_data["y_val"] == 1)
        if n_anomalies_val == 0:
            warnings.append("ℹ️ Validation sans anomalies (acceptable pour autoencoders)")
    
    else:  # supervised
        original_classes = set(np.unique(STATE.data.y))  # toutes les classes du dataset
        expected_classes = original_classes
        
        for name in ["y_train", "y_val", "y_test"]:
            y = split_data[name]
            present_classes = set(np.unique(y))
            
            # Vérifier que toutes les classes attendues sont présentes
            missing = expected_classes - present_classes
            if missing:
                warnings.append(f"❌ {name}: Classes manquantes {missing}")
                return False, warnings
        
        # Équilibre raisonnable
        train_counts = Counter(split_data["y_train"])
        max_count = max(train_counts.values())
        min_count = min(train_counts.values())
        
        # CORRECTION : Vérifier division par zéro
        if min_count == 0:
            warnings.append("❌ Certaines classes ont 0 échantillon en training")
            return False, warnings
            
        if max_count / min_count > 10:
            warnings.append(f"⚠️ Déséquilibre élevé (ratio {max_count/min_count:.1f}:1)")
    
    is_valid = len([w for w in warnings if "❌" in w]) == 0
    
    return is_valid, warnings


# ============================================================================
# COMPOSANTS UI VISUALISATION
# ============================================================================

def render_mode_badge(mode: str, metadata: Dict):
    """Affiche un badge du mode détecté"""
    if mode == "supervised":
        badge_class = "mode-badge-supervised"
        icon = "🎯"
        label = "SUPERVISÉ"
        desc = f"Classification {metadata['n_classes']} classes"
    else:
        badge_class = "mode-badge-unsupervised"
        icon = "🔍"
        label = "NON-SUPERVISÉ"
        desc = "Détection d'anomalies"
    
    st.markdown(
        f"""
        <div class="{badge_class}">
            {icon} Mode {label}
        </div>
        <p style="color: #666; margin-top: 0.5rem;">{desc}</p>
        """,
        unsafe_allow_html=True
    )


def render_split_distribution_chart(split_data: Dict, mode: str):
    """Graphique de distribution du split"""
    
    # Pie chart global
    fig = go.Figure(data=[
        go.Pie(
            labels=['Train', 'Validation', 'Test'],
            values=[
                split_data['split_info']['train_samples'],
                split_data['split_info']['val_samples'],
                split_data['split_info']['test_samples']
            ],
            hole=0.4,
            marker_colors=['#28a745', '#17a2b8', '#6c757d'],
            textinfo='percent+label+value',
            hovertemplate='<b>%{label}</b><br>Échantillons: %{value}<br>Pourcentage: %{percent}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Distribution Train/Val/Test",
        showlegend=True,
        height=350,
        annotations=[dict(
            text=f'Total: {len(split_data["X_train"]) + len(split_data["X_val"]) + len(split_data["X_test"]):,}',
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False
        )]
    )
    
    st.plotly_chart(
        fig, 
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
        },
        width="stretch"
    )
    
    # Distribution des classes par split
    if mode == "unsupervised":
        # Normal vs Anomalie
        splits = {
            'Train': split_data['y_train'],
            'Validation': split_data['y_val'],
            'Test': split_data['y_test']
        }
        
        data_normal = []
        data_anomaly = []
        
        for name, y in splits.items():
            n_normal = np.sum(y == 0)
            n_anomaly = np.sum(y == 1)
            data_normal.append(n_normal)
            data_anomaly.append(n_anomaly)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=list(splits.keys()),
            y=data_normal,
            name='Normal',
            marker_color='#4facfe',
            text=data_normal,
            textposition='auto'
        ))
        fig2.add_trace(go.Bar(
            x=list(splits.keys()),
            y=data_anomaly,
            name='Anomalie',
            marker_color='#f5576c',
            text=data_anomaly,
            textposition='auto'
        ))
        
        fig2.update_layout(
            title="Distribution Normal/Anomalie par Split",
            barmode='stack',
            xaxis_title="Split",
            yaxis_title="Nombre d'images",
            height=300
        )
        
        st.plotly_chart(fig2, width="stretch")


def render_split_stats_table(split_data: Dict, mode: str, metadata: Dict):
    """Tableau statistiques du split"""
    
    stats_data = []
    
    # Mapping clés
    split_mapping = {
        'Train': ('y_train', 'X_train'),
        'Validation': ('y_val', 'X_val'), 
        'Test': ('y_test', 'X_test')
    }
    
    for split_name, (y_key, x_key) in split_mapping.items():
        y = split_data[y_key]
        x = split_data[x_key]
        
        if mode == "unsupervised":
            n_normal = int(np.sum(y == 0))
            n_anomaly = int(np.sum(y == 1))
            
            stats_data.append({
                "Split": split_name,
                "Total": len(x), 
                "Normal": n_normal,
                "Anomalies": n_anomaly,
                "% Anomalies": f"{(n_anomaly/len(y)*100):.1f}%"
            })
        else:
            class_counts = Counter(y)
            
            stats_data.append({
                "Split": split_name,
                "Total": len(x), 
                "Classes": len(class_counts),
                "Min/Max": f"{min(class_counts.values())}/{max(class_counts.values())}",
                "Ratio": f"{max(class_counts.values())/min(class_counts.values()):.1f}:1" if min(class_counts.values()) > 0 else "∞"
            })
    
    df_stats = pd.DataFrame(stats_data)
    st.dataframe(df_stats, width="stretch", hide_index=True)


def render_validation_warnings(warnings: List[str]):
    """Affiche les warnings de validation"""
    if not warnings:
        st.success("✅ Split validé - Aucun problème détecté")
        return
    
    critical = [w for w in warnings if "❌" in w]
    cautions = [w for w in warnings if "⚠️" in w]
    infos = [w for w in warnings if "ℹ️" in w]
    
    if critical:
        st.error("❌ Problèmes Critiques")
        for w in critical:
            st.markdown(f"- {w}")
    
    if cautions:
        st.warning("⚠️ Avertissements")
        for w in cautions:
            st.markdown(f"- {w}")
    
    if infos:
        st.info("ℹ️ Informations")
        for w in infos:
            st.markdown(f"- {w}")


# ============================================================================
# FILTRAGE MODÈLES PAR MODE
# ============================================================================

def filter_models_by_mode(all_models: Dict, mode: str, metadata: Dict) -> Dict:
    """
    Filtre les modèles disponibles selon le mode.
    
    Args:
        all_models: Dict complet des modèles
        mode: "supervised" | "unsupervised"
        metadata: Métadonnées du dataset
        
    Returns:
        Dict filtré des modèles compatibles
    """
    if mode == "supervised":
        # Retirer modèles non-supervisés
        return {
            cat: data for cat, data in all_models.items()
            if cat != "🔍 Détection d'Anomalies"
        }
    else:
        # Retirer modèles supervisés
        return {
            "🔍 Détection d'Anomalies": all_models.get("🔍 Détection d'Anomalies", {})
        }


# ============================================================================
# GESTION DÉSÉQUILIBRE PAR MODE
# ============================================================================

def analyze_imbalance_by_mode(
    y_train: np.ndarray,
    mode: str,
    metadata: Dict
) -> Dict[str, Any]:
    """
    Analyse le déséquilibre selon le mode.
    
    Returns:
        Dict avec métrique d'imbalance et recommandations
    """
    from collections import Counter
    
    counts = Counter(y_train)
    
    if mode == "supervised":
        # Ratio max/min standard
        max_count = max(counts.values())
        min_count = min(counts.values())
        ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Détermination niveau
        if ratio > 10:
            level = "critique"
            color = "#dc3545"
            icon = "🚨"
            recommendation = "Class weights + SMOTE impératifs"
            use_weights = True
            use_smote = True
        elif ratio > 5:
            level = "élevé"
            color = "#fd7e14"
            icon = "⚠️"
            recommendation = "Class weights fortement recommandés"
            use_weights = True
            use_smote = False
        elif ratio > 2:
            level = "modéré"
            color = "#ffc107"
            icon = "ℹ️"
            recommendation = "Class weights recommandés"
            use_weights = True
            use_smote = False
        else:
            level = "faible"
            color = "#28a745"
            icon = "✅"
            recommendation = "Aucune correction nécessaire"
            use_weights = False
            use_smote = False
    
    else:  # unsupervised
        # Pour anomalies: ratio normal/anomalie
        n_normal = counts.get(0, 0)
        n_anomaly = counts.get(1, 0)
        ratio = n_normal / n_anomaly if n_anomaly > 0 else float('inf')
        
        # IMPORTANT: Ne PAS utiliser class weights pour anomalies
        level = "normal (anomalies)"
        color = "#17a2b8"
        icon = "🔍"
        
        if n_anomaly > n_normal * 0.2:
            recommendation = "⚠️ Trop d'anomalies en train (>20%) - Nettoyer le dataset"
            use_weights = False
            use_smote = False
        else:
            recommendation = "✅ Ratio normal/anomalie acceptable pour autoencoders"
            use_weights = False
            use_smote = False
    
    return {
        "ratio": ratio,
        "level": level,
        "color": color,
        "icon": icon,
        "recommendation": recommendation,
        "use_class_weights": use_weights,
        "use_smote": use_smote,
        "counts": dict(counts),
        "mode": mode
    }


def render_imbalance_analysis(imbalance_info: Dict, y_train: np.ndarray):
    """Affiche l'analyse du déséquilibre"""
    
    # Badge niveau
    st.markdown(
        f"""
        <div style='background: {imbalance_info["color"]}; color: white; 
                    padding: 1.5rem; border-radius: 10px; text-align: center;'>
            <h3 style='margin: 0; font-size: 2rem;'>{imbalance_info["icon"]}</h3>
            <h4 style='margin: 0.5rem 0;'>Niveau: {imbalance_info["level"].title()}</h4>
            <h2 style='margin: 0;'>Ratio: {imbalance_info["ratio"]:.1f}:1</h2>
            <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>{imbalance_info["recommendation"]}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Graphique distribution
    st.markdown("### 📊 Distribution des Classes")
    
    if imbalance_info["mode"] == "supervised":
        labels = [f"Classe {k}" for k in sorted(imbalance_info["counts"].keys())]
    else:
        labels = ['Normal', 'Anomalie']
    
    values = [imbalance_info["counts"][k] for k in sorted(imbalance_info["counts"].keys())]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            text=[f"{v}<br>({v/sum(values)*100:.1f}%)" for v in values],
            textposition='auto',
            marker_color=['#4facfe' if i == 0 else '#f5576c' for i in range(len(values))]
        )
    ])
    
    fig.update_layout(
        title="Distribution des Classes en Training",
        xaxis_title="Classe",
        yaxis_title="Nombre d'images",
        height=400
    )
    
    st.plotly_chart(
        fig, 
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
        },
        width="stretch"
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'inject_training_vision_css',
    'detect_training_mode',
    'perform_stratified_split',
    'validate_split_quality',
    'render_mode_badge',
    'render_split_distribution_chart',
    'render_split_stats_table',
    'render_validation_warnings',
    'filter_models_by_mode',
    'analyze_imbalance_by_mode',
    'render_imbalance_analysis'
]
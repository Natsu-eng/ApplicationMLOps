"""
Page d'évaluation des modèles pour l'application Datalab Pro.
"""
import os
import pickle
import traceback
import numpy as np
import streamlit as st
import pandas as pd
import time
import json
import plotly.express as px
import plotly.graph_objects as go
import gc
import concurrent.futures
from typing import Dict, Optional, List, Any
from src.evaluation.model_plots import ModelEvaluationVisualizer, _generate_color_palette, _safe_get_model_task_type
from src.evaluation.metrics import get_system_metrics
from utils.report_generator import generate_pdf_report
from src.config.constants import TRAINING_CONSTANTS, LOGGING_CONSTANTS, VALIDATION_CONSTANTS, VISUALIZATION_CONSTANTS
from logging import getLogger
from datetime import datetime
import logging
from monitoring.decorators import monitor_operation

from monitoring.state_managers import init, AppPage
STATE = init()

logger = getLogger(__name__)

# Import MLflow avec gestion robuste
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

# Configuration PyArrow
os.environ["PANDAS_USE_PYARROW"] = "0"
try:
    pd.options.mode.dtype_backend = "numpy_nullable"
except Exception:
    pass

# Configuration page
st.set_page_config(
    page_title="Évaluation des Modèles",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
        border-bottom: 2px solid #3498db;
        padding-bottom: 1rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e1e8ed;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
    .best-model-card {
        border-left: 4px solid #27ae60;
        background: linear-gradient(135deg, #f8fff9 0%, #e8f5e8 100%);
    }
    .metric-title {
        font-size: 0.8rem;
        color: #7f8c8d;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .metric-subtitle {
        font-size: 0.7rem;
        color: #95a5a6;
    }
    .performance-high { color: #27ae60; }
    .performance-medium { color: #f39c12; }
    .performance-low { color: #e74c3c; }
    .tab-content {
        padding: 1rem;
        background: #ffffff;
        border-radius: 8px;
        border: 1px solid #ecf0f1;
    }
    .plot-container {
        border: 1px solid #e1e8ed;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background: #fafbfc;
    }
</style>
""", unsafe_allow_html=True)

def log_structured(level: str, message: str, extra: Dict = None):
    """Logging structuré avec gestion d'erreurs"""
    try:
        log_dict = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "module": "evaluation_page"
        }
        if extra:
            log_dict.update(extra)
        logger.log(getattr(logging, level.upper()), json.dumps(log_dict, default=str))
    except Exception as e:
        print(f"Logging error: {str(e)[:200]}")

@st.cache_data(ttl=3600, max_entries=20, show_spinner=False)
def cached_plot(fig, plot_key: str):
    """Cache les graphiques avec gestion robuste"""
    try:
        if fig is None:
            log_structured("WARNING", "Figure None dans cached_plot", {"plot_key": plot_key})
            return None
        if hasattr(fig, 'to_json'):
            return fig
        return fig
    except Exception as e:
        log_structured("ERROR", "Erreur cache graphique", {"plot_key": plot_key, "error": str(e)[:200]})
        return fig

def get_mlflow_artifact(run_id: str, artifact_path: str, client: Optional[Any] = None) -> Optional[bytes]:
    """Récupère un artefact MLflow avec gestion robuste des erreurs"""
    try:
        if not MLFLOW_AVAILABLE or client is None:
            log_structured("ERROR", "MLflow non disponible")
            return None
        artifact_data = client.download_artifacts(run_id, artifact_path)
        log_structured("INFO", "Artefact MLflow téléchargé", {
            "run_id": run_id[:8],
            "artifact_path": artifact_path
        })
        return artifact_data
    except Exception as e:
        log_structured("ERROR", "Échec téléchargement artefact", {
            "run_id": run_id[:8] if run_id else "unknown",
            "error": str(e)[:200]
        })
        return None

def display_metrics_header(validation_result: Dict[str, Any]):
    """Affiche l'en-tête avec métriques principales - VERSION CORRIGÉE"""
    try:
        if not isinstance(validation_result, dict):
            st.error("❌ Format invalide des résultats de validation")
            return

        # Extraction sécurisée des métriques
        successful_models = validation_result.get("successful_models", [])
        failed_models = validation_result.get("failed_models", [])
        total_models = len(successful_models) + len(failed_models)
        
        best_model_name = validation_result.get("best_model", "N/A")
        if not isinstance(best_model_name, str) or best_model_name == "":
            best_model_name = "N/A"
            
        task_type = validation_result.get("task_type", "unknown")

        # Calcul des métriques
        success_rate = (len(successful_models) / total_models * 100) if total_models > 0 else 0
        status_color = "#27ae60" if success_rate > 80 else "#f39c12" if success_rate > 50 else "#e74c3c"

        # Affichage des métriques
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Taux de Réussite</div>
                <div class="metric-value" style="color: {status_color};">{success_rate:.1f}%</div>
                <div class="metric-subtitle">{len(successful_models)}/{total_models} modèles</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            best_model_class = "best-model-card" if best_model_name != "N/A" else ""
            st.markdown(f"""
            <div class="metric-card {best_model_class}">
                <div class="metric-title">Meilleur Modèle</div>
                <div class="metric-value" style="font-size: 1.2rem;">{best_model_name}</div>
                <div class="metric-subtitle">Type: {task_type.title()}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Échecs</div>
                <div class="metric-value" style="color: #e74c3c;">{len(failed_models)}</div>
                <div class="metric-subtitle">Modèles échoués</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            # ✅ CORRECTION : Métrique système simplifiée et sécurisée
            try:
                import psutil
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_available_gb = memory.available / (1024**3)
                memory_color = "#27ae60" if memory_percent < 70 else "#f39c12" if memory_percent < 85 else "#e74c3c"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Mémoire Système</div>
                    <div class="metric-value" style="color: {memory_color};">{memory_percent:.1f}%</div>
                    <div class="metric-subtitle">{memory_available_gb:.1f} GB disponible</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                # Fallback si psutil n'est pas disponible
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Système</div>
                    <div class="metric-value">✅</div>
                    <div class="metric-subtitle">Prêt</div>
                </div>
                """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"❌ Erreur dans l'affichage des métriques: {str(e)[:100]}")

def create_pdf_report_latex(model_result: Dict[str, Any], task_type: str) -> Optional[bytes]:
    """Génère un rapport PDF avec LaTeX"""
    try:
        metrics = model_result.get('metrics', {})
        model_name = model_result.get('model_name', 'Unknown')
        training_time = model_result.get('training_time', 0)

        latex_content = f"""
\\documentclass[a4paper,11pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\usepackage{{noto}}
\\geometry{{margin=1in}}
\\begin{{document}}

\\begin{{center}}
    \\textbf{{\\Large Rapport d'Évaluation du Modèle: {model_name}}} \\\\[0.5cm]
    \\textit{{Type de tâche: {task_type.title()}}}
\\end{{center}}

\\section*{{Métriques de Performance}}
\\begin{{tabular}}{{ll}}
    \\toprule
    \\textbf{{Métrique}} & \\textbf{{Valeur}} \\\\
    \\midrule
"""
        if task_type == 'classification':
            latex_content += f"    Accuracy & {metrics.get('accuracy', 0):.3f} \\\\\\ \n"
            latex_content += f"    Precision & {metrics.get('precision', 0):.3f} \\\\\\ \n"
            latex_content += f"    Recall & {metrics.get('recall', 0):.3f} \\\\\\ \n"
            latex_content += f"    F1-Score & {metrics.get('f1', 0):.3f} \\\\\\ \n"
        elif task_type == 'regression':
            latex_content += f"    R² Score & {metrics.get('r2', 0):.3f} \\\\\\ \n"
            latex_content += f"    MAE & {metrics.get('mae', 0):.3f} \\\\\\ \n"
            latex_content += f"    RMSE & {metrics.get('rmse', 0):.3f} \\\\\\ \n"
        else:  # clustering
            latex_content += f"    Silhouette Score & {metrics.get('silhouette_score', 0):.3f} \\\\\\ \n"
            latex_content += f"    Nombre de Clusters & {metrics.get('n_clusters', 'N/A')} \\\\\\ \n"

        latex_content += f"    Temps d'entraînement & {training_time:.1f}s \\\\\\ \n"
        latex_content += """
    \\bottomrule
\\end{tabular}

\\section*{{Résumé}}
Ce rapport présente les performances du modèle \\textbf{""" + model_name + """} pour la tâche de """ + task_type + """.
Veuillez consulter les visualisations pour plus de détails.

\\end{{document}}
"""
        pdf_bytes = generate_pdf_report({'content': latex_content})
        log_structured("INFO", "Rapport PDF généré", {"model_name": model_name})
        return pdf_bytes
    except Exception as e:
        log_structured("ERROR", "Génération PDF échouée", {"error": str(e)[:200]})
        return None

def model_has_predict_proba(model) -> bool:
    """Vérifie si le modèle supporte predict_proba"""
    try:
        if model is None:
            return False
        if hasattr(model, 'named_steps'):
            final_step = list(model.named_steps.values())[-1]
            return hasattr(final_step, 'predict_proba')
        return hasattr(model, 'predict_proba')
    except Exception:
        return False


@monitor_operation
def display_model_details(visualizer, model_result: Dict[str, Any], task_type: str):
    """Affiche les détails complets d'un modèle - VERSION CORRIGÉE AVEC GESTION DES DONNÉES"""
    try:
        model_name = model_result.get('model_name', 'Unknown')
        unique_id = f"{model_name}_{int(time.time())}"
        
        st.markdown(f"#### 🔍 Détails du modèle: **{model_name}**")

        # ========================================================================
        # DIAGNOSTIC COMPLET DES DONNÉES DISPONIBLES
        # ========================================================================
        with st.expander("🔧 Diagnostic des Données", expanded=False):
            diagnostic_info = {
                "model_name": model_name,
                "task_type": task_type,
                "has_model": model_result.get('model') is not None,
                "has_metrics": bool(model_result.get('metrics')),
                "has_X_train": model_result.get('X_train') is not None,
                "has_X_test": model_result.get('X_test') is not None,
                "has_y_train": model_result.get('y_train') is not None,
                "has_y_test": model_result.get('y_test') is not None,
                "has_labels": model_result.get('labels') is not None,
                "has_X_sample": model_result.get('X_sample') is not None,
                "metrics_available": list(model_result.get('metrics', {}).keys()),
                "feature_names_count": len(model_result.get('feature_names', [])),
                "success_status": model_result.get('success', False)
            }
            st.json(diagnostic_info)

        # ========================================================================
        # VALIDATION DES DONNÉES OBLIGATOIRES
        # ========================================================================
        has_model = model_result.get('model') is not None
        has_metrics = bool(model_result.get('metrics'))
        
        if not has_model:
            st.error("❌ **Modèle non disponible** - Impossible d'analyser ce modèle")
            return
            
        if not has_metrics:
            st.error("❌ **Métriques non disponibles** - Aucune performance à afficher")
            return

        # ========================================================================
        # AFFICHAGE DES MÉTRIQUES PRINCIPALES
        # ========================================================================
        st.markdown("---")
        st.markdown("#### 📊 Métriques de Performance")
        
        metrics = model_result.get('metrics', {})
        
        if task_type == 'classification':
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                accuracy = metrics.get('accuracy', 0)
                st.metric("Accuracy", f"{accuracy:.3f}")
            with col2:
                precision = metrics.get('precision', 0)
                st.metric("Precision", f"{precision:.3f}")
            with col3:
                recall = metrics.get('recall', 0)
                st.metric("Recall", f"{recall:.3f}")
            with col4:
                f1 = metrics.get('f1_score', 0)
                st.metric("F1-Score", f"{f1:.3f}")
                
            # Métriques supplémentaires
            if 'roc_auc' in metrics:
                col5, col6 = st.columns(2)
                with col5:
                    st.metric("AUC-ROC", f"{metrics['roc_auc']:.3f}")
                    
        elif task_type == 'regression':
            col1, col2, col3 = st.columns(3)
            with col1:
                r2 = metrics.get('r2', 0)
                st.metric("R² Score", f"{r2:.3f}")
            with col2:
                mae = metrics.get('mae', 0)
                st.metric("MAE", f"{mae:.3f}")
            with col3:
                rmse = metrics.get('rmse', 0)
                st.metric("RMSE", f"{rmse:.3f}")
                
        elif task_type == 'clustering':
            col1, col2, col3 = st.columns(3)
            with col1:
                silhouette = metrics.get('silhouette_score', 0)
                st.metric("Silhouette", f"{silhouette:.3f}")
            with col2:
                n_clusters = metrics.get('n_clusters', 'N/A')
                st.metric("Clusters", str(n_clusters))
            with col3:
                db_index = metrics.get('davies_bouldin_score', 'N/A')
                if isinstance(db_index, (int, float)):
                    st.metric("DB Index", f"{db_index:.3f}")
                else:
                    st.metric("DB Index", str(db_index))

        # ========================================================================
        # 🆕 CORRECTION : GESTION ROBUSTE DES DONNÉES DE VISUALISATION
        # ========================================================================
        st.markdown("---")
        st.markdown("#### 📈 Visualisations")

        # Vérification améliorée des données pour visualisations
        has_visualization_data = False
        missing_data_details = []
        
        if task_type == 'clustering':
            has_X = model_result.get('X_train') is not None
            has_labels = model_result.get('labels') is not None
            has_visualization_data = has_X and has_labels
            
            if not has_X:
                missing_data_details.append("❌ Données d'entraînement (X_train) manquantes")
            if not has_labels:
                missing_data_details.append("❌ Labels de clustering manquants")
                
        else:
            has_X_test = model_result.get('X_test') is not None
            has_y_test = model_result.get('y_test') is not None
            has_visualization_data = has_X_test and has_y_test
            
            if not has_X_test:
                missing_data_details.append("❌ Données de test (X_test) manquantes")
            if not has_y_test:
                missing_data_details.append("❌ Labels de test (y_test) manquants")

        if not has_visualization_data:
            st.warning("""
            ⚠️ **Données de visualisation limitées**
            
            Les données nécessaires aux visualisations ne sont pas disponibles pour ce modèle.
            """)
            
            if missing_data_details:
                st.error("**Détail des données manquantes:**")
                for detail in missing_data_details:
                    st.error(detail)
            
            # 🆕 TENTATIVE DE RÉCUPÉRATION ALTERNATIVE
            st.info("🔄 Tentative de récupération des données alternatives...")
            
            recovery_attempted = False
            
            if task_type != 'clustering':
                # Essai 1: Utiliser les données d'entraînement si disponibles
                if model_result.get('X_train') is not None and model_result.get('y_train') is not None:
                    st.success("✅ Utilisation des données d'entraînement comme alternative")
                    model_result['X_test'] = model_result['X_train']
                    model_result['y_test'] = model_result['y_train']
                    has_visualization_data = True
                    recovery_attempted = True
                
                # Essai 2: Utiliser l'échantillon réduit si disponible
                elif model_result.get('X_sample') is not None and model_result.get('y_sample') is not None:
                    st.success("✅ Utilisation de l'échantillon réduit comme alternative")
                    model_result['X_test'] = model_result['X_sample']
                    model_result['y_test'] = model_result['y_sample']
                    has_visualization_data = True
                    recovery_attempted = True
            
            if not recovery_attempted:
                st.info("💡 **Solutions possibles:**\n- Relancez l'entraînement\n- Vérifiez les logs d'erreur\n- Contactez l'administrateur")
                return
            else:
                st.warning("⚠️ Visualisations basées sur des données alternatives (entraînement/échantillon)")

        # ========================================================================
        # VISUALISATIONS SPÉCIFIQUES PAR TYPE DE TÂCHE
        # ========================================================================
        if task_type == 'clustering':
            # VISUALISATION CLUSTERING
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔮 Visualisation des Clusters**")
                try:
                    cluster_plot = visualizer.create_cluster_visualization(model_result)
                    if cluster_plot:
                        st.plotly_chart(
                            cached_plot(cluster_plot, f"cluster_{unique_id}"),
                            use_container_width=True,
                            key=f"cluster_{unique_id}"
                        )
                    else:
                        st.info("ℹ️ Visualisation clusters non générable")
                except Exception as e:
                    st.error(f"❌ Erreur visualisation clusters: {str(e)[:100]}")
                    logger.warning(f"Erreur cluster visualization: {e}")

            with col2:
                st.markdown("**📊 Analyse Silhouette**")
                try:
                    silhouette_plot = visualizer.create_silhouette_analysis(model_result)
                    if silhouette_plot:
                        st.plotly_chart(
                            cached_plot(silhouette_plot, f"silhouette_{unique_id}"),
                            use_container_width=True,
                            key=f"silhouette_{unique_id}"
                        )
                    else:
                        st.info("ℹ️ Analyse silhouette non générable")
                except Exception as e:
                    st.error(f"❌ Erreur analyse silhouette: {str(e)[:100]}")
                    logger.warning(f"Erreur silhouette analysis: {e}")

        elif task_type == 'classification':
            # VISUALISATIONS CLASSIFICATION
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Matrice de Confusion**")
                try:
                    cm_plot = visualizer.create_confusion_matrix(model_result)
                    if cm_plot:
                        st.plotly_chart(
                            cached_plot(cm_plot, f"cm_{unique_id}"),
                            use_container_width=True,
                            key=f"cm_{unique_id}"
                        )
                    else:
                        st.info("ℹ️ Matrice de confusion non disponible")
                except Exception as e:
                    st.error(f"❌ Erreur matrice confusion: {str(e)[:100]}")

            with col2:
                st.markdown("**📈 Courbe ROC**")
                try:
                    roc_plot = visualizer.create_roc_curve(model_result)
                    if roc_plot:
                        st.plotly_chart(
                            cached_plot(roc_plot, f"roc_{unique_id}"),
                            use_container_width=True,
                            key=f"roc_{unique_id}"
                        )
                    else:
                        st.info("ℹ️ Courbe ROC non disponible")
                except Exception as e:
                    st.error(f"❌ Erreur courbe ROC: {str(e)[:100]}")

            # Importance des features
            st.markdown("**🎯 Importance des Features**")
            try:
                feature_plot = visualizer.create_feature_importance_plot(model_result)
                if feature_plot:
                    st.plotly_chart(
                        cached_plot(feature_plot, f"feature_{unique_id}"),
                        use_container_width=True,
                        key=f"feature_{unique_id}"
                    )
                else:
                    st.info("ℹ️ Importance des features non disponible")
            except Exception as e:
                st.error(f"❌ Erreur importance features: {str(e)[:100]}")

        elif task_type == 'regression':
            # VISUALISATIONS RÉGRESSION
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📉 Graphique des Résidus**")
                try:
                    residuals_plot = visualizer.create_residuals_plot(model_result)
                    if residuals_plot:
                        st.plotly_chart(
                            cached_plot(residuals_plot, f"residuals_{unique_id}"),
                            use_container_width=True,
                            key=f"residuals_{unique_id}"
                        )
                    else:
                        st.info("ℹ️ Graphique des résidus non disponible")
                except Exception as e:
                    st.error(f"❌ Erreur graphique résidus: {str(e)[:100]}")

            with col2:
                st.markdown("**🎯 Prédictions vs Réelles**")
                try:
                    pred_plot = visualizer.create_predicted_vs_actual(model_result)
                    if pred_plot:
                        st.plotly_chart(
                            cached_plot(pred_plot, f"pred_{unique_id}"),
                            use_container_width=True,
                            key=f"pred_{unique_id}"
                        )
                    else:
                        st.info("ℹ️ Graphique prédictions non disponible")
                except Exception as e:
                    st.error(f"❌ Erreur graphique prédictions: {str(e)[:100]}")

        # ========================================================================
        # INFORMATIONS COMPLÉMENTAIRES
        # ========================================================================
        st.markdown("---")
        st.markdown("#### ℹ️ Informations Complémentaires")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.markdown("**⏱️ Performances**")
            training_time = model_result.get('training_time', 0)
            st.metric("Temps d'entraînement", f"{training_time:.2f}s")
            
        with col_info2:
            st.markdown("**📊 Données**")
            if model_result.get('X_train') is not None:
                n_samples = len(model_result['X_train'])
                st.metric("Échantillons d'entraînement", f"{n_samples:,}")
                
        with col_info3:
            st.markdown("**🔧 Statut**")
            if model_result.get('success', False):
                st.metric("Entraînement", "✅ Réussi")
            else:
                st.metric("Entraînement", "❌ Échoué")

    except Exception as e:
        log_structured("ERROR", f"Erreur détaillée {model_name}", {
            "error": str(e),
            "task_type": task_type,
            "model_keys": list(model_result.keys()) if isinstance(model_result, dict) else "N/A"
        })
        st.error(f"❌ Erreur critique dans l'analyse du modèle: {str(e)}")
        
        with st.expander("🔧 Détails Techniques de l'Erreur", expanded=False):
            import traceback
            st.code(traceback.format_exc())



@monitor_operation
def sync_mlflow_runs():
    """Synchronise les runs MLflow entre tous les états - VERSION COMPLÈTE"""
    try:
        log_structured("INFO", "🔄 Démarrage synchronisation MLflow")
        
        # 🎯 SOURCE 1: Session Streamlit (priorité haute)
        session_runs = getattr(st.session_state, 'mlflow_runs', [])
        if session_runs and len(session_runs) > 0:
            STATE.mlflow_runs = session_runs
            log_structured("INFO", "Runs synchronisés depuis session_state", {
                "n_runs": len(session_runs),
                "source": "session_state"
            })
            return session_runs
        
        # 🎯 SOURCE 2: TrainingState (état global)
        training_runs = getattr(STATE.training, 'mlflow_runs', [])
        if training_runs and len(training_runs) > 0:
            # Synchronise aussi vers session_state pour cohérence
            st.session_state.mlflow_runs = training_runs
            log_structured("INFO", "Runs synchronisés depuis training_state", {
                "n_runs": len(training_runs),
                "source": "training_state"
            })
            return training_runs
        
        # 🎯 SOURCE 3: Rechargement direct depuis serveur MLflow
        if MLFLOW_AVAILABLE:
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                
                # Récupère toutes les expériences
                experiments = client.search_experiments()
                all_runs = []
                
                for exp in experiments:
                    runs = client.search_runs(
                        experiment_ids=[exp.experiment_id],
                        max_results=50  # Limite pour performance
                    )
                    all_runs.extend(runs)
                
                if all_runs:
                    formatted_runs = []
                    for run in all_runs:
                        try:
                            formatted_run = {
                                'run_id': run.info.run_id,
                                'status': run.info.status,
                                'model_name': run.data.tags.get('mlflow.runName', 'Unknown'),
                                'metrics': run.data.metrics,
                                'tags': run.data.tags,
                                'start_time': run.info.start_time,
                                'end_time': run.info.end_time,
                                'experiment_id': run.info.experiment_id
                            }
                            formatted_runs.append(formatted_run)
                        except Exception as e:
                            log_structured("WARNING", "Erreur formatage run", {
                                "run_id": getattr(run.info, 'run_id', 'unknown'),
                                "error": str(e)[:100]
                            })
                    
                    # Stocke dans TOUS les états
                    STATE.mlflow_runs = formatted_runs
                    st.session_state.mlflow_runs = formatted_runs
                    
                    log_structured("INFO", "Runs rechargés depuis serveur MLflow", {
                        "n_runs": len(formatted_runs),
                        "source": "mlflow_server"
                    })
                    return formatted_runs
                    
            except Exception as e:
                log_structured("ERROR", "Échec rechargement MLflow", {
                    "error": str(e),
                    "tracking_uri": mlflow.get_tracking_uri() if MLFLOW_AVAILABLE else "N/A"
                })
        
        # 🎯 SOURCE 4: Résultats d'entraînement (fallback)
        if hasattr(STATE, 'training_results') and STATE.training_results:
            try:
                results = getattr(STATE.training_results, 'results', [])
                if results and len(results) > 0:
                    # Extrait les informations des runs depuis les résultats
                    extracted_runs = []
                    for result in results:
                        if isinstance(result, dict) and result.get('success', False):
                            extracted_runs.append({
                                'run_id': f"generated_{result.get('model_name', 'unknown')}_{int(time.time())}",
                                'status': 'FINISHED',
                                'model_name': result.get('model_name', 'Unknown'),
                                'metrics': result.get('metrics', {}),
                                'tags': {'source': 'training_results'},
                                'start_time': time.time() - result.get('training_time', 0),
                                'end_time': time.time()
                            })
                    
                    if extracted_runs:
                        STATE.mlflow_runs = extracted_runs
                        st.session_state.mlflow_runs = extracted_runs
                        log_structured("INFO", "Runs extraits des résultats d'entraînement", {
                            "n_runs": len(extracted_runs),
                            "source": "training_results"
                        })
                        return extracted_runs
            except Exception as e:
                log_structured("ERROR", "Erreur extraction runs depuis résultats", {
                    "error": str(e)[:100]
                })
        
        log_structured("WARNING", "Aucun run MLflow trouvé dans aucune source")
        return []
        
    except Exception as e:
        log_structured("ERROR", "Erreur critique synchronisation MLflow", {
            "error": str(e),
            "traceback": traceback.format_exc()[:500]
        })
        return []

@monitor_operation
def create_mlflow_run_plot(runs: List[Any], task_type: str, metric_to_plot: str = None, chart_type: str = "Bar") -> Optional[go.Figure]:
    """
    Crée un graphique comparant les performances des runs MLflow.
    
    ✅ CORRECTIONS APPLIQUÉES:
    - Filtrage strict des métriques numériques
    - Conversion float explicite avec try-except
    - Limitation à 10 meilleurs modèles pour lisibilité
    """
    try:
        if not runs:
            log_structured("WARNING", "Aucun run MLflow fourni")
            return None

        valid_runs_data = []
        available_metrics = set()

        for i, run in enumerate(runs):
            # Gestion safe de différents types de runs
            if isinstance(run, dict):
                run_dict = run
            elif hasattr(run, '__dict__'):
                run_dict = run.__dict__
            elif hasattr(run, 'to_dict'):
                run_dict = run.to_dict()
            else:
                log_structured("WARNING", f"Run {i} format inconnu", {"type": str(type(run))})
                continue
            
            metrics = run_dict.get('metrics', {})
            if not metrics or not isinstance(metrics, dict):
                continue
            
            # Extraction safe du nom de modèle
            model_name = (
                run_dict.get('tags', {}).get('mlflow.runName') or
                run_dict.get('model_name') or
                run_dict.get('runName') or
                f'Modèle_{i}'
            )
            
            # ✅ CORRECTION CRITIQUE : Filtrage des métriques numériques uniquement
            numeric_metrics = {}
            for k, v in metrics.items():
                try:
                    # Tentative de conversion en float
                    v_float = float(v)
                    # Vérifier que ce n'est pas NaN ou Inf
                    if not (np.isnan(v_float) or np.isinf(v_float)):
                        numeric_metrics[k] = v_float
                except (ValueError, TypeError):
                    # Ignorer les valeurs non convertibles
                    pass
            
            if not numeric_metrics:
                continue
            
            available_metrics.update(numeric_metrics.keys())
            valid_runs_data.append({
                'model_name': str(model_name),
                'metrics': numeric_metrics
            })

        if not valid_runs_data:
            log_structured("WARNING", "Aucune donnée valide dans les runs MLflow")
            return None

        # Sélection intelligente de la métrique
        if not metric_to_plot or metric_to_plot not in available_metrics:
            if task_type == 'classification':
                metric_to_plot = next(
                    (m for m in ['accuracy', 'f1', 'precision', 'recall'] if m in available_metrics),
                    None
                )
            elif task_type == 'regression':
                metric_to_plot = next(
                    (m for m in ['r2', 'rmse', 'mae'] if m in available_metrics),
                    None
                )
            elif task_type == 'clustering':
                metric_to_plot = next(
                    (m for m in ['silhouette_score', 'calinski_harabasz_score'] if m in available_metrics),
                    None
                )
            
            if not metric_to_plot:
                # Prendre la première métrique disponible
                metric_to_plot = next(iter(available_metrics), None)
            
            if not metric_to_plot:
                log_structured("WARNING", "Aucune métrique exploitable trouvée")
                return None

        # Construction safe des données du graphique
        plot_data = []
        for run_data in valid_runs_data:
            model_name = run_data['model_name']
            metrics = run_data['metrics']
            if metric_to_plot in metrics:
                value = metrics[metric_to_plot]
                plot_data.append({'Modèle': model_name, metric_to_plot: value})

        if not plot_data:
            log_structured("WARNING", f"Aucune donnée pour la métrique {metric_to_plot}")
            return None

        df = pd.DataFrame(plot_data)
        
        # Limiter le nombre de modèles affichés pour la lisibilité
        if len(df) > 10:
            df = df.nlargest(10, metric_to_plot)
            log_structured("INFO", "Limitation aux 10 meilleurs modèles pour le graphique")

        fig = go.Figure()
        colors = px.colors.qualitative.Plotly[:len(df)]

        if chart_type == "Radar":
            # Radar chart avec valeurs normalisées
            for run_data in valid_runs_data[:10]:
                model_name = run_data['model_name']
                metrics = run_data['metrics']
                valid_metrics = [m for m in available_metrics if m in metrics]
                values = [float(metrics[m]) for m in valid_metrics]
                if values:
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=valid_metrics,
                        fill='toself',
                        name=model_name,
                        line=dict(color=colors[valid_runs_data.index(run_data) % len(colors)])
                    ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                title=f"Comparaison des Modèles - Radar ({task_type.capitalize()})",
                template=VISUALIZATION_CONSTANTS.get("PLOTLY_TEMPLATE", "plotly_white"),
                height=500
            )

        elif chart_type == "Line":
            fig.add_trace(go.Scatter(
                x=df['Modèle'],
                y=df[metric_to_plot],
                mode='lines+markers+text',
                name=metric_to_plot,
                line=dict(color=colors[0], width=2),
                marker=dict(size=10),
                text=df[metric_to_plot].round(3),
                textposition='top center'
            ))
            fig.update_layout(
                title=f"Évolution - {metric_to_plot} ({task_type.capitalize()})",
                xaxis_title="Modèles",
                yaxis_title=metric_to_plot,
                template=VISUALIZATION_CONSTANTS.get("PLOTLY_TEMPLATE", "plotly_white"),
                height=500,
                xaxis=dict(tickangle=45, tickfont=dict(size=11)),
                margin=dict(b=120, t=80)
            )

        else:  # Bar (par défaut)
            fig.add_trace(go.Bar(
                x=df['Modèle'],
                y=df[metric_to_plot],
                name=metric_to_plot,
                marker_color=colors,
                text=df[metric_to_plot].round(3),
                textposition='auto',
                textfont=dict(size=10)
            ))
            fig.update_layout(
                title=f"Comparaison - {metric_to_plot} ({task_type.capitalize()})",
                xaxis_title="Modèles",
                yaxis_title=metric_to_plot,
                template=VISUALIZATION_CONSTANTS.get("PLOTLY_TEMPLATE", "plotly_white"),
                height=500,
                showlegend=False,
                xaxis=dict(tickangle=45, tickfont=dict(size=11)),
                margin=dict(b=120, t=80)
            )

        log_structured("INFO", "Graphique MLflow généré avec succès", {
            "n_models": len(df),
            "metric": metric_to_plot,
            "chart_type": chart_type
        })
        return fig
        
    except Exception as e:
        log_structured("ERROR", "Erreur création graphique MLflow", {
            "error": str(e)[:200],
            "traceback": str(e)
        })
        return None

@monitor_operation
def display_mlflow_tab():
    """Affiche l'onglet MLflow avec synchronisation forcée - VERSION CORRIGÉE"""
    st.markdown("### 🔗 Exploration des Runs MLflow")
    
    # 🎯 SYNCHRONISATION FORCÉE IMMÉDIATE
    mlflow_runs = sync_mlflow_runs()
    
    if not MLFLOW_AVAILABLE:
        st.error("🚫 MLflow non disponible")
        st.info("Installez MLflow pour accéder aux runs: `pip install mlflow`")
        return
    
    # 🔍 DIAGNOSTIC COMPLET
    with st.expander("🔧 Diagnostic MLflow Détaillé", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Sources des runs:**")
            diagnostic_data = {
                "session_state_runs": len(getattr(st.session_state, 'mlflow_runs', [])),
                "state_runs": len(getattr(STATE, 'mlflow_runs', [])),
                "training_state_runs": len(getattr(STATE.training, 'mlflow_runs', [])),
                "sync_runs": len(mlflow_runs),
                "mlflow_available": MLFLOW_AVAILABLE
            }
            st.json(diagnostic_data)
        
        with col2:
            st.write("**🔄 Actions:**")
            
            if st.button("🔄 Recharger depuis MLflow Server", key="reload_mlflow_server", use_container_width=True):
                with st.spinner("Rechargement depuis serveur MLflow..."):
                    if MLFLOW_AVAILABLE:
                        try:
                            from mlflow.tracking import MlflowClient
                            client = MlflowClient()
                            experiments = client.search_experiments()
                            all_runs = []
                            
                            for exp in experiments:
                                runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=50)
                                all_runs.extend(runs)
                            
                            if all_runs:
                                formatted_runs = []
                                for run in all_runs:
                                    formatted_runs.append({
                                        'run_id': run.info.run_id,
                                        'status': run.info.status,
                                        'model_name': run.data.tags.get('mlflow.runName', 'Unknown'),
                                        'metrics': run.data.metrics,
                                        'tags': run.data.tags,
                                        'start_time': run.info.start_time,
                                        'end_time': run.info.end_time
                                    })
                                
                                # Stockage multi-état
                                STATE.mlflow_runs = formatted_runs
                                st.session_state.mlflow_runs = formatted_runs
                                STATE.training.mlflow_runs = formatted_runs
                                
                                st.success(f"✅ {len(formatted_runs)} runs rechargés")
                                st.rerun()
                            else:
                                st.warning("Aucun run trouvé sur le serveur")
                        except Exception as e:
                            st.error(f"Erreur rechargement: {str(e)[:200]}")
            
            if st.button("🗑️ Nettoyer les runs", key="clear_mlflow_runs", use_container_width=True):
                STATE.mlflow_runs = []
                st.session_state.mlflow_runs = []
                STATE.training.mlflow_runs = []
                st.success("Runs nettoyés")
                st.rerun()
    
    # 🎯 AFFICHAGE PRINCIPAL
    if not mlflow_runs:
        st.warning("⚠️ Aucun run MLflow disponible")
        st.info("""
        **Pour générer des runs MLflow:**
        1. Allez dans l'onglet **Configuration ML**
        2. Chargez un dataset et configurez l'entraînement
        3. **Assurez-vous que MLflow est activé** dans les paramètres avancés
        4. Lancez l'entraînement des modèles
        5. Revenez sur cette page pour voir les runs
        """)
        
        # Vérification supplémentaire
        if hasattr(STATE, 'training_results') and STATE.training_results:
            st.info("💡 **Des résultats d'entraînement sont disponibles mais pas de runs MLflow.**")
            st.info("Activez MLflow dans les paramètres d'entraînement pour le prochain lancement.")
        
        return
    
    st.success(f"**📊 {len(mlflow_runs)} runs MLflow disponibles**")

    # ========================================================================
    # 🆕 TRADUCTION DES STATUTS MLflow (ANGLAIS → FRANÇAIS)
    # ========================================================================
    STATUS_TRANSLATION = {
        'FINISHED': 'FINI',
        'RUNNING': 'EN COURS',
        'FAILED': 'ÉCHEC',
        'SCHEDULED': 'PROGRAMMÉ',
        'KILLED': 'ARRÊTÉ'
    }
    
    # Normalisation des statuts dans les runs
    for run in mlflow_runs:
        if isinstance(run, dict) and 'status' in run:
            original_status = run['status']
            # Traduction si nécessaire
            if original_status in STATUS_TRANSLATION:
                run['status'] = STATUS_TRANSLATION[original_status]
    
    # Statuts disponibles en FRANÇAIS pour le filtre
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.multiselect(
            "Filtrer par statut",
            options=['FINI', 'EN COURS', 'ÉCHEC', 'PROGRAMMÉ', 'ARRÊTÉ'],
            default=['FINI'],  # Par défaut uniquement les runs terminés
            key="mlflow_status_filter"
        )
    
    with col2:
        # Extraction safe des noms de modèles
        model_names = []
        for run in mlflow_runs:
            if isinstance(run, dict):
                name = run.get('tags', {}).get('mlflow.runName') or run.get('model_name', 'Unknown')
                if name and name != 'Unknown':
                    model_names.append(name)
        
        model_names = sorted(set(model_names)) if model_names else ['Tous']
        
        model_filter = st.multiselect(
            "Filtrer par modèle",
            options=model_names,
            default=model_names,
            key="mlflow_model_filter"
        )
    
    with col3:
        # ========================================================================
        # 🆕 EXTRACTION STRICTE DES MÉTRIQUES NUMÉRIQUES
        # ========================================================================
        available_metrics = set()
        for run in mlflow_runs:
            if isinstance(run, dict) and 'metrics' in run and isinstance(run['metrics'], dict):
                for key, value in run['metrics'].items():
                    # Validation stricte que c'est un nombre
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        try:
                            # Vérifier que ce n'est pas NaN ou Inf
                            v_float = float(value)
                            if not (np.isnan(v_float) or np.isinf(v_float)):
                                available_metrics.add(key)
                        except (ValueError, TypeError):
                            pass
        
        if not available_metrics:
            st.warning("⚠️ Aucune métrique numérique disponible")
            return
        
        metric_to_plot = st.selectbox(
            "Métrique à afficher",
            options=sorted(available_metrics),
            key="mlflow_metric_selector"
        )
    
    # ========================================================================
    # FILTRAGE DES RUNS
    # ========================================================================
    filtered_runs = []
    for run in mlflow_runs:
        if not isinstance(run, dict):
            continue
        
        # Vérification statut (déjà traduit)
        run_status = run.get('status', 'INCONNU')
        if run_status not in status_filter:
            continue
        
        # Vérification nom modèle
        run_name = run.get('tags', {}).get('mlflow.runName') or run.get('model_name', 'Unknown')
        if model_filter and run_name not in model_filter:
            continue
        
        filtered_runs.append(run)
    
    if not filtered_runs:
        st.info("ℹ️ Aucun run ne correspond aux filtres sélectionnés")
        return
    
    # ========================================================================
    # AFFICHAGE TABLEAU DES RUNS
    # ========================================================================
    st.markdown("#### 📋 Liste des Runs")
    run_data = []
    for run in filtered_runs:
        run_id = run.get('run_id', 'N/A')
        status = run.get('status', 'INCONNU')
        model_name = run.get('tags', {}).get('mlflow.runName') or run.get('model_name', 'Unknown')
        metrics = run.get('metrics', {})
        
        row = {
            'Run ID': run_id[:8] + '...' if isinstance(run_id, str) and len(run_id) > 8 else str(run_id),
            'Modèle': model_name,
            'Statut': status
        }
        
        # Ajout safe des métriques avec validation numérique
        for metric in available_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, (int, float, np.integer, np.floating)):
                    try:
                        v_float = float(value)
                        if not (np.isnan(v_float) or np.isinf(v_float)):
                            row[metric] = f"{v_float:.3f}"
                        else:
                            row[metric] = 'N/A'
                    except (ValueError, TypeError):
                        row[metric] = 'N/A'
                else:
                    row[metric] = 'N/A'
            else:
                row[metric] = 'N/A'
        
        run_data.append(row)
    
    if run_data:
        st.dataframe(pd.DataFrame(run_data), use_container_width=True)
    else:
        st.info("ℹ️ Aucune donnée à afficher")
        return
    
    # ========================================================================
    # GRAPHIQUE DE COMPARAISON
    # ========================================================================
    st.markdown("#### 📈 Comparaison des Performances")
    
    # Sélection du type de graphique
    col_chart1, col_chart2 = st.columns([3, 1])
    with col_chart2:
        chart_type = st.selectbox(
            "Type de graphique",
            options=["Bar", "Line", "Radar"],
            key="mlflow_chart_type"
        )
    
    task_type = getattr(STATE, 'task_type', 'classification')
    
    # Création du graphique avec runs FILTRÉS et métriques validées
    mlflow_plot = create_mlflow_run_plot(
        filtered_runs, 
        task_type, 
        metric_to_plot,
        chart_type
    )
    
    if mlflow_plot:
        st.plotly_chart(
            cached_plot(mlflow_plot, f"mlflow_comparison_{metric_to_plot}_{chart_type}"),
            use_container_width=True,
            key=f"mlflow_comparison_{metric_to_plot}_{chart_type}_{int(time.time())}"
        )
    else:
        st.info(f"ℹ️ Données insuffisantes pour générer le graphique de {metric_to_plot}")
    
    # ========================================================================
    # 🆕 STATISTIQUES SUPPLÉMENTAIRES
    # ========================================================================
    with st.expander("📊 Statistiques des Runs", expanded=False):
        if filtered_runs and metric_to_plot:
            values = []
            for run in filtered_runs:
                metrics = run.get('metrics', {})
                if metric_to_plot in metrics:
                    value = metrics[metric_to_plot]
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        try:
                            v_float = float(value)
                            if not (np.isnan(v_float) or np.isinf(v_float)):
                                values.append(v_float)
                        except (ValueError, TypeError):
                            pass
            
            if values:
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                with stat_col1:
                    st.metric("Moyenne", f"{np.mean(values):.3f}")
                with stat_col2:
                    st.metric("Médiane", f"{np.median(values):.3f}")
                with stat_col3:
                    st.metric("Min", f"{np.min(values):.3f}")
                with stat_col4:
                    st.metric("Max", f"{np.max(values):.3f}")
            else:
                st.info("ℹ️ Aucune statistique disponible")

def display_overview_tab(validation_result: Dict[str, Any], results_data: List[Dict]):
    """Affiche l'onglet Vue d'ensemble"""
    st.markdown("### 📊 Vue d'Ensemble des Performances")
    
    # Métriques globales
    successful_models = validation_result.get("successful_models", [])
    failed_models = validation_result.get("failed_models", [])
    
    if not successful_models:
        st.info("ℹ️ Aucun modèle à afficher dans la vue d'ensemble")
        return
    
    # Graphique de comparaison des modèles
    st.markdown("#### 📈 Comparaison des Modèles")
    
    # Préparation des données pour le graphique
    model_names = []
    accuracies = []
    
    for model in successful_models:
        name = model.get('model_name', 'Unknown')
        metrics = model.get('metrics', {})
        accuracy = metrics.get('accuracy', 0)
        
        model_names.append(name)
        accuracies.append(accuracy)
    
    if model_names and accuracies:
        # Création du graphique
        fig = go.Figure(data=[
            go.Bar(
                x=model_names,
                y=accuracies,
                text=[f'{acc:.3f}' for acc in accuracies],
                textposition='auto',
                marker_color=['#28a745' if acc == max(accuracies) else '#17a2b8' for acc in accuracies]
            )
        ])
        
        fig.update_layout(
            title="Comparaison de l'Accuracy des Modèles",
            xaxis_title="Modèles",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1]),
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tableau récapitulatif
    st.markdown("#### 📋 Tableau Récapitulatif")
    
    summary_data = []
    for model in successful_models:
        metrics = model.get('metrics', {})
        summary_data.append({
            'Modèle': model.get('model_name', 'Unknown'),
            'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
            'Precision': f"{metrics.get('precision', 0):.3f}",
            'Recall': f"{metrics.get('recall', 0):.3f}",
            'F1-Score': f"{metrics.get('f1_score', 0):.3f}",
            'Temps (s)': f"{model.get('training_time', 0):.2f}"
        })
    
    if summary_data:
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

def display_metrics_tab(validation_result: Dict[str, Any]):
    """Affiche l'onglet Métriques détaillées"""
    st.markdown("### 📈 Analyse des Métriques Détaillées")
    
    successful_models = validation_result.get("successful_models", [])
    
    if not successful_models:
        st.info("ℹ️ Aucune métrique détaillée à afficher")
        return
    
    # Sélection du modèle pour l'analyse détaillée
    model_names = [m.get('model_name', f'Modèle_{i}') for i, m in enumerate(successful_models)]
    selected_model_name = st.selectbox(
        "Sélectionnez un modèle pour voir ses métriques détaillées:",
        options=model_names,
        key="metrics_model_selector"
    )
    
    # Trouver le modèle sélectionné
    selected_model = None
    for model in successful_models:
        if model.get('model_name') == selected_model_name:
            selected_model = model
            break
    
    if selected_model:
        metrics = selected_model.get('metrics', {})
        
        st.markdown(f"#### 📊 Métriques pour **{selected_model_name}**")
        
        # Affichage structuré des métriques
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Métriques Principales**")
            if 'accuracy' in metrics:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            if 'precision' in metrics:
                st.metric("Precision", f"{metrics['precision']:.3f}")
            if 'recall' in metrics:
                st.metric("Recall", f"{metrics['recall']:.3f}")
            if 'f1_score' in metrics:
                st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
        
        with col2:
            st.markdown("**📊 Métriques Avancées**")
            if 'roc_auc' in metrics:
                st.metric("AUC-ROC", f"{metrics['roc_auc']:.3f}")
            if 'confusion_matrix' in metrics:
                st.metric("Matrice de Confusion", "Disponible")
            if 'classification_report' in metrics:
                st.metric("Rapport de Classification", "Disponible")
        
        # Affichage des métriques brutes
        with st.expander("🔍 Métriques Brutes (JSON)", expanded=False):
            st.json(metrics)

@monitor_operation
def main():
    """Fonction principale de la page d'évaluation - VERSION PRODUCTION"""
    try:
        # ========================================================================
        # VALIDATION RENFORCÉE DES DONNÉES D'ENTRÉE
        # ========================================================================
        st.markdown('<div class="main-header">📈 Évaluation des Modèles</div>', unsafe_allow_html=True)
        
        # Vérification de base
        if not hasattr(STATE, 'training_results') or STATE.training_results is None:
            st.error("🚫 Aucun résultat d'entraînement disponible")
            st.info("""
            **Pour utiliser cette page :**
            1. Allez dans l'onglet **'Configuration ML'**
            2. Chargez un dataset et configurez l'entraînement  
            3. Lancez l'entraînement des modèles
            4. Revenez sur cette page pour analyser les résultats
            """)
            if st.button("⚙️ Aller à l'Entraînement", type="primary", use_container_width=True):
                st.switch_page("pages/2_training.py")
            return

        # Validation du type d'objet
        training_results = STATE.training_results
        if not hasattr(training_results, 'results'):
            st.error("❌ Format invalide des résultats d'entraînement")
            logger.error("training_results n'a pas d'attribut 'results'", extra={
                "type": type(training_results).__name__,
                "attributes": dir(training_results)
            })
            return

        # Extraction sécurisée des données
        results_data = training_results.results
        if not results_data or not isinstance(results_data, list):
            st.error("📭 Aucun résultat détaillé disponible")
            st.info("Les résultats d'entraînement semblent vides. Veuillez relancer l'entraînement.")
            return

        # ========================================================================
        # INITIALISATION DES COMPOSANTS
        # ========================================================================
        task_type = getattr(STATE, 'task_type', 'classification')
        st.success(f"🔧 **Type de tâche détecté :** {task_type.upper()}")

        # Création du visualizer avec gestion d'erreur
        try:
            visualizer = ModelEvaluationVisualizer(results_data)
            validation_result = visualizer.validation_result
            validation_result["task_type"] = task_type  # Override cohérent
        except Exception as e:
            st.error(f"❌ Erreur lors de l'initialisation du visualiseur : {str(e)}")
            logger.error("Erreur initialisation ModelEvaluationVisualizer", extra={"error": str(e)})
            return

        # Validation finale des résultats
        if not validation_result.get("has_results", False):
            st.error("📭 Aucune donnée valide trouvée dans les résultats")
            with st.expander("🔍 Debug des données", expanded=False):
                st.json({
                    "results_count": len(results_data),
                    "validation_result_keys": list(validation_result.keys()),
                    "first_result_keys": list(results_data[0].keys()) if results_data else []
                })
            return

        # ========================================================================
        # AFFICHAGE PRINCIPAL
        # ========================================================================
        display_metrics_header(validation_result)

        # Configuration des onglets
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Vue d'Ensemble", 
            "🔍 Détails Modèles",
            "📈 Métriques", 
            "💾 Export", 
            "🔗 MLflow"
        ])

        # ========================================================================
        # ONGLET 1 : VUE D'ENSEMBLE (NOUVEAU)
        # ========================================================================
        with tab1:
            display_overview_tab(validation_result, results_data)

        # ========================================================================
        # ONGLET 2 : DÉTAILS DES MODÈLES (EXISTANT)
        # ========================================================================
        with tab2:
            st.markdown("### 🔍 Analyse Détaillée par Modèle")
            
            # Extraction sécurisée des modèles réussis
            successful_models = []
            for result in results_data:
                if isinstance(result, dict) and result.get('success', False):
                    successful_models.append(result)
            
            if successful_models:
                # Sélection du modèle
                model_names = []
                for i, model in enumerate(successful_models):
                    name = model.get('model_name', f'Modèle_{i}')
                    model_names.append(name)
                
                selected_idx = st.selectbox(
                    "Sélectionnez un modèle à analyser :",
                    range(len(model_names)),
                    format_func=lambda x: model_names[x],
                    key="model_selector_main"
                )
                
                if 0 <= selected_idx < len(successful_models):
                    selected_model = successful_models[selected_idx]
                    display_model_details(visualizer, selected_model, task_type)
                else:
                    st.error("❌ Index de modèle invalide")
            else:
                st.warning("⚠️ Aucun modèle n'a terminé avec succès l'entraînement")

        # ========================================================================
        # ONGLET 3 : MÉTRIQUES DÉTAILLÉES (NOUVEAU)
        # ========================================================================
        with tab3:
            display_metrics_tab(validation_result)

        # ========================================================================
        # ONGLET 4 : EXPORT (CORRIGÉ)
        # ========================================================================
        with tab4:
            st.markdown("### 💾 Export des Résultats")
            
            if successful_models:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Données Structurées")
                    try:
                        df_comparison = visualizer.get_comparison_dataframe()
                        if df_comparison is not None and not df_comparison.empty:
                            csv_data = df_comparison.to_csv(index=False, encoding='utf-8')
                            st.download_button(
                                label="📥 Télécharger CSV",
                                data=csv_data,
                                file_name=f"comparaison_modeles_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        else:
                            st.warning("⚠️ Aucune donnée pour l'export CSV")
                    except Exception as e:
                        st.error(f"❌ Erreur génération CSV : {str(e)}")
                    
                    try:
                        export_data = visualizer.get_export_data()
                        if export_data:
                            json_data = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
                            st.download_button(
                                label="📥 Télécharger JSON",
                                data=json_data,
                                file_name=f"evaluation_complete_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        else:
                            st.warning("⚠️ Aucune donnée pour l'export JSON")
                    except Exception as e:
                        st.error(f"❌ Erreur génération JSON : {str(e)}")
                
                with col2:
                    st.markdown("#### 📈 Rapports Détaillés")
                    
                    # Export PDF pour le meilleur modèle
                    best_model_name = validation_result.get("best_model")
                    if best_model_name:
                        best_model_result = None
                        for model in successful_models:
                            if model.get('model_name') == best_model_name:
                                best_model_result = model
                                break
                        
                        if best_model_result:
                            with st.spinner("🔄 Préparation du rapport PDF..."):
                                pdf_bytes = create_pdf_report_latex(best_model_result, task_type)
                                
                            if pdf_bytes:
                                st.download_button(
                                    label="📄 Rapport PDF (Meilleur Modèle)",
                                    data=pdf_bytes,
                                    file_name=f"rapport_{best_model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                            else:
                                st.warning("⚠️ Impossible de générer le rapport PDF")
                        else:
                            st.info("ℹ️ Meilleur modèle non trouvé dans les résultats")
                    else:
                        st.info("ℹ️ Aucun meilleur modèle identifié")
                
                # Aperçu des données
                with st.expander("👁️ Aperçu des Données Exportables", expanded=False):
                    try:
                        preview_data = visualizer.get_export_data()
                        if preview_data:
                            st.json(preview_data)
                        else:
                            st.info("ℹ️ Aucune donnée d'export disponible")
                    except Exception as e:
                        st.error(f"❌ Erreur aperçu : {str(e)}")
            else:
                st.info("ℹ️ Aucune donnée disponible pour l'export")

        # ========================================================================
        # ONGLET 5 : MLflow
        # ========================================================================
        with tab5:
            display_mlflow_tab()


        # 🎯 DIAGNOSTIC MLflow DANS SIDEBAR
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔍 Diagnostic MLflow")
        
        mlflow_status = {
            "MLflow disponible": MLFLOW_AVAILABLE,
            "Runs session_state": len(getattr(st.session_state, 'mlflow_runs', [])),
            "Runs STATE": len(getattr(STATE, 'mlflow_runs', [])),
            "Runs training_state": len(getattr(STATE.training, 'mlflow_runs', [])),
            "Tracking URI": mlflow.get_tracking_uri() if MLFLOW_AVAILABLE else "N/A",
            "Dernière synchro": datetime.now().strftime('%H:%M:%S')
        }
        
        with st.sidebar.expander("📊 Statut MLflow", expanded=False):
            st.json(mlflow_status)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Synchro", key="sidebar_sync_mlflow"):
                    sync_mlflow_runs()
                    st.rerun()
            with col2:
                if st.button("📋 Logs", key="sidebar_mlflow_logs"):
                    st.info("Vérifiez les logs pour les détails de synchronisation")

        # ========================================================================
        # GESTION DES AVERTISSMENTS
        # ========================================================================
        if hasattr(STATE, 'warnings') and STATE.warnings:
            with st.expander("⚠️ Avertissements Système", expanded=False):
                for warning in STATE.warnings:
                    st.warning(warning)
            if st.button("🗑️ Effacer les Avertissements", use_container_width=True):
                STATE.warnings = []
                st.rerun()

        # Nettoyage mémoire
        gc.collect()

    except Exception as e:
        log_structured("ERROR", "Erreur critique dans main()", {
            "error": str(e),
            "traceback": traceback.format_exc()[:500]
        })
        st.error("❌ Erreur critique dans la page d'évaluation")
        
        with st.expander("🔧 Détails Techniques (Debug)", expanded=False):
            st.code(traceback.format_exc())
            
        st.info("""
        **Solutions possibles :**
        - Rechargez la page (F5)
        - Retournez à l'entraînement et relancez-le
        - Vérifiez les logs pour plus de détails
        """)

        if st.button("🔄 Redémarrer l'Application", type="primary"):
            st.rerun()

# ========================================================================
# POINT D'ENTRÉE STREAMLIT - VERSION WINDOWS COMPATIBLE
# ========================================================================

def safe_main():
    """
    Point d'entrée sécurisé sans signaux UNIX
    """
    try:
        log_structured("INFO", "🚀 Démarrage page évaluation")
        main()
        
    except Exception as e:
        log_structured("CRITICAL", "Erreur critique", {
            "error": str(e),
            "traceback": traceback.format_exc()[:500]
        })
        
        st.error("💥 Erreur critique dans l'application")
        st.info("Veuillez recharger la page ou retourner à l'entraînement.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Redémarrer", type="primary", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("⚙️ Aller à l'Entraînement", use_container_width=True):
                st.switch_page("pages/2_training.py")

# === POINT D'ENTRÉE OBLIGATOIRE ===
if __name__ == "__main__":
    safe_main()
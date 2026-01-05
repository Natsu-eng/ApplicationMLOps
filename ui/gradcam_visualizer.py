"""
ui/gradcam_visualizer.py - INTERFACE GRAD-CAM STREAMLIT

Features:
- Sélection d'images à visualiser
- Choix du mode (overlay, heatmap, side-by-side)
- Réglage transparence
- Export des visualisations
- Batch processing
"""

import streamlit as st
import numpy as np
import torch
from typing import Optional, List
import matplotlib.pyplot as plt
from src.evaluation.grad_cam import generate_gradcam_visualization
from src.shared.logging import get_logger

logger = get_logger(__name__)


def render_gradcam_visualizer(STATE):
    """
    Affiche l'interface Grad-CAM dans Streamlit.
    
    Args:
        STATE: StateManager avec model, X_test, y_test, predictions
    """
    st.header("🔥 Grad-CAM - Visualisation des Zones d'Attention")
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    if not hasattr(STATE, 'trained_model') or STATE.trained_model is None:
        st.warning("⚠️ Aucun modèle entraîné. Entraînez d'abord un modèle.")
        return
    
    if not hasattr(STATE, 'data') or STATE.data.X_test is None:
        st.warning("⚠️ Aucune donnée de test. Chargez d'abord un dataset.")
        return
    
    model = STATE.trained_model
    X_test = STATE.data.X_test
    y_test = STATE.data.y_test
    
    # Récupération prédictions si disponibles
    predictions = None
    if hasattr(STATE, 'predictions') and STATE.predictions is not None:
        if isinstance(STATE.predictions, dict) and 'y_pred_binary' in STATE.predictions:
            predictions = STATE.predictions['y_pred_binary']
        elif isinstance(STATE.predictions, np.ndarray):
            predictions = STATE.predictions
    
    # Récupération noms de classes
    class_names = None
    if hasattr(STATE, 'training_results') and STATE.training_results:
        if isinstance(STATE.training_results, dict):
            class_names = STATE.training_results.get('class_names')
    
    if class_names is None and hasattr(STATE.data, 'class_names'):
        class_names = STATE.data.class_names
    
    # ========================================================================
    # INTERFACE
    # ========================================================================
    
    st.markdown("""
    ### 📊 Qu'est-ce que Grad-CAM ?
    
    **Grad-CAM** (Gradient-weighted Class Activation Mapping) visualise les régions 
    de l'image qui ont le plus influencé la prédiction du modèle.
    
    - 🔴 **Rouge/Jaune**: Régions importantes (forte activation)
    - 🔵 **Bleu/Violet**: Régions peu importantes (faible activation)
    """)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Mode de visualisation
        mode = st.selectbox(
            "🎨 Mode de visualisation",
            options=["overlay", "side_by_side", "heatmap"],
            index=0,
            help=(
                "**overlay**: Superposition heatmap sur image\n"
                "**side_by_side**: Image | Heatmap | Overlay\n"
                "**heatmap**: Heatmap seule"
            )
        )
    
    with col2:
        # Transparence
        alpha = st.slider(
            "🎚️ Transparence heatmap",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0 = image visible, 1 = heatmap visible"
        )
    
    with col3:
        # Nombre d'images
        n_samples = st.number_input(
            "📊 Nombre d'images",
            min_value=1,
            max_value=min(20, len(X_test)),
            value=min(5, len(X_test)),
            step=1
        )
    
    # ========================================================================
    # SÉLECTION D'IMAGES
    # ========================================================================
    
    st.markdown("### 🖼️ Sélection des images")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selection_mode = st.radio(
            "Mode de sélection",
            options=["Aléatoire", "Premières N", "Indices spécifiques", "Erreurs uniquement"],
            index=0
        )
    
    with col2:
        if selection_mode == "Indices spécifiques":
            indices_str = st.text_input(
                "Indices (séparés par des virgules)",
                value="0,1,2,3,4",
                help="Ex: 0,5,10,15,20"
            )
    
    # ========================================================================
    # GÉNÉRATION
    # ========================================================================
    
    if st.button("🔥 Générer Grad-CAM", type="primary", use_container_width=True):
        with st.spinner("Génération des visualisations Grad-CAM..."):
            try:
                # Sélection des indices
                if selection_mode == "Aléatoire":
                    indices = np.random.choice(len(X_test), size=n_samples, replace=False)
                
                elif selection_mode == "Premières N":
                    indices = np.arange(n_samples)
                
                elif selection_mode == "Indices spécifiques":
                    indices = [int(i.strip()) for i in indices_str.split(",")]
                    indices = [i for i in indices if 0 <= i < len(X_test)]
                
                elif selection_mode == "Erreurs uniquement":
                    if predictions is not None and y_test is not None:
                        error_indices = np.where(predictions != y_test)[0]
                        if len(error_indices) == 0:
                            st.warning("⚠️ Aucune erreur de prédiction trouvée!")
                            return
                        indices = error_indices[:n_samples]
                    else:
                        st.error("❌ Prédictions non disponibles")
                        return
                
                # Extraction des données
                X_selected = X_test[indices]
                y_selected = y_test[indices] if y_test is not None else None
                pred_selected = predictions[indices] if predictions is not None else None
                
                # Génération Grad-CAM
                result = generate_gradcam_visualization(
                    model=model,
                    images=X_selected,
                    predictions=pred_selected,
                    ground_truths=y_selected,
                    class_names=class_names,
                    target_layer=None,  # Auto-détection
                    mode=mode,
                    alpha=alpha,
                    use_cuda=torch.cuda.is_available()
                )
                
                if result['success']:
                    st.success(f"✅ {len(indices)} visualisations Grad-CAM générées!")
                    
                    # Affichage de la figure
                    st.pyplot(result['figure'])
                    
                    # Métriques
                    st.markdown("### 📊 Statistiques")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Images visualisées", len(indices))
                    
                    with col2:
                        if pred_selected is not None and y_selected is not None:
                            accuracy = (pred_selected == y_selected).mean()
                            st.metric("Accuracy (échantillon)", f"{accuracy:.2%}")
                    
                    with col3:
                        if pred_selected is not None and y_selected is not None:
                            errors = (pred_selected != y_selected).sum()
                            st.metric("Erreurs détectées", errors)
                    
                    # Export
                    st.markdown("### 💾 Export")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("📥 Télécharger Figure (PNG)"):
                            import io
                            buf = io.BytesIO()
                            result['figure'].savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            
                            st.download_button(
                                label="💾 Télécharger PNG",
                                data=buf,
                                file_name=f"gradcam_{mode}.png",
                                mime="image/png"
                            )
                    
                    with col2:
                        if st.button("📥 Télécharger CAMs (NPZ)"):
                            import io
                            buf = io.BytesIO()
                            np.savez_compressed(
                                buf,
                                cams=np.array(result['cams']),
                                indices=indices,
                                predictions=pred_selected,
                                ground_truths=y_selected
                            )
                            buf.seek(0)
                            
                            st.download_button(
                                label="💾 Télécharger NPZ",
                                data=buf,
                                file_name="gradcam_data.npz",
                                mime="application/x-npz"
                            )
                    
                    # Stockage dans STATE
                    STATE.gradcam_results = result
                    STATE.gradcam_indices = indices
                    
                else:
                    st.error(f"❌ Erreur génération Grad-CAM: {result.get('error', 'Inconnue')}")
            
            except Exception as e:
                st.error(f"❌ Erreur inattendue: {e}")
                logger.error(f"Erreur Grad-CAM UI: {e}", exc_info=True)
    
    # ========================================================================
    # AIDE
    # ========================================================================
    
    with st.expander("ℹ️ À propos de Grad-CAM"):
        st.markdown("""
        ### 🔬 Comment ça marche ?
        
        1. **Forward Pass**: L'image passe dans le modèle
        2. **Backward Pass**: Calcul des gradients par rapport à la classe prédite
        3. **Pondération**: Les activations sont pondérées par les gradients
        4. **Visualisation**: Génération d'une heatmap montrant les zones importantes
        
        ### 📚 Références
        
        - [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
        - [Grad-CAM++](https://arxiv.org/abs/1710.11063)
        - [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)
        
        ### ⚠️ Limitations
        
        - Fonctionne uniquement avec des CNNs (modèles convolutionnels)
        - Ne montre pas toujours toutes les zones importantes
        - Peut être trompeur pour des modèles mal entraînés
        """)


# ============================================================================
# INTÉGRATION DANS PAGE ÉVALUATION
# ============================================================================

def add_gradcam_section_to_evaluation(STATE):
    """
    Ajoute une section Grad-CAM à la page d'évaluation.
    
    À appeler dans pages/5_anomaly_evaluation.py
    """
    st.markdown("---")
    st.header("🔥 Visualisation des Zones d'Attention (Grad-CAM)")
    
    # Check si modèle compatible
    model_type = STATE.training_results.get('model_type', '') if hasattr(STATE, 'training_results') else ''
    
    compatible_models = [
        'transfer_learning', 'resnet', 'vgg', 'efficientnet',
        'mobilenet', 'densenet', 'simple_cnn', 'custom_resnet'
    ]
    
    is_compatible = any(m in model_type.lower() for m in compatible_models)
    
    if not is_compatible:
        st.info(
            f"ℹ️ Grad-CAM n'est pas disponible pour ce type de modèle ({model_type}).\n\n"
            f"Modèles compatibles: CNN, ResNet, VGG, EfficientNet, MobileNet, DenseNet"
        )
        return
    
    # Affichage de l'interface
    render_gradcam_visualizer(STATE)
"""
Helpers simples pour l'affichage des images dans Streamlit.
"""

import streamlit as st
import os
from PIL import Image
from src.explorations.image_exploration_plots import detect_dataset_structure, _get_image_files, get_dataset_info

def preview_images(data_dir: str, max_images: int = 5):
    """
    Version compatible ancien code - utilise la nouvelle fonction.
    """
    show_dataset_preview(data_dir, max_images)

def show_dataset_preview(data_dir: str, max_images: int = 6):
    """
    Affiche un aperçu intelligent du dataset.
    """
    if not os.path.exists(data_dir):
        st.error("❌ Dossier introuvable")
        return
    
    structure = detect_dataset_structure(data_dir)
    st.info(f"**Structure détectée** : {structure['type'].replace('_', ' ').title()}")
    
    if structure["type"] == "mvtec_ad":
        _show_mvtec_preview(data_dir, max_images)
    elif structure["type"] == "categorical_folders":
        _show_categorical_preview(data_dir, max_images)
    elif structure["type"] == "flat_directory":
        _show_flat_preview(data_dir, max_images)
    else:
        st.warning("Structure non reconnue")

def _show_mvtec_preview(data_dir: str, max_images: int):
    """Aperçu pour structure MVTec AD."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 Images normales")
        normal_paths = [
            os.path.join(data_dir, "train", "good"),
            os.path.join(data_dir, "test", "good")
        ]
        for path in normal_paths:
            if os.path.exists(path):
                _show_images_from_folder(path, max_images//2)

def _show_categorical_preview(data_dir: str, max_images: int):
    """Aperçu pour dossiers catégoriels."""
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for category in categories[:4]:  # Max 4 catégories
        st.subheader(f"📁 {category}")
        path = os.path.join(data_dir, category)
        _show_images_from_folder(path, max_images)

def _show_flat_preview(data_dir: str, max_images: int):
    """Aperçu pour dossier plat."""
    st.subheader("📸 Images du dataset")
    _show_images_from_folder(data_dir, max_images)

def _show_images_from_folder(folder_path: str, max_images: int):
    """Affiche les images d'un dossier spécifique."""
    if not os.path.exists(folder_path):
        return
    
    image_files = _get_image_files(folder_path)[:max_images]
    
    if not image_files:
        st.write("*Aucune image trouvée*")
        return
    
    # Afficher en grille responsive
    cols = st.columns(min(3, len(image_files)))
    
    for idx, img_file in enumerate(image_files):
        try:
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path)
            
            # Redimensionner pour l'affichage
            img.thumbnail((200, 200))
            
            with cols[idx % len(cols)]:
                st.image(img, use_column_width=True)
                st.caption(img_file[:20] + "..." if len(img_file) > 20 else img_file)
                
        except Exception as e:
            st.error(f"Erreur image: {img_file}")

def show_quick_stats(data_dir: str):
    """
    Affiche des statistiques rapides et claires.
    """
    try:
        info = get_dataset_info(data_dir)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Structure", info["structure"].replace("_", " ").title())
        
        with col2:
            if "total" in info:
                st.metric("🖼️ Images totales", info["total"])
        
        with col3:
            if "normal" in info and "anomaly" in info:
                if info["anomaly"] > 0:
                    st.metric("🎯 Anomalies", info["anomaly"])
                else:
                    st.metric("🎯 Classes", len(info.get("categories", {})))
        
        # Détails supplémentaires
        with st.expander("📋 Détails de la structure"):
            if info["structure"] == "mvtec_ad":
                st.write(f"**Normal** : {info.get('normal', 0)} images")
                st.write(f"**Anomalie** : {info.get('anomaly', 0)} images")
            
            elif info["structure"] == "categorical_folders":
                for cat, count in info.get("categories", {}).items():
                    st.write(f"**{cat}** : {count} images")
            
            elif info["structure"] == "flat_directory":
                st.write(f"**Images** : {info.get('total', 0)} dans le dossier racine")
                
    except Exception as e:
        st.error(f"Erreur analyse: {str(e)}")
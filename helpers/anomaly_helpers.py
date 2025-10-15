"""
Fonctions utilitaires pour la détection d'anomalies visuelles.
"""

import streamlit as st
import os
from PIL import Image
from src.models.training import log_structured
from src.shared.logging import get_logger
import json
from datetime import datetime

logger = get_logger(__name__)

def preview_images(data_dir: str, max_images: int = 5):
    """
    Affiche un aperçu des images par catégorie dans Streamlit.
    
    Args:
        data_dir: Chemin du dossier MVTec AD.
        max_images: Nombre maximum d'images à afficher par catégorie.
    """
    try:
        categories = ['train/good', 'test/good'] + [f'test/{d}' for d in os.listdir(os.path.join(data_dir, 'test')) if d != 'good']
        for category in categories:
            folder_path = os.path.join(data_dir, category)
            if os.path.exists(folder_path):
                st.markdown(f"### {category}")
                image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))][:max_images]
                if not image_files:
                    st.info(f"Aucune image dans {category}")
                    continue
                cols = st.columns(min(len(image_files), max_images))
                for i, img_file in enumerate(image_files):
                    img_path = os.path.join(folder_path, img_file)
                    img = Image.open(img_path)
                    cols[i].image(img, caption=img_file, use_column_width=True)
        log_structured("INFO", "Aperçu images affiché", {
            "n_categories": len(categories),
            "n_images": len(image_files)
        })
    except Exception as e:
        log_structured("ERROR", "Erreur affichage aperçu images", {"error": str(e)[:200]})
        raise
"""
Taxonomie complète des types d'anomalies pour la détection visuelle.
Standardisé pour l'industrie manufacturière et le contrôle qualité.
"""

ANOMALY_TAXONOMY = {
    "structural": {
        "name": "Défauts Structurels",
        "description": "Anomalies affectant l'intégrité physique du produit",
        "types": {
            "scratch": {
                "name": "Rayure",
                "description": "Marque linéaire sur la surface",
                "severity_levels": ["fine", "deep", "multiple"],
                "detection_difficulty": "medium",
                "business_impact": "medium"
            },
            "crack": {
                "name": "Fissure",
                "description": "Fracture dans la matière",
                "severity_levels": ["micro", "macro", "critical"],
                "detection_difficulty": "high",
                "business_impact": "high"
            },
            "hole": {
                "name": "Trou",
                "description": "Perforation ou cavité",
                "severity_levels": ["pinhole", "medium", "large"],
                "detection_difficulty": "low",
                "business_impact": "high"
            },
            "dent": {
                "name": "Bosse",
                "description": "Déformation concave ou convexe",
                "severity_levels": ["light", "medium", "severe"],
                "detection_difficulty": "medium",
                "business_impact": "medium"
            }
        }
    },
    "visual": {
        "name": "Défauts Visuels",
        "description": "Anomalies affectant l'apparence et la finition",
        "types": {
            "stain": {
                "name": "Tache",
                "description": "Marque ou coloration localisée",
                "severity_levels": ["faint", "visible", "prominent"],
                "detection_difficulty": "medium",
                "business_impact": "low"
            },
            "discoloration": {
                "name": "Décoloration",
                "description": "Variation anormale de couleur",
                "severity_levels": ["slight", "moderate", "severe"],
                "detection_difficulty": "high",
                "business_impact": "medium"
            },
            "contamination": {
                "name": "Contamination",
                "description": "Présence de corps étrangers",
                "severity_levels": ["dust", "residue", "foreign_body"],
                "detection_difficulty": "low",
                "business_impact": "high"
            },
            "blur": {
                "name": "Flou",
                "description": "Manque de netteté ou de focus",
                "severity_levels": ["slight", "moderate", "severe"],
                "detection_difficulty": "high",
                "business_impact": "low"
            }
        }
    },
    "geometric": {
        "name": "Défauts Géométriques",
        "description": "Anomalies de forme, dimension ou alignement",
        "types": {
            "deformation": {
                "name": "Déformation",
                "description": "Altération de la forme originale",
                "severity_levels": ["warping", "bending", "twisting"],
                "detection_difficulty": "high",
                "business_impact": "high"
            },
            "misalignment": {
                "name": "Mauvais Alignement",
                "description": "Position incorrecte des composants",
                "severity_levels": ["slight", "moderate", "critical"],
                "detection_difficulty": "medium",
                "business_impact": "high"
            },
            "size_variation": {
                "name": "Variation de Taille",
                "description": "Dimension hors tolérance",
                "severity_levels": ["minor", "significant", "reject"],
                "detection_difficulty": "low",
                "business_impact": "high"
            },
            "shape_defect": {
                "name": "Défaut de Forme",
                "description": "Forme non conforme aux spécifications",
                "severity_levels": ["edge_defect", "contour_issue", "major_shape"],
                "detection_difficulty": "high",
                "business_impact": "high"
            }
        }
    }
}

# Mapping des difficultés de détection vers des scores numériques
DETECTION_DIFFICULTY_SCORES = {
    "low": 1.0,
    "medium": 1.5,
    "high": 2.0
}

# Mapping des impacts business vers des poids
BUSINESS_IMPACT_WEIGHTS = {
    "low": 1.0,
    "medium": 1.5,
    "high": 2.0
}

def get_anomaly_type_display_name(anomaly_type):
    """Retourne le nom d'affichage pour un type d'anomalie"""
    for category in ANOMALY_TAXONOMY.values():
        if anomaly_type in category["types"]:
            return category["types"][anomaly_type]["name"]
    return anomaly_type

def get_anomaly_category(anomaly_type):
    """Retourne la catégorie d'un type d'anomalie"""
    for category_id, category in ANOMALY_TAXONOMY.items():
        if anomaly_type in category["types"]:
            return category_id
    return "unknown"

def validate_anomaly_type(anomaly_type):
    """Valide qu'un type d'anomalie existe dans la taxonomie"""
    for category in ANOMALY_TAXONOMY.values():
        if anomaly_type in category["types"]:
            return True
    return False

def get_all_anomaly_types():
    """Retourne tous les types d'anomalies disponibles"""
    all_types = []
    for category in ANOMALY_TAXONOMY.values():
        all_types.extend(list(category["types"].keys()))
    return all_types

def get_anomaly_types_by_category(category_id):
    """Retourne les types d'anomalies d'une catégorie spécifique"""
    if category_id in ANOMALY_TAXONOMY:
        return list(ANOMALY_TAXONOMY[category_id]["types"].keys())
    return []

def get_anomaly_difficulty_score(anomaly_type):
    """Retourne le score de difficulté pour un type d'anomalie"""
    category = get_anomaly_category(anomaly_type)
    if category in ANOMALY_TAXONOMY and anomaly_type in ANOMALY_TAXONOMY[category]["types"]:
        difficulty = ANOMALY_TAXONOMY[category]["types"][anomaly_type]["detection_difficulty"]
        return DETECTION_DIFFICULTY_SCORES.get(difficulty, 1.0)
    return 1.0

def get_anomaly_business_impact(anomaly_type):
    """Retourne l'impact business pour un type d'anomalie"""
    category = get_anomaly_category(anomaly_type)
    if category in ANOMALY_TAXONOMY and anomaly_type in ANOMALY_TAXONOMY[category]["types"]:
        return ANOMALY_TAXONOMY[category]["types"][anomaly_type]["business_impact"]
    return "medium"
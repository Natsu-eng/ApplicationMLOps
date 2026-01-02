# 🚀 DataLab Pro - Plateforme MLOps Complète

**DataLab Pro** est une plateforme d'analyse de données et de Machine Learning automatisé construite avec Streamlit. Elle supporte à la fois les données tabulaires (ML classique) et les images (Computer Vision) avec des fonctionnalités avancées pour la classification supervisée et la détection d'anomalies non supervisée.

> **Version Production-Ready** - Architecture robuste avec support complet des modes supervisé et non supervisé, logging centralisé, et intégration MLflow.

---

## ✨ Fonctionnalités Principales

### 📊 Machine Learning Classique (Données Tabulaires)
- **Classification** : Binaire et multiclasse
- **Régression** : Prédiction de valeurs continues
- **Clustering** : Groupement non supervisé
- **Prétraitement avancé** : Imputation, normalisation, encodage, PCA
- **Gestion du déséquilibre** : SMOTE, class weights
- **Modèles** : XGBoost, LightGBM, CatBoost, Random Forest, SVM, etc.

### 🖼️ Computer Vision
- **Classification Supervisée** : Binaire et multiclasse
  - CNNs personnalisées (SimpleCNN, CustomResNet)
  - Transfer Learning (ResNet, VGG, EfficientNet)
- **Détection d'Anomalies Non Supervisée** :
  - Autoencoders (Convolutif, Variationnel, Denoising)
  - PatchCore
  - Réseaux Siamese
- **Prétraitement Images** :
  - Normalisation (MinMax, Standard, ImageNet)
  - Augmentation (Albumentations)
  - Redimensionnement adaptatif
- **Évaluation Complète** :
  - Métriques de classification (Accuracy, F1, AUC-ROC)
  - Métriques de reconstruction (MSE, erreur adaptative)
  - Visualisations (heatmaps, courbes ROC/PR)

### 🔄 Modes Supportés

#### 1. **Non Supervisé** (`y=None`)
- Détection d'anomalies par reconstruction
- Entraînement uniquement sur images normales
- Évaluation basée sur erreur de reconstruction

#### 2. **Supervisé Binaire** (`y ∈ {0,1}`)
- Classification binaire classique
- Détection d'anomalies supervisée (classes déséquilibrées)
- Métriques adaptées (AUC-ROC, F1, Precision, Recall)

#### 3. **Supervisé Multiclasse** (`y ∈ {0,1,...,N-1}`)
- Classification multiclasse
- Labels 0-indexed et consécutifs
- Métriques avec moyennes (macro, micro, weighted)

---

## 🏗️ Architecture

Le projet suit une architecture modulaire pour une séparation claire des responsabilités :

```
app-analyse/
├── src/
│   ├── app/                    # Interface utilisateur Streamlit
│   │   └── pages/              # Pages: Dashboard, Training, Evaluation
│   ├── config/                 # Configuration (Pydantic Settings)
│   ├── data/                   # Chargement et prétraitement
│   │   ├── data_loader.py      # Chargement données tabulaires
│   │   ├── computer_vision_preprocessing.py  # Preprocessing images
│   │   └── mvtec_ad/           # Support MVTec AD dataset
│   ├── models/                 # Modèles et entraînement
│   │   ├── training.py         # Trainer ML classique
│   │   ├── computer_vision_training.py  # Trainer Computer Vision
│   │   └── computer_vision/    # Modèles CV
│   │       ├── classification/ # CNNs, Transfer Learning
│   │       └── anomaly_detection/  # Autoencoders, PatchCore, Siamese
│   ├── evaluation/             # Métriques et visualisations
│   ├── explorations/           # Exploratory Data Analysis
│   └── shared/                 # Modules partagés (logging, utils)
├── orchestrators/             # Orchestrateurs métier
│   ├── ml_training_orchestrator.py      # Orchestrateur ML classique
│   └── visio_training_orchestrator.py   # Orchestrateur Computer Vision
├── helpers/                    # Helpers réutilisables
├── utils/                      # Utilitaires génériques
├── monitoring/                  # Monitoring et décorateurs
│   ├── state_managers.py       # Gestion état global
│   ├── mlflow_collector.py     # Collecteur MLflow
│   └── decorators.py          # Décorateurs production
├── ui/                         # Composants UI centralisés
├── requirements.txt            # Dépendances Python
└── .env                        # Variables d'environnement
```

### 🔄 Flux de Données Computer Vision

```
Page Streamlit (4_training_computer.py)
    ↓
    Détection Mode (supervisé/non supervisé)
    ↓
    Split Stratifié (ou aléatoire si non supervisé)
    ↓
    Configuration Preprocessing
    ↓
    Sélection Modèle
    ↓
    TrainingContext → Orchestrateur
    ↓
    ComputerVisionTrainer
    ↓
    Résultats → STATE.training_results
    ↓
Page Évaluation (5_anomaly_evaluation.py)
```

---

## 🚀 Démarrage Rapide

### 1. Prérequis

- **Python 3.11+**
- **Docker & Docker Compose** (optionnel, pour production)
- **PostgreSQL** (optionnel, pour MLflow tracking)
- **CUDA** (optionnel, pour GPU - PyTorch)

### 2. Installation Locale

1. **Clonez le projet :**
   ```bash
   git clone <repository_url>
   cd app-analyse
   ```

2. **Créez un environnement virtuel et installez les dépendances :**
   ```bash
   python -m venv env
   source env/bin/activate  # sur Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Pour PyTorch avec CUDA (optionnel) :**
   ```bash
   # Visitez https://pytorch.org/get-started/locally/
   # et installez la version appropriée pour votre système
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Configurez l'environnement :**
   - Créez un fichier `.env` à la racine du projet
   - Ajoutez vos configurations :
     ```env
     # Logging
     LOG_LEVEL=INFO
     
     # MLflow (optionnel)
     MLFLOW_TRACKING_URI=postgresql+psycopg2://user:password@host:port/dbname
     MLFLOW_EXPERIMENT_NAME=production_experiments
     
     # Performance
     MAX_MEMORY_USAGE=85
     MEMORY_CHECK_INTERVAL=180
     ```

5. **Lancez l'application :**
   ```bash
   streamlit run src/app/main.py
   ```

### 3. Démarrage avec Docker

1. **Assurez-vous que votre fichier `.env` est configuré**

2. **Lancez les services :**
   ```bash
   docker-compose up --build
   ```
   - Application : `http://localhost:8501`
   - MLflow UI : `http://localhost:5000`

---

## 📖 Guide d'Utilisation

### Machine Learning Classique

1. **Dashboard** : Chargez votre dataset (CSV, Parquet, Excel)
2. **Exploration** : Analysez les données (qualité, corrélations, distributions)
3. **Training** : Configurez votre expérimentation (cible, features, modèles)
4. **Évaluation** : Comparez les modèles et analysez les métriques

### Computer Vision

#### Mode Supervisé

1. **Chargement** : Chargez vos images depuis un dossier structuré par classes
   ```
   dataset/
   ├── class_0/
   │   ├── img1.jpg
   │   └── img2.jpg
   ├── class_1/
   │   └── ...
   ```

2. **Split** : Le système détecte automatiquement le mode et effectue un split stratifié

3. **Configuration** :
   - Prétraitement : Normalisation, redimensionnement
   - Augmentation : Flip, rotation, zoom, etc.
   - Modèle : CNN, Transfer Learning

4. **Entraînement** : Lancez l'entraînement avec monitoring en temps réel

5. **Évaluation** : Visualisez les métriques, confusion matrix, courbes ROC/PR

#### Mode Non Supervisé (Détection d'Anomalies)

1. **Chargement** : Chargez vos images depuis un dossier plat (pas de sous-dossiers par classe)
   ```
   dataset/
   ├── normal_img1.jpg
   ├── normal_img2.jpg
   └── ...
   ```

2. **Split** : Split aléatoire simple (pas de stratification)

3. **Configuration** :
   - Modèle : Autoencoder, VAE, PatchCore, ou Siamese
   - Prétraitement : Normalisation adaptée

4. **Entraînement** : Le modèle apprend à reconstruire les images normales

5. **Évaluation** :
   - Erreur de reconstruction par image
   - Seuil adaptatif pour détection
   - Visualisation des erreurs (heatmaps)

---

## 🔧 Bonnes Pratiques de Développement

### Logging Centralisé

Le système de logging est **centralisé** dans `src/shared/logging.py`. Tous les modules doivent utiliser `get_logger()` :

```python
from src.shared.logging import get_logger

logger = get_logger(__name__)
logger.info("Message d'information")
logger.error("Erreur", exc_info=True)
```

**⚠️ Ne pas** configurer `logging.basicConfig()` ou créer des handlers manuellement.

### Décorateurs Standardisés

Tous les décorateurs sont dans `monitoring/decorators.py` :

```python
from monitoring.decorators import safe_execute, monitor_performance

@safe_execute(fallback_value=None, max_retries=2)
def ma_fonction():
    # Code avec gestion d'erreurs automatique
    pass

@monitor_performance
def operation_longue():
    # Monitoring automatique des performances
    pass
```

### Structure des Imports

```python
# 1. Imports standards
import os
import numpy as np

# 2. Imports de logging
from src.shared.logging import get_logger

# 3. Imports de configuration
from src.config.constants import ...

# 4. Imports de décorateurs
from monitoring.decorators import safe_execute

# 5. Imports internes
from src.models.computer_vision_training import ...

# 6. Imports helpers/utils
from helpers.data_validators import ...
```

### Gestion des Modes (Computer Vision)

**Toujours vérifier le mode avant d'accéder aux labels :**

```python
# ✅ CORRECT
if y_train is not None:
    classes = np.unique(y_train)
    # Calculs supervisés
else:
    # Mode non supervisé
    pass

# ❌ INCORRECT
classes = np.unique(y_train)  # Crash si y_train is None
```

---

## 📚 Documentation Technique

### Système de Logging

- Fichiers de logs avec rotation automatique
- Support MLflow intégré
- Niveaux configurables via `LOG_LEVEL`
- Format standardisé pour tous les logs

### State Management

Le state management est **thread-safe** et utilise un pattern singleton :

```python
from monitoring.state_managers import STATE

# Accès aux données
if STATE.loaded:
    X = STATE.data.X
    y = STATE.data.y
    # ...
```

### Configuration

La configuration utilise **Pydantic Settings** :

```python
from src.config.settings import app_settings, training_settings

max_size = app_settings.MAX_FILE_SIZE_MB
threshold = training_settings.HIGH_MEMORY_THRESHOLD
```

### Orchestrateurs

Les orchestrateurs coordonnent le workflow complet :

- **`ml_training_orchestrator`** : ML classique
- **`visio_training_orchestrator`** : Computer Vision

Ils gèrent :
- Validation des données
- Preprocessing
- Entraînement
- Logging MLflow
- Gestion d'erreurs

---

## 🔍 Audits et Qualité

### Audits Effectués

Le projet a été audité de manière complète :

1. **Audit Pipeline Computer Vision** (`AUDIT_PIPELINE_COMPUTER_VISION.md`)
   - Support complet des 3 modes (non supervisé, binaire, multiclasse)
   - Vérification end-to-end du pipeline
   - Corrections des problèmes bloquants (P0)

2. **Audit Organisation Modèles** (`AUDIT_ORGANISATION_MODELS_COMPUTER_VISION.md`)
   - Structure des modèles optimisée
   - Séparation claire classification/anomaly detection
   - Exports publics documentés

3. **Audit Pages Streamlit** (`AUDIT_PAGES_STREAMLIT_COMPUTER_VISION.md`)
   - Cohérence UI avec orchestrateur
   - Support mode non supervisé dans l'interface
   - Validation conditionnelle selon mode

### Qualité du Code

- ✅ Support complet des modes supervisé/non supervisé
- ✅ Gestion robuste de `y=None`
- ✅ Validation conditionnelle selon mode
- ✅ Logging détaillé
- ✅ Gestion d'erreurs complète
- ✅ Architecture modulaire et maintenable

---

## 🚀 Déploiement Production

### Prérequis

- Python 3.11+
- PostgreSQL (pour MLflow tracking)
- Variables d'environnement configurées

### Checklist Production

- [ ] Variables d'environnement configurées (`.env`)
- [ ] MLflow tracking URI configuré (PostgreSQL recommandé)
- [ ] Logs configurés avec rotation
- [ ] Niveau de logging adapté (INFO ou WARNING)
- [ ] Monitoring système activé
- [ ] Health checks configurés
- [ ] GPU configuré si nécessaire (CUDA)

### Variables d'Environnement Critiques

```env
# Logging
LOG_LEVEL=INFO

# MLflow
MLFLOW_TRACKING_URI=postgresql+psycopg2://user:password@host:port/dbname
MLFLOW_EXPERIMENT_NAME=production_experiments

# Performance
MAX_MEMORY_USAGE=85
MEMORY_CHECK_INTERVAL=180
```

---

## 🔍 Troubleshooting

### Problèmes de Logging

1. Vérifier que `setup_logging()` est appelé dans `main.py`
2. Vérifier le niveau de log dans `.env` (`LOG_LEVEL`)
3. Vérifier les permissions d'écriture dans `logs/`

### Problèmes MLflow

1. Vérifier la connexion à PostgreSQL
2. Vérifier les variables `MLFLOW_TRACKING_URI` et `MLFLOW_EXPERIMENT_NAME`
3. L'application continue sans MLflow si non disponible (graceful degradation)

### Problèmes Computer Vision

1. **Mode non détecté** : Vérifier que `y` est bien `None` ou un array valide
2. **Erreur split** : Vérifier que le mode est correctement propagé
3. **Erreur training** : Vérifier que `y_train=None` est géré pour autoencoders
4. **Erreur évaluation** : Vérifier que `y_test=None` est géré pour mode non supervisé

### Problèmes GPU

1. Vérifier l'installation PyTorch avec CUDA :
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
2. Vérifier les drivers NVIDIA
3. L'application fonctionne en CPU si GPU non disponible

---

## 📝 Contribution

Lors de l'ajout de code :

1. **Utiliser `get_logger()`** de `src/shared/logging.py`
2. **Utiliser les décorateurs** de `monitoring/decorators.py`
3. **Suivre la structure d'imports** documentée
4. **Documenter** les fonctions publiques avec docstrings
5. **Tester** les modifications avec les tests existants
6. **Gérer `y=None`** correctement pour le mode non supervisé
7. **Vérifier le mode** avant d'accéder aux labels

---

## 📄 Structure des Modèles Computer Vision

```
src/models/computer_vision/
├── __init__.py                 # Exports publics
├── model_builder.py            # Factory pour construire modèles
├── persistence.py              # Sauvegarde/chargement modèles
├── cross_validator.py          # Cross-validation
├── hyperparameter_visio.py      # Hyperparameter tuning
├── classification/
│   ├── __init__.py
│   ├── cnn_models.py           # SimpleCNN, CustomResNet
│   └── transfer_learning.py   # TransferLearningModel
└── anomaly_detection/
    ├── __init__.py
    ├── autoencoders.py         # ConvAutoEncoder, VAE, DenoisingAE
    ├── patch_core.py           # ProfessionalPatchCore
    ├── siamese_networks.py     # ProfessionalSiameseNetwork
    └── anomaly_type_classifier.py  # Classification types d'anomalies
```

---

## 🎯 Fonctionnalités Avancées

### Détection Automatique du Mode

Le système détecte automatiquement le mode d'entraînement :

- **Non supervisé** : `y=None` ou array vide
- **Supervisé binaire** : `len(unique(y)) == 2`
- **Supervisé multiclasse** : `len(unique(y)) > 2`

### Split Adaptatif

- **Non supervisé** : Split aléatoire simple
- **Supervisé** : Split stratifié (préservation des ratios de classes)

### Métriques Adaptatives

- **Non supervisé** : Erreur de reconstruction, seuil adaptatif
- **Supervisé** : Accuracy, F1, Precision, Recall, AUC-ROC

### Preprocessing Intelligent

- Détection automatique du format (channels_first/last)
- Normalisation adaptée selon le modèle
- Augmentation uniquement sur train (jamais sur val/test)

---

## 📊 Métriques et Évaluation

### Classification Supervisée

- **Accuracy** : Précision globale
- **F1-Score** : Moyenne harmonique précision/rappel
- **Precision** : Vrais positifs / (Vrais + Faux positifs)
- **Recall** : Vrais positifs / (Vrais positifs + Faux négatifs)
- **AUC-ROC** : Aire sous la courbe ROC
- **Confusion Matrix** : Matrice de confusion détaillée

### Détection d'Anomalies

- **Reconstruction Error** : Erreur moyenne de reconstruction
- **Adaptive Threshold** : Seuil adaptatif pour détection
- **Localization Maps** : Cartes de localisation des anomalies
- **Error Distribution** : Distribution des erreurs

---

## 📚 Documentation Complémentaire

- **`AUDIT_PIPELINE_COMPUTER_VISION.md`** : Audit complet du pipeline
- **`AUDIT_ORGANISATION_MODELS_COMPUTER_VISION.md`** : Organisation des modèles
- **`AUDIT_PAGES_STREAMLIT_COMPUTER_VISION.md`** : Audit des pages UI
- **`DOCUMENTATION_ORCHESTRATEUR_TRAINING.md`** : Documentation orchestrateur
- **`ANALYSE_COMPLETE_COMPUTER_VISION.md`** : Analyse complète Computer Vision

---

## 📄 Licence

[Votre licence ici]

---

## 🙏 Remerciements

- Streamlit pour l'interface utilisateur
- PyTorch pour le deep learning
- MLflow pour le tracking d'expériences
- Albumentations pour l'augmentation d'images

---

**Version** : 2.0.0  
**Dernière mise à jour** : 2024  
**Statut** : Production-Ready ✅

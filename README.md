# DataLab Pro 🧪

**DataLab Pro** est une plateforme d'analyse de données et de Machine Learning automatisé construite avec Streamlit. Elle permet de charger, d'explorer, de prétraiter des données, ainsi que d'entraîner et d'évaluer des modèles de classification, de régression et de clustering.

> **Version Production-Ready** - Consolidation complète du système de logging et des décorateurs pour une architecture robuste et maintenable.

## Architecture

Le projet suit une architecture modulaire pour une séparation claire des responsabilités :

```
app-analyse/
├── src/
│   ├── app/          # Interface utilisateur Streamlit
│   ├── config/       # Configuration de l'application (Pydantic Settings)
│   ├── data/         # Chargement et prétraitement des données
│   ├── models/       # Logique d'entraînement et catalogue de modèles
│   ├── evaluation/   # Calcul des métriques et visualisations
│   ├── monitoring/   # Détection de dérive et surveillance
│   └── shared/       # Modules partagés (état, logging centralisé)
├── orchestrators/    # Orchestrateurs métier (ML et Computer Vision)
├── helpers/          # Helpers réutilisables
├── utils/            # Utilitaires génériques
├── monitoring/       # Monitoring et décorateurs (point d'entrée unique)
├── .env              # Fichier pour les variables d'environnement
├── requirements.txt  # Dépendances Python
├── Dockerfile        # Fichier de build Docker
└── docker-compose.yml # Orchestration des services
```

### 🔧 Bonnes Pratiques de Développement

#### Logging Centralisé

Le système de logging est **centralisé** dans `src/shared/logging.py`. Tous les modules doivent utiliser `get_logger()` :

```python
from src.shared.logging import get_logger

logger = get_logger(__name__)
logger.info("Message d'information")
```

**⚠️ Ne pas** configurer `logging.basicConfig()` ou créer des handlers manuellement. Le système de logging est initialisé automatiquement via `setup_logging()` appelé dans `main.py`.

#### Décorateurs Standardisés

Tous les décorateurs de gestion d'erreurs et de monitoring sont **centralisés** dans `monitoring/decorators.py` :

```python
from monitoring.decorators import safe_execute, monitor_performance, handle_mlflow_errors

@safe_execute(fallback_value=None, max_retries=2)
def ma_fonction():
    # Code avec gestion d'erreurs automatique
    pass

@monitor_performance
def operation_longue():
    # Monitoring automatique des performances
    pass

@handle_mlflow_errors
def log_mlflow():
    # Gestion gracieuse des erreurs MLflow
    pass
```

**⚠️ Ne pas** créer de nouveaux décorateurs de gestion d'erreurs. Utiliser ceux de `monitoring/decorators.py`.

#### Structure des Imports

Pour maintenir la cohérence, suivre cette structure d'imports :

```python
# 1. Imports standards
import os
import pandas as pd

# 2. Imports de logging (TOUJOURS utiliser get_logger)
from src.shared.logging import get_logger

# 3. Imports de configuration
from src.config.constants import ...
from src.config.settings import ...

# 4. Imports de décorateurs (monitoring/decorators.py)
from monitoring.decorators import safe_execute, monitor_performance

# 5. Imports internes (src/)
from src.models.training import ...
from src.data.data_loader import ...

# 6. Imports helpers/utils
from helpers.data_validators import ...
from utils.system_utils import ...
```

## 🚀 Démarrage Rapide

### 1. Prérequis

- Python 3.11+
- Docker & Docker Compose
- Un client PostgreSQL (optionnel, pour MLflow)

### 2. Installation Locale

1.  **Clonez le projet :**
    ```bash
    git clone <repository_url>
    cd app-analyse
    ```

2.  **Créez un environnement virtuel et installez les dépendances :**
    ```bash
    python -m venv env
    source env/bin/activate  # sur Windows: env\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configurez l'environnement :**
    - Créez un fichier `.env` à la racine du projet.
    - Ajoutez vos configurations, notamment pour MLflow si vous l'utilisez :
      ```env
      MLFLOW_TRACKING_URI=postgresql+psycopg2://user:password@host:port/dbname
      ```

4.  **Lancez l'application :**
    ```bash
    streamlit run src/app/main.py
    ```

### 3. Démarrage avec Docker

Cette méthode est recommandée pour un environnement de production reproductible.

1.  **Assurez-vous que votre fichier `.env` est configuré.** Le `docker-compose.yml` l'utilisera.

2.  **Lancez les services :**
    - Pour lancer l'application, la base de données et MLflow :
      ```bash
      docker-compose up --build
      ```
    - L'application sera disponible sur `http://localhost:8501`.
    - L'interface MLflow sera sur `http://localhost:5000`.

## Utilisation de l'Application

1.  **Accueil** : Chargez votre jeu de données (CSV, Parquet, Excel).
2.  **Dashboard** : Explorez les données via les onglets (qualité, analyse univariée, corrélations, etc.).
3.  **Entraînement** : Configurez votre expérimentation ML (cible, features, modèles) et lancez l'entraînement.
4.  **Évaluation** : Comparez les modèles, analysez les métriques et visualisez les résultats détaillés.

## 📚 Documentation Technique

### Système de Logging

Le système de logging est **centralisé et idempotent**. Il est configuré automatiquement au démarrage de l'application via `setup_logging()` dans `main.py`.

**Configuration** :
- Fichiers de logs avec rotation automatique
- Support MLflow intégré
- Niveaux configurables via variables d'environnement (`LOG_LEVEL`)
- Format standardisé pour tous les logs

**Utilisation** :
```python
from src.shared.logging import get_logger

logger = get_logger(__name__)
logger.info("Message")
logger.error("Erreur", exc_info=True)
```

### Décorateurs de Production

Tous les décorateurs sont dans `monitoring/decorators.py` :

- **`@safe_execute`** : Exécution sécurisée avec fallback et retry
- **`@monitor_performance`** : Monitoring automatique des performances
- **`@monitor_operation`** : Monitoring avec logs structurés
- **`@handle_mlflow_errors`** : Gestion gracieuse des erreurs MLflow
- **`@safe_metric_calculation`** : Calculs de métriques avec retry
- **`@timeout`** : Timeout automatique sur opérations longues

### Configuration

La configuration utilise **Pydantic Settings** pour validation et chargement depuis `.env` :

```python
from src.config.settings import app_settings, training_settings, mlflow_settings

# Utilisation
max_size = app_settings.MAX_FILE_SIZE_MB
threshold = training_settings.HIGH_MEMORY_THRESHOLD
mlflow_uri = mlflow_settings.MLFLOW_TRACKING_URI
```

### State Management

Le state management est **thread-safe** et utilise un pattern singleton :

```python
from monitoring.state_managers import STATE

# Accès aux données
if STATE.loaded:
    df = STATE.data.df
    # ...
```

## 🚀 Déploiement Production

### Prérequis

- Python 3.11+
- PostgreSQL (pour MLflow tracking)
- Variables d'environnement configurées (voir `.env.example`)

### Checklist Production

- [ ] Variables d'environnement configurées (`.env`)
- [ ] MLflow tracking URI configuré (PostgreSQL recommandé)
- [ ] Logs configurés avec rotation
- [ ] Niveau de logging adapté (INFO ou WARNING en production)
- [ ] Monitoring système activé
- [ ] Health checks configurés

### Variables d'Environnement Critiques

```env
# Logging
LOG_LEVEL=INFO  # ou WARNING en production

# MLflow
MLFLOW_TRACKING_URI=postgresql+psycopg2://user:password@host:port/dbname
MLFLOW_EXPERIMENT_NAME=production_experiments

# Performance
MAX_MEMORY_USAGE=85
MEMORY_CHECK_INTERVAL=180
```

## 🔍 Troubleshooting

### Problèmes de Logging

Si les logs ne s'affichent pas, vérifier :
1. Que `setup_logging()` est appelé dans `main.py`
2. Le niveau de log dans `.env` (`LOG_LEVEL`)
3. Les permissions d'écriture dans le dossier `logs/`

### Problèmes MLflow

Si MLflow ne fonctionne pas :
1. Vérifier la connexion à PostgreSQL
2. Vérifier les variables `MLFLOW_TRACKING_URI` et `MLFLOW_EXPERIMENT_NAME`
3. L'application continue sans MLflow si non disponible (graceful degradation)

## 📝 Contribution

Lors de l'ajout de code :

1. **Utiliser `get_logger()`** de `src/shared/logging.py`
2. **Utiliser les décorateurs** de `monitoring/decorators.py`
3. **Suivre la structure d'imports** documentée ci-dessus
4. **Documenter** les fonctions publiques avec docstrings
5. **Tester** les modifications avec les tests existants

## 📄 Licence

[Votre licence ici]

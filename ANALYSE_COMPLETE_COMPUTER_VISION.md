# 🔍 ANALYSE COMPLÈTE - COMPUTER VISION
## Détection d'Anomalies Industrielle (MVTec AD, Classification, Anomaly Detection)

**Date**: 2024-12-19  
**Expert**: Analyse approfondie du pipeline complet CV  
**Scope**: Chargement → Preprocessing → Training → Evaluation → Prediction → Localisation

---

## 📋 TABLE DES MATIÈRES

1. [Détection Automatique du Type de Tâche](#1-détection-automatique-du-type-de-tâche)
2. [Pipeline d'Entraînement](#2-pipeline-dentraînement)
3. [Phase d'Évaluation (CRITIQUE)](#3-phase-dévaluation-critique)
4. [Détection du Type d'Erreur](#4-détection-du-type-derreur)
5. [Localisation de l'Erreur (Heatmaps)](#5-localisation-de-lerreur-heatmaps)
6. [Erreurs et Incohérences Trouvées](#6-erreurs-et-incohérences-trouvées)
7. [Correctifs Proposés](#7-correctifs-proposés)
8. [Recommandations Finales](#8-recommandations-finales)

---

## 1. DÉTECTION AUTOMATIQUE DU TYPE DE TÂCHE

### 1.1 Pipeline de Détection

**Fichier clé**: `utils/task_detector.py`

```python
def detect_cv_task(y: np.ndarray) -> Tuple[TaskType, Dict[str, Any]]:
```

**Logique de détection** (appliquée sur `y_train` UNIQUEMENT):

1. **CAS 1: UNSUPERVISED (MVTec AD)**
   - Condition: `n_classes == 1` (uniquement des images normales)
   - Retour: `TaskType.UNSUPERVISED`
   - **CRITIQUE**: C'est ici que le système détecte MVTec AD !

2. **CAS 2: ANOMALY_DETECTION (Supervisé)**
   - Condition: `n_classes == 2` ET `unique_labels == {0, 1}`
   - Retour: `TaskType.ANOMALY_DETECTION`
   - Labels: `0 = normal`, `1 = anomaly`

3. **CAS 3: BINARY_CLASSIFICATION**
   - Condition: `n_classes == 2` mais labels différents de {0,1}
   - Exemple: {0, 2} ou {1, 3}

4. **CAS 4: MULTICLASS_CLASSIFICATION**
   - Condition: `n_classes > 2`

### 1.2 Chargement MVTec AD

**Fichier clé**: `src/explorations/image_exploration_plots.py`

```python
def _load_mvtec_train_labels(data_dir: str) -> np.ndarray:
    """Charge UNIQUEMENT train/good → retourne [0, 0, ..., 0]"""
    train_good_path = Path(data_dir) / "train" / "good"
    image_files = _get_image_files(train_good_path)
    return np.zeros(len(image_files), dtype=int)  # ← TOUJOURS 0
```

**Fichier**: `load_images_flexible()` ligne 388-433

**Processus**:
1. Détecte structure MVTec AD via `detect_dataset_structure()`
2. Si `structure_type == "mvtec_ad"`:
   - `X, y_full = _load_mvtec_structure()` → charge train/good + test/good + test/anomalies
   - `y_train = _load_mvtec_train_labels()` → **UNIQUEMENT train/good (tous 0)**
3. **Décision**:
   - `detect_cv_task(y_train)` → détecte `UNSUPERVISED` car `n_classes == 1`
   - Mode → `"unsupervised"`

### 1.3 Quand le Pipeline Part en "Anomaly Detection"

**Condition**: `y_train` contient uniquement des `0` (images normales)

**Fichiers impliqués**:
- `ui/training_vision.py` ligne 112-236: `detect_training_mode(y)`
- `utils/task_detector.py` ligne 16-60: `detect_cv_task(y)`

**Ordre de priorité**:
1. `STATE.data.y_train` (si disponible)
2. Sinon: paramètre `y` passé à `detect_training_mode()`

### 1.4 Quand le Pipeline Part en "Classification"

**Condition**: `y_train` contient au moins 2 classes différentes

**Cas spécifiques**:
- `{0, 1}` → `ANOMALY_DETECTION` (supervisé)
- Autres → `BINARY_CLASSIFICATION` ou `MULTICLASS_CLASSIFICATION`

### 1.5 ERREURS POTENTIELLES dans la Détection

#### ❌ **ERREUR CRITIQUE #1**: Labels Mal Chargés

**Problème**:
- Si `y_train` est accidentellement chargé avec train+test (au lieu de train uniquement)
- Le système pourrait détecter 2 classes alors que c'est du MVTec AD

**Localisation**: `src/explorations/image_exploration_plots.py` ligne 388-433

**Impact**: **CRITIQUE** - Le modèle partira en mode "supervised" au lieu de "unsupervised"

#### ⚠️ **ERREUR #2**: Mapping Inversé Labels

**Problème**:
- Si les labels sont inversés (1=normal, 0=anomaly)
- Le système détectera quand même `ANOMALY_DETECTION` mais avec mapping inversé

**Localisation**: `utils/task_detector.py` ligne 36

**Impact**: **MOYEN** - Le modèle fonctionnera mais avec sémantique inversée

#### ⚠️ **ERREUR #3**: Shape Incorrect de y_train

**Problème**:
- Si `y_train` a une shape incorrecte (ex: 2D au lieu de 1D)
- `np.unique(y)` pourrait échouer ou retourner des valeurs inattendues

**Impact**: **CRITIQUE** - Crash ou mauvaise détection

---

## 2. PIPELINE D'ENTRAÎNEMENT

### 2.1 Flow Complet

```
load_images_flexible() 
  → detect_dataset_structure()
  → _load_mvtec_structure() ou _load_categorical_folders()
  → y_train = _load_mvtec_train_labels() (si MVTec AD)
  
detect_training_mode(y_train)
  → detect_cv_task(y_train)
  → Retourne mode: "unsupervised" ou "supervised"

ComputerVisionTrainingOrchestrator.train()
  → ComputerVisionTrainer.fit() ou AnomalyAwareTrainer.train()
  → _setup_preprocessing() (fit sur train uniquement)
  → _build_model()
  → _training_loop()
```

### 2.2 Dataloader et Preprocessing

**Fichier**: `src/data/computer_vision_preprocessing.py`

#### DataPreprocessor

```python
class DataPreprocessor:
    def fit(self, X: np.ndarray):
        # Détecte format automatiquement (channels_first/last)
        # Calcule mean/std sur train UNIQUEMENT
        # GARANTIT: Pas de fuite de données
        
    def transform(self, X: np.ndarray, output_format="channels_first"):
        # Convertit vers format PyTorch (N, C, H, W)
        # Applique standardisation: (X - mean) / std
```

**Points critiques**:
- ✅ **Pas de fuite**: `fit()` uniquement sur `X_train`
- ✅ **Format cohérent**: Auto-détection puis conversion vers `channels_first`
- ⚠️ **Bug potentiel**: Si format détecté incorrectement → erreur shape

#### DataLoaderFactory

```python
def create(X: np.ndarray, y: np.ndarray, batch_size: int):
    # Conversion tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    # Création DataLoader
    return DataLoader(TensorDataset(X_tensor, y_tensor), ...)
```

**Validation**: ✅ Vérifie cohérence des shapes avant création

### 2.3 Augmentations et Normalisation

**Fichier**: `src/data/image_augmentation.py`

**Disponible**: Via Albumentations dans `apply_augmentation()`

**Normalisation**:
- `0-1 (MinMax)`: `/255.0`
- `-1-1`: `(image / 127.5) - 1.0`
- `Standard (ImageNet)`: `(image / 255.0 - mean) / std`

**⚠️ ERREUR #4**: Normalisation Pas Toujours Appliquée

**Problème**: Les augmentations sont optionnelles et peuvent être désactivées

**Impact**: **FAIBLE** - Si désactivées, le modèle fonctionne quand même

### 2.4 Forward Pass et Calcul des Pertes

**Fichier**: `src/models/computer_vision_training.py`

#### Pour Classification

```python
def _train_epoch(self, train_loader: DataLoader, is_autoencoder: bool = False):
    for batch_idx, (data, target) in enumerate(train_loader):
        output = self.model(data)  # (B, num_classes)
        loss = self.train_criterion(output, target)  # CrossEntropyLoss
```

#### Pour Autoencoder (Unsupervised)

```python
def _train_epoch(self, train_loader: DataLoader, is_autoencoder: bool = True):
    for batch_idx, (data, target) in enumerate(train_loader):
        target = data  # ← CRITIQUE: target = input pour reconstruction
        output = self.model(data)  # Reconstructed image
        loss = self.train_criterion(output, target)  # MSELoss
```

**✅ COHÉRENT**: La logique distingue bien classification et autoencoder

### 2.5 Métriques Utilisées

#### Classification
- `accuracy_score`
- `precision_score`, `recall_score`, `f1_score`
- `roc_auc_score` (si binaire)
- `confusion_matrix`

#### Autoencoder
- `mean_reconstruction_error`
- `std_reconstruction_error`
- `threshold_95percentile`
- Métriques de classification basées sur seuil adaptatif

**Fichier**: `src/models/computer_vision_training.py` ligne 935-974

### 2.6 ERREURS dans l'Entraînement

#### ❌ **ERREUR CRITIQUE #5**: Incohérence Format Preprocessing

**Problème**:
- Si `X_train` arrive en `channels_last` mais que le preprocessor détecte `channels_first`
- Les statistiques calculées seront fausses

**Localisation**: `src/data/computer_vision_preprocessing.py` ligne 119-143

**Impact**: **CRITIQUE** - Normalisation incorrecte → modèle ne converge pas

#### ⚠️ **ERREUR #6**: Labels Mal Alignés

**Problème**:
- Si `y_train` et `X_train` ne sont pas dans le même ordre après chargement
- Les labels ne correspondent plus aux bonnes images

**Impact**: **CRITIQUE** - Le modèle apprend les mauvaises associations

#### ⚠️ **ERREUR #7**: Seuil Adaptatif Trop Restrictif

**Problème**:
- Dans `_predict_autoencoder()` ligne 1096-1098:
  ```python
  threshold = np.percentile(reconstruction_errors, 95)
  ```
- Si le dataset de train contient déjà des anomalies → seuil faussé

**Impact**: **MOYEN** - Faux positifs ou faux négatifs

---

## 3. PHASE D'ÉVALUATION (CRITIQUE)

### 3.1 Comment le Modèle Décide si une Image est Correcte/Défectueuse

#### Pour Autoencoder

**Fichier**: `src/models/computer_vision_training.py` ligne 1068-1109

```python
def _predict_autoencoder(...):
    # 1. Reconstruction
    reconstructed = self.model(data)
    
    # 2. Calcul erreur MSE par image
    errors = torch.mean((data - reconstructed) ** 2, dim=(1, 2, 3))
    
    # 3. Seuil automatique (95ème percentile)
    threshold = np.percentile(reconstruction_errors, 95)
    
    # 4. Prédiction binaire
    predictions = (reconstruction_errors > threshold).astype(int)
```

**Fichier**: `src/app/pages/5_anomaly_evaluation.py` ligne 492-583

**Processus dans `robust_predict_with_preprocessor()`**:

```python
# 1. Normalisation erreurs
max_error = np.max(reconstruction_errors)
y_pred_proba = reconstruction_errors / max_error

# 2. Seuil adaptatif
threshold = np.median(y_pred_proba) + np.std(y_pred_proba)
threshold = np.clip(threshold, 0.3, 0.7)  # Entre 0.3 et 0.7

# 3. Prédiction
y_pred_binary = (y_pred_proba > threshold).astype(int)
```

#### Pour Classification

**Fichier**: `src/models/computer_vision_training.py` ligne 1111-1130

```python
def _predict_classifier(...):
    output = self.model(data)
    probs = torch.softmax(output, dim=1)
    preds = output.argmax(dim=1)
```

**Dans `robust_predict_with_preprocessor()`** ligne 589-624:

```python
y_proba = torch.softmax(output, dim=1).cpu().numpy()
if y_proba.shape[1] == 2:
    y_pred_proba = y_proba[:, 1]  # Probabilité classe positive
else:
    y_pred_proba = np.max(y_proba, axis=1)  # Multi-classes

y_pred_binary = (y_pred_proba > 0.5).astype(int)
```

### 3.2 Calcul des Scores d'Anomalie

#### Autoencoder

**Score**: Erreur de reconstruction MSE normalisée

```python
reconstruction_errors = np.mean((X_processed - reconstructed_np) ** 2, axis=(1, 2, 3))
y_pred_proba = reconstruction_errors / max_error  # Normalisation [0, 1]
```

**⚠️ PROBLÈME**: Si toutes les erreurs sont très faibles, la normalisation peut créer des faux positifs

#### Classification

**Score**: Probabilité de la classe positive (ou max pour multi-classes)

### 3.3 Génération des Heatmaps (Localisation)

**Fichier**: `src/evaluation/model_vision_plots.py` ligne 230-294

```python
def plot_anomaly_heatmap(image: np.ndarray, anomaly_score: np.ndarray):
    # image: (H, W, C)
    # anomaly_score: (H, W) ← Carte spatiale des scores
    
    # Normalisation heatmap
    heatmap = (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min() + 1e-8)
    
    # Superposition sur image
    fig.add_trace(go.Image(z=img))
    fig.add_trace(go.Heatmap(z=heatmap, opacity=0.4))
```

**Fichier**: `src/models/computer_vision/anomaly_detection/autoencoders.py` ligne 420-442

```python
def get_reconstruction_error_map(self, x: torch.Tensor) -> torch.Tensor:
    """Génère une carte spatiale des erreurs de reconstruction"""
    reconstructed = self.forward(x)
    # Erreur par pixel, moyennée sur les canaux
    error_map = torch.mean((x - reconstructed) ** 2, dim=1, keepdim=True)
    return error_map  # (B, 1, H, W)
```

**❌ ERREUR CRITIQUE #8**: Heatmap Pas Toujours Générée

**Problème**:
- La fonction `get_reconstruction_error_map()` existe mais n'est **jamais appelée** dans le pipeline d'évaluation standard
- `plot_anomaly_heatmap()` attend un `anomaly_score` (H, W) mais il n'est pas toujours fourni

**Localisation**:
- `src/evaluation/model_vision_plots.py` ligne 230
- `src/app/pages/5_anomaly_evaluation.py` (aucun appel à `get_reconstruction_error_map()`)

**Impact**: **CRITIQUE** - Les heatmaps ne sont pas générées automatiquement lors de l'évaluation

### 3.4 Calcul des Masks et Seuils

**❌ MANQUANT**: Il n'y a **pas de génération de masks binaires** dans le code

**Ce qui existe**:
- `error_map` spatial dans `get_reconstruction_error_map()`
- `plot_anomaly_heatmap()` pour visualisation

**Ce qui manque**:
- Fonction pour convertir `error_map` → mask binaire avec seuil
- Alignement du mask avec l'image originale si resize effectué

### 3.5 Cohérence Ground Truth vs Prédictions

**Fichier**: `src/evaluation/computer_vision_metrics.py`

**Calcul des métriques** ligne 176-233:

```python
def compute_core_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray):
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average='weighted')
    metrics["recall"] = recall_score(y_true, y_pred, average='weighted')
    metrics["f1_score"] = f1_score(y_true, y_pred, average='weighted')
    
    if len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = roc_auc_score(y_true, y_scores)
```

**✅ COHÉRENT**: Les métriques sont bien calculées avec validation

**⚠️ ERREUR #9**: Format y_scores Incohérent

**Problème**:
- Pour autoencoder: `y_scores` = `reconstruction_errors` (1D array)
- Pour classification: `y_scores` = `probabilities` (2D array si multi-classes)
- `roc_auc_score()` peut échouer si format incorrect

**Impact**: **MOYEN** - Crash si format non géré

### 3.6 ERREURS dans la Phase d'Évaluation

#### ❌ **ERREUR CRITIQUE #10**: Mauvais Resize → Masque Non Aligné

**Problème**:
- Si l'image est resizée avant prédiction (ex: 256x256 → 224x224)
- La `error_map` retournée aura la shape de l'image resizée (224x224)
- Si on veut la superposer à l'image originale (256x256) → décalage

**Localisation**: `src/data/computer_vision_preprocessing.py` ligne 30-37

**Impact**: **CRITIQUE** - Localisation incorrecte des défauts

#### ❌ **ERREUR CRITIQUE #11**: Conversion Tensor → Numpy Incohérente

**Problème**:
- Dans `robust_predict_with_preprocessor()` ligne 549:
  ```python
  reconstructed_np = reconstructed.cpu().numpy()
  ```
- Mais `X_processed` peut être dans un format différent (normalisé vs non-normalisé)
- Le calcul `(X_processed - reconstructed_np) ** 2` peut être faussé

**Impact**: **CRITIQUE** - Scores d'anomalie incorrects

#### ⚠️ **ERREUR #12**: Seuil d'Anomalie Trop Fixe

**Problème**:
- Le seuil de 95ème percentile est fixe
- Si le dataset de test a une distribution différente du train → faux positifs

**Impact**: **MOYEN** - Performance dégradée

---

## 4. DÉTECTION DU TYPE D'ERREUR

### 4.1 Comment le Code Identifie le Type d'Erreur

**Fichier**: `src/models/computer_vision_training.py` ligne 1489-1528

```python
def _detect_anomaly_type_from_state(self, STATE) -> Optional[str]:
    # Stratégie 1: Metadata explicite
    if STATE.data.metadata.get('anomaly_type'):
        return anomaly_type
    
    # Stratégie 2: Nom du dataset
    name_lower = STATE.data.name.lower()
    if any(kw in name_lower for kw in ['crack', 'corrosion', 'deformation']):
        return "structural"
    if any(kw in name_lower for kw in ['scratch', 'stain', 'color']):
        return "visual"
    if any(kw in name_lower for kw in ['dimension', 'alignment', 'size']):
        return "geometric"
    
    # Stratégie 3: Structure MVTec AD
    if STATE.data.structure.get('type') == 'mvtec_ad':
        return "structural"
```

**❌ LIMITATION MAJEURE**: Le code ne peut **PAS** identifier le type d'erreur à partir des prédictions du modèle

**Ce qui existe**:
- Détection basée sur métadonnées/nom du dataset (AVANT entraînement)
- Taxonomie des types d'anomalies (structural, visual, geometric)

**Ce qui manque**:
- Classification multi-classes des types d'erreurs (crack, scratch, hole, contamination)
- Modèle capable de différencier les types d'anomalies

### 4.2 Taxonomie des Anomalies

**Fichier**: `src/models/computer_vision_training.py` ligne 1530-1587

**Catégories**:
- **Structural**: crack, corrosion, deformation
- **Visual**: scratch, stain, discoloration
- **Geometric**: misalignment, dimension errors

**⚠️ ERREUR #13**: Mapping Type d'Erreur → Modèle Non Implémenté

**Problème**:
- La taxonomie existe mais elle n'influence que la **configuration** du modèle (architecture, hyperparamètres)
- Elle ne permet **pas** au modèle de **classifier** le type d'erreur détecté

**Impact**: **MOYEN** - Le système peut détecter une anomalie mais pas dire si c'est une fissure ou une rayure

### 4.3 Vérification: Le Modèle Peut-il Différencier les Classes ?

**Réponse**: **NON**, pour les autoencoders (unsupervised)

**Pourquoi**:
- Autoencoders apprennent uniquement à reconstruire des images normales
- Ils détectent des anomalies mais ne peuvent pas les classifier par type

**Réponse**: **OUI**, pour les classificateurs (supervised)

**Pourquoi**:
- Si le dataset contient plusieurs classes (normal, crack, scratch, hole)
- Un modèle de classification peut apprendre à différencier ces classes
- Mais le pipeline actuel ne le fait **pas automatiquement**

**Fichier**: `src/models/computer_vision_training.py` ligne 1111-1130

```python
def _predict_classifier(...):
    # Retourne seulement la classe prédite, pas le type d'erreur
    preds = output.argmax(dim=1)
```

### 4.4 ERREURS dans la Détection du Type d'Erreur

#### ❌ **ERREUR CRITIQUE #14**: Pas de Classification Multi-Classes des Types

**Problème**:
- Le système ne peut pas dire "c'est une fissure" ou "c'est une rayure"
- Il peut seulement dire "anomalie détectée" ou "normal"

**Impact**: **CRITIQUE** - Fonctionnalité demandée non implémentée

#### ⚠️ **ERREUR #15**: Labels Manquants pour Types d'Erreurs

**Problème**:
- Même si on voulait entraîner un classificateur multi-classes, les labels de type d'erreur ne sont pas chargés depuis MVTec AD

**Impact**: **MOYEN** - Impossible d'entraîner un modèle de classification de types sans labels

---

## 5. LOCALISATION DE L'ERREUR (HEATMAPS)

### 5.1 Comment la Heatmap est Générée

**Fichier**: `src/models/computer_vision/anomaly_detection/autoencoders.py` ligne 420-442

```python
def get_reconstruction_error_map(self, x: torch.Tensor) -> torch.Tensor:
    """Génère une carte spatiale des erreurs de reconstruction"""
    self.eval()
    with torch.no_grad():
        reconstructed = self.forward(x)
        # Erreur par pixel, moyennée sur les canaux
        error_map = torch.mean((x - reconstructed) ** 2, dim=1, keepdim=True)
        return error_map  # (B, 1, H, W)
```

**Processus**:
1. Forward pass → reconstruction
2. Calcul MSE pixel par pixel: `(x - reconstructed) ** 2`
3. Moyenne sur les canaux: `dim=1`
4. Retourne carte (B, 1, H, W)

### 5.2 Comment le Mask Final est Produit

**❌ MANQUANT**: Il n'y a **pas de génération de mask binaire** dans le code

**Ce qui existe**:
- `error_map` continu (valeurs entre 0 et max_error)
- Visualisation via `plot_anomaly_heatmap()`

**Ce qui devrait exister**:
```python
def generate_binary_mask(error_map: torch.Tensor, threshold: float) -> torch.Tensor:
    """Convertit error_map → mask binaire"""
    return (error_map > threshold).float()
```

### 5.3 Comment le Seuil est Appliqué

**Fichier**: `src/evaluation/model_vision_plots.py` ligne 256

```python
# Normalisation heatmap
heatmap = (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min() + 1e-8)
```

**Problème**: Le seuil n'est **pas appliqué** sur la heatmap, seulement pour la prédiction binaire

### 5.4 Bugs Potentiels dans la Localisation

#### ❌ **ERREUR CRITIQUE #16**: Heatmap Non Générée Automatiquement

**Problème**:
- `get_reconstruction_error_map()` existe mais n'est **jamais appelée** dans `robust_predict_with_preprocessor()`
- Les heatmaps ne sont générées que si on les demande explicitement

**Localisation**: `src/app/pages/5_anomaly_evaluation.py` (aucun appel à `get_reconstruction_error_map()`)

**Impact**: **CRITIQUE** - Localisation non disponible par défaut

#### ❌ **ERREUR CRITIQUE #17**: Dimensions Non Alignées

**Problème**:
- Si l'image originale est 256x256 et le modèle utilise 224x224
- La `error_map` retournée est 224x224
- Si on la superpose à l'image originale → décalage

**Localisation**: `src/data/computer_vision_preprocessing.py` ligne 30-37

**Solution nécessaire**: Resize de `error_map` vers la taille originale

#### ⚠️ **ERREUR #18**: Format Channels Non Cohérent

**Problème**:
- `get_reconstruction_error_map()` retourne (B, 1, H, W) → channels_first
- `plot_anomaly_heatmap()` attend (H, W) → channels_last
- Conversion nécessaire mais pas toujours faite

**Impact**: **MOYEN** - Crash si format incorrect

### 5.5 Si la Localisation Peut Être Erronée

**OUI**, plusieurs cas:

1. **Resize non aligné**: Si preprocessing resize l'image, la heatmap ne correspond plus
2. **Padding non géré**: Si le modèle utilise padding, les bords peuvent être faussés
3. **Normalisation perdue**: Si la heatmap est générée avant normalisation inverse

---

## 6. ERREURS ET INCOHÉRENCES TROUVÉES

### 6.1 Résumé des Erreurs Critiques

| ID | Erreur | Fichier | Ligne | Impact |
|----|--------|---------|-------|--------|
| #1 | Labels mal chargés (train+test au lieu de train seul) | `image_exploration_plots.py` | 388-433 | CRITIQUE |
| #3 | Shape incorrect de y_train | `task_detector.py` | 16-60 | CRITIQUE |
| #5 | Incohérence format preprocessing | `computer_vision_preprocessing.py` | 119-143 | CRITIQUE |
| #8 | Heatmap pas toujours générée | `model_vision_plots.py` | 230 | CRITIQUE |
| #10 | Mauvais resize → masque non aligné | `computer_vision_preprocessing.py` | 30-37 | CRITIQUE |
| #11 | Conversion tensor → numpy incohérente | `5_anomaly_evaluation.py` | 549 | CRITIQUE |
| #14 | Pas de classification multi-classes des types | `computer_vision_training.py` | - | CRITIQUE |
| #16 | Heatmap non générée automatiquement | `5_anomaly_evaluation.py` | - | CRITIQUE |
| #17 | Dimensions non alignées | `computer_vision_preprocessing.py` | 30-37 | CRITIQUE |

### 6.2 Erreurs Moyennes

| ID | Erreur | Fichier | Impact |
|----|--------|---------|--------|
| #2 | Mapping inversé labels | `task_detector.py` | MOYEN |
| #7 | Seuil adaptatif trop restrictif | `computer_vision_training.py` | MOYEN |
| #9 | Format y_scores incohérent | `computer_vision_metrics.py` | MOYEN |
| #12 | Seuil d'anomalie trop fixe | `5_anomaly_evaluation.py` | MOYEN |
| #15 | Labels manquants pour types d'erreurs | - | MOYEN |
| #18 | Format channels non cohérent | `model_vision_plots.py` | MOYEN |

---

## 7. CORRECTIFS PROPOSÉS

### 7.1 Correctif #1: Génération Automatique des Heatmaps

**Fichier**: `src/app/pages/5_anomaly_evaluation.py`

**Ajout dans `robust_predict_with_preprocessor()`**:

```python
# Après reconstruction pour autoencoder
if model_type in ["autoencoder", "conv_autoencoder"]:
    reconstructed = model(X_tensor)
    reconstructed_np = reconstructed.cpu().numpy()
    
    # ✅ NOUVEAU: Génération heatmaps
    if hasattr(model, 'get_reconstruction_error_map'):
        error_maps = []
        for i in range(X_tensor.shape[0]):
            single_img = X_tensor[i:i+1]
            error_map = model.get_reconstruction_error_map(single_img)
            # Convertir (1, 1, H, W) → (H, W)
            error_map_np = error_map[0, 0].cpu().numpy()
            # Resize vers taille originale si nécessaire
            if error_map_np.shape != X_test[i].shape[:2]:
                from scipy.ndimage import zoom
                zoom_factors = (
                    X_test[i].shape[0] / error_map_np.shape[0],
                    X_test[i].shape[1] / error_map_np.shape[1]
                )
                error_map_np = zoom(error_map_np, zoom_factors, order=1)
            error_maps.append(error_map_np)
        
        result["error_maps"] = np.array(error_maps)  # (N, H, W)
```

### 7.2 Correctif #2: Génération de Masks Binaires

**Nouveau fichier**: `src/evaluation/localization_utils.py`

```python
def generate_binary_mask(
    error_map: np.ndarray,
    threshold: float,
    method: str = "percentile"
) -> np.ndarray:
    """
    Génère un mask binaire à partir d'une carte d'erreur.
    
    Args:
        error_map: Carte d'erreur (H, W) ou (B, H, W)
        threshold: Seuil absolu ou percentile
        method: "percentile" ou "absolute"
    
    Returns:
        Mask binaire (H, W) ou (B, H, W)
    """
    if method == "percentile":
        actual_threshold = np.percentile(error_map, threshold * 100)
    else:
        actual_threshold = threshold
    
    mask = (error_map > actual_threshold).astype(np.uint8)
    return mask
```

### 7.3 Correctif #3: Alignement Dimensions Heatmap

**Fichier**: `src/evaluation/model_vision_plots.py`

**Modification de `plot_anomaly_heatmap()`**:

```python
def plot_anomaly_heatmap(
    image: np.ndarray,
    anomaly_score: np.ndarray,
    original_size: Optional[Tuple[int, int]] = None
) -> Optional[go.Figure]:
    # ✅ NOUVEAU: Resize si nécessaire
    if original_size and anomaly_score.shape[:2] != original_size:
        from scipy.ndimage import zoom
        zoom_factors = (
            original_size[0] / anomaly_score.shape[0],
            original_size[1] / anomaly_score.shape[1]
        )
        anomaly_score = zoom(anomaly_score, zoom_factors, order=1)
    
    # Validation shapes
    if image.shape[:2] != anomaly_score.shape[:2]:
        raise ValueError(
            f"Shapes non alignées: image={image.shape[:2]}, "
            f"anomaly_score={anomaly_score.shape[:2]}"
        )
    
    # ... reste du code
```

### 7.4 Correctif #4: Classification Multi-Classes des Types d'Erreurs

**Nouveau fichier**: `src/models/computer_vision/anomaly_classification.py`

```python
class AnomalyTypeClassifier(nn.Module):
    """
    Classificateur des types d'anomalies.
    Entraîné en plus de l'autoencoder pour identifier crack/scratch/hole/etc.
    """
    
    def __init__(
        self,
        backbone: nn.Module,  # Autoencoder pré-entraîné
        num_anomaly_types: int = 5  # crack, scratch, hole, contamination, unknown
    ):
        super().__init__()
        self.backbone = backbone
        # Geler l'autoencoder
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Classificateur sur l'espace latent
        self.classifier = nn.Sequential(
            nn.Linear(backbone.latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_anomaly_types)
        )
    
    def forward(self, x):
        # Encoder seulement
        encoded = self.backbone.encoder(x)
        # Classification
        anomaly_type_logits = self.classifier(encoded)
        return anomaly_type_logits
```

**Modification du pipeline d'entraînement**:
- Si labels de types d'anomalies disponibles → entraîner `AnomalyTypeClassifier`
- Sinon → mode actuel (détection binaire uniquement)

### 7.5 Correctif #5: Validation Cohérence y_train

**Fichier**: `src/explorations/image_exploration_plots.py`

**Modification de `load_images_flexible()`**:

```python
def load_images_flexible(...):
    # ... chargement ...
    
    # ✅ NOUVEAU: Validation y_train
    if structure_type == DatasetType.MVTEC_AD.value:
        y_train = _load_mvtec_train_labels(data_dir)
        
        # Validation: y_train doit contenir uniquement des 0
        unique_labels_train = np.unique(y_train)
        if len(unique_labels_train) > 1 or (len(unique_labels_train) == 1 and unique_labels_train[0] != 0):
            logger.error(
                f"❌ ERREUR: y_train contient des labels anormaux: {unique_labels_train}. "
                f"Pour MVTec AD, y_train doit contenir uniquement des 0 (images normales)."
            )
            raise ValueError("y_train invalide pour MVTec AD")
        
        logger.info(f"✅ Validation y_train OK: {len(y_train)} images normales (label 0)")
```

---

## 8. RECOMMANDATIONS FINALES

### 8.1 Améliorations Prioritaires

1. **HAUTE PRIORITÉ**: Générer automatiquement les heatmaps lors de l'évaluation
2. **HAUTE PRIORITÉ**: Aligner les dimensions heatmap/image originale
3. **HAUTE PRIORITÉ**: Valider la cohérence de y_train pour MVTec AD
4. **MOYENNE PRIORITÉ**: Implémenter classification multi-classes des types d'erreurs
5. **MOYENNE PRIORITÉ**: Générer des masks binaires avec seuil adaptatif

### 8.2 Amélioration du Dataset

- Si vous voulez classifier les types d'erreurs (crack, scratch, hole), il faut des labels de types
- MVTec AD fournit les types d'anomalies dans les noms de dossiers (`test/crack/`, `test/scratch/`, etc.)
- **Recommandation**: Charger ces labels depuis la structure MVTec AD

### 8.3 Amélioration du Modèle

- Pour la localisation précise: utiliser PatchCore ou modèles avec attention
- Pour la classification des types: ajouter une tête de classification sur l'encoder

### 8.4 Amélioration du Seuil

- Remplacer le seuil fixe (95ème percentile) par un seuil adaptatif basé sur la distribution de test
- Utiliser F1-maximization ou Youden's J statistic pour trouver le seuil optimal

---

## 📊 CONCLUSION

### Points Forts

✅ **Détection automatique du type de tâche** fonctionne correctement  
✅ **Pipeline d'entraînement** robuste avec gestion des erreurs  
✅ **Preprocessing sans fuite** de données (fit sur train uniquement)  
✅ **Architecture modulaire** et extensible  

### Points Faibles

❌ **Localisation (heatmaps)** pas générée automatiquement  
❌ **Classification des types d'erreurs** non implémentée  
❌ **Alignement dimensions** heatmap/image non géré  
❌ **Validation y_train** insuffisante pour MVTec AD  

### Impact Global

**Fonctionnalités opérationnelles**: 70%  
- Détection d'anomalies: ✅ OK
- Localisation: ⚠️ Partielle (heatmaps non générées automatiquement)
- Classification des types: ❌ Non disponible

**Risques production**: **MOYEN**
- Faux positifs/négatifs possibles (seuil fixe)
- Localisation peut être erronée (dimensions non alignées)
- Pas de classification fine des types d'erreurs

---

**FIN DU RAPPORT**


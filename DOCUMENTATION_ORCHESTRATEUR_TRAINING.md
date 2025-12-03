# 📚 Documentation Complète - Orchestrateur, Training Bas Niveau et Prédictions

## 📋 Table des Matières

1. [Architecture Globale](#architecture-globale)
2. [Orchestrateur (Niveau Haut)](#orchestrateur-niveau-haut)
3. [Trainer Bas Niveau](#trainer-bas-niveau)
4. [Pipeline de Prédiction](#pipeline-de-prédiction)
5. [Pipeline d'Évaluation](#pipeline-dévaluation)
6. [Flux Complet de Bout en Bout](#flux-complet-de-bout-en-bout)

---

## 🏗️ Architecture Globale

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI (Pages)                            │
│  • 4_training_computer.py  →  Lancement training                       │
│  • 5_anomaly_evaluation.py →  Visualisation résultats                  │
└────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STATE MANAGER (Global State)                         │
│  • STATE.data           → Données chargées                             │
│  • STATE.training_results → Résultats training                         │
│  • STATE.model_config   → Configuration modèle                         │
└────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              ORCHESTRATEUR (visio_training_orchestrator.py)             │
│  • ComputerVisionTrainingOrchestrator                                   │
│  • Coordination complète du workflow                                    │
│  • Intégration MLflow                                                   │
│  • Gestion preprocessing                                                │
└────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              TRAINER BAS NIVEAU (computer_vision_training.py)           │
│  • ComputerVisionTrainer (Supervisé)                                    │
│  • AnomalyAwareTrainer (Anomalies)                                      │
│  • Méthodes: fit(), predict(), evaluate()                               │
└────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    MODÈLES & UTILITAIRES                                │
│  • ModelBuilder          → Construction modèles                        │
│  • DataPreprocessor      → Preprocessing                               │
│  • DataLoaderFactory     → Création DataLoaders                        │
│  • OptimizerFactory      → Création optimizers                         │
│  • SchedulerFactory      → Création schedulers                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Orchestrateur (Niveau Haut)

### **Fichier**: `orchestrators/visio_training_orchestrator.py`

### **Classe Principale**: `ComputerVisionTrainingOrchestrator`

L'orchestrateur est le **coordinateur central** qui orchestre tout le workflow d'entraînement. Il ne fait **PAS** l'entraînement lui-même, mais :
- ✅ Valide les configurations
- ✅ Démarre le run MLflow
- ✅ Crée le trainer approprié
- ✅ Lance l'entraînement
- ✅ Log les métriques et artifacts dans MLflow
- ✅ Retourne un `TrainingResult` standardisé

---

### **Méthode Principale**: `train(context: TrainingContext) -> TrainingResult`

#### **Flux d'Exécution**:

```python
def train(self, context: TrainingContext) -> TrainingResult:
    """
    Pipeline complet orchestré:
    
    1. VALIDATION DES CONFIGS
    2. VALIDATION DES DONNÉES
    3. DÉMARRAGE MLFLOW RUN
    4. LOG CONFIGURATION
    5. CRÉATION TRAINER
    6. EXÉCUTION ENTRAÎNEMENT
    7. RÉCUPÉRATION PREPROCESSOR
    8. LOG MÉTRIQUES & ARTIFACTS
    9. FINALISATION
    """
```

#### **Étape 1: Conversion et Validation des Configs**

```python
# Ligne 83
context.training_config = self._ensure_training_config_object(context.training_config)
```

**Fonctionnalités**:
- ✅ Convertit `dict` → `TrainingConfig` object
- ✅ Valide les Enums (`OptimizerType`, `SchedulerType`)
- ✅ Validation stricte des valeurs (epochs > 0, lr dans ]0,1[, etc.)
- ✅ Fallback sécurisé sur valeurs par défaut

**Exemple**:
```python
# Input: dict
training_config = {
    "epochs": 100,
    "batch_size": 32,
    "optimizer": "adamw",  # String → Converti en OptimizerType.ADAMW
    "scheduler": "reduce_on_plateau"  # String → Converti en SchedulerType.REDUCE_ON_PLATEAU
}

# Output: TrainingConfig object avec Enums validés
```

---

#### **Étape 2: Validation des Données**

```python
# Ligne 88
self._validate_training_context(context)
```

**Vérifications**:
- ✅ `X_train` et `X_val` non None
- ✅ Datasets non vides
- ✅ Logging des shapes pour observabilité

---

#### **Étape 3: Démarrage MLflow Run**

```python
# Ligne 93
run_id = self._start_mlflow_run(context)
```

**Actions**:
- ✅ Création run MLflow avec tags enrichis
- ✅ Tags: `anomaly_type`, `dataset_size`, `train_val_split`, `model_type`
- ✅ Retourne `run_id` pour tracking

---

#### **Étape 4: Log Configuration dans MLflow**

```python
# Ligne 98
self._log_configuration_to_mlflow(context)
```

**Logs**:
- ✅ `model_config`: type, params du modèle
- ✅ `training_config`: epochs, batch_size, lr, optimizer, scheduler
- ✅ `preprocessing_config`: stratégie, augmentation

---

#### **Étape 5: Création du Trainer**

```python
# Ligne 103
trainer = self._create_trainer(context)
```

**Logique**:
```python
if context.anomaly_type:
    # Création AnomalyAwareTrainer (wrapper autour de ComputerVisionTrainer)
    return AnomalyAwareTrainer(
        anomaly_type=context.anomaly_type,
        model_config=model_config,
        training_config=context.training_config,
        callbacks=callbacks
    )
else:
    # Création ComputerVisionTrainer standard
    return ComputerVisionTrainer(
        model_config=model_config,
        training_config=context.training_config,
        callbacks=callbacks
    )
```

---

#### **Étape 6: Exécution de l'Entraînement**

```python
# Ligne 108
result = self._execute_training(trainer, context)
```

**Fonctionnalités Clés** (`_execute_training`):

1. **Appel du Trainer**:
   ```python
   if context.anomaly_type:
       raw_result = trainer.train(X_train, y_train, X_val, y_val)  # AnomalyAwareTrainer
   else:
       raw_result = trainer.fit(X_train, y_train, X_val, y_val)  # ComputerVisionTrainer
   ```

2. **Normalisation Robuste du Résultat**:
   - ✅ Gère 4 formats possibles:
     - `dict` direct
     - Objet `Result` avec `.data`
     - Objet avec `.success` et `.history`
     - Format inconnu (erreur)
   - ✅ **Garantit** un dict normalisé avec clés: `success`, `data`, `error`, `metadata`
   - ✅ Vérifie que `data.history` existe (fallback si manquant)

3. **Validation Finale**:
   ```python
   if result['success'] and not result.get('data', {}).get('history'):
       result['success'] = False
       result['error'] = "Historique d'entraînement manquant"
   ```

---

#### **Étape 7: Récupération Preprocessor**

```python
# Ligne 122
preprocessor = self._get_preprocessor(trainer, context)
```

**Actions**:
- ✅ Récupère `trainer.preprocessor`
- ✅ Si None → Création fallback avec config par défaut
- ✅ Fit le fallback sur `X_train`

---

#### **Étape 8: Log Métriques et Artifacts**

```python
# Lignes 128-129
self._log_training_metrics(history)
self._log_training_artifacts(trainer.model, preprocessor, context, run_id)
```

**Log Métriques**:
- ✅ Métriques par epoch: `train_loss`, `val_loss`, `val_accuracy`, `val_f1`, `learning_rate`
- ✅ Métriques finales: `best_val_loss`, `training_time`, `total_epochs`, `best_epoch`
- ✅ Log courbes d'entraînement

**Log Artifacts**:
- ✅ Modèle PyTorch (`.pt` file)
- ✅ Preprocessor (pickle ou config JSON si non-picklable)
- ✅ `model_config.json` (sérialisation safe)
- ✅ Gestion robuste des erreurs (try/catch sur chaque artifact)

---

#### **Étape 9: Finalisation**

```python
# Ligne 134
cv_mlflow_tracker.end_run("FINISHED")

# Ligne 137
final_history = self._build_final_history(history, context, run_id, preprocessor)
```

**Construction Historique Final** (`_build_final_history`):
- ✅ Extraction safe des métriques (gestion None, NaN, Inf)
- ✅ Conversion types (float, int, list)
- ✅ Ajout métadonnées: `mlflow_run_id`, `preprocessor_config`, `anomaly_type`
- ✅ Fallback historique minimal si erreur

**Retour `TrainingResult`**:
```python
return TrainingResult(
    success=True,
    model=trainer.model,  # Modèle PyTorch entraîné
    history=final_history,  # Dict normalisé avec métriques
    preprocessor=preprocessor,  # Preprocessor fitté
    mlflow_run_id=run_id,  # ID du run MLflow
    metadata={
        "model_type": context.model_config["model_type"],
        "total_epochs": history.get('total_epochs_trained', 0),
        "best_epoch": history.get('best_epoch', 0)
    }
)
```

---

### **Gestion des Erreurs**

```python
except Exception as e:
    # Log erreur
    logger.error(f"Erreur orchestration entraînement: {e}", exc_info=True)
    
    # End MLflow run en FAILED
    if run_id:
        cv_mlflow_tracker.log_metrics({"training_failed": 1.0})
        cv_mlflow_tracker.end_run("FAILED")
    
    # Retour TrainingResult avec success=False
    return TrainingResult(
        success=False,
        error=str(e),
        mlflow_run_id=run_id,
        metadata={"error_type": type(e).__name__, ...}
    )
```

---

## 🔧 Trainer Bas Niveau

### **Fichier**: `src/models/computer_vision_training.py`

### **Classe Principale**: `ComputerVisionTrainer`

Le trainer est responsable de l'**entraînement réel** du modèle. Il gère:
- ✅ Preprocessing (fit sur train, transform sur val)
- ✅ Construction du modèle
- ✅ Setup optimizer/scheduler/criterion
- ✅ Boucle d'entraînement avec early stopping
- ✅ Prédictions et évaluations

---

### **Méthode Principale**: `fit(X_train, y_train, X_val, y_val) -> Result`

#### **Pipeline Complet**:

```python
def fit(self, X_train, y_train, X_val, y_val) -> Result:
    """
    Pipeline d'entraînement:
    
    1. VALIDATION DES DONNÉES
    2. SETUP PREPROCESSING (fit sur train, transform sur val)
    3. CONSTRUCTION MODÈLE
    4. SETUP TRAINING (optimizer, scheduler, criterion)
    5. CRÉATION DATALOADERS
    6. BOUCLE D'ENTRAÎNEMENT
    7. RETOUR RÉSULTATS STRUCTURÉS
    """
```

---

#### **Étape 1: Validation des Données**

```python
# Ligne 368
val_result = self._validate_data(X_train, y_train, X_val, y_val)
```

**Vérifications** (`_validate_data`):
- ✅ Validation `X_train`, `y_train` (shapes, types, non-vides)
- ✅ Validation `X_val`, `y_val`
- ✅ Cohérence shapes entre train et val
- ✅ Analyse déséquilibre classes (warning si `ratio < 0.1`)

**Retour**: `Result.err()` si validation échoue, sinon `Result.ok(None, imbalance=...)`

---

#### **Étape 2: Setup Preprocessing**

```python
# Ligne 373
prep_result = self._setup_preprocessing(X_train, y_train, X_val, y_val)
```

**Actions Critiques** (`_setup_preprocessing`):

1. **Création Preprocessor**:
   ```python
   self.preprocessor = DataPreprocessor(
       strategy="standardize",
       auto_detect_format=True  # Détection automatique channels_first/last
   )
   ```

2. **FIT sur TRAIN UNIQUEMENT** (⚠️ Pas de fuite de données):
   ```python
   X_train_norm = self.preprocessor.fit_transform(
       X_train,
       output_format="channels_first"  # Format PyTorch
   )
   ```

3. **TRANSFORM sur VALIDATION** (même format):
   ```python
   X_val_norm = self.preprocessor.transform(
       X_val,
       output_format="channels_first"
   )
   ```

4. **Validations Post-Processing**:
   - ✅ Vérification non-None
   - ✅ Format 4D (N, C, H, W)
   - ✅ Canaux valides (1 ou 3)
   - ✅ Vérification NaN/Inf

**Retour**: `Result.ok((X_train_norm, y_train, X_val_norm, y_val))` ou `Result.err(...)`

---

#### **Étape 3: Construction du Modèle**

```python
# Ligne 380
model_result = self._build_model()
```

**Actions** (`_build_model`):
- ✅ Utilise `ModelBuilder` pour construire le modèle selon `model_config`
- ✅ Log nombre de paramètres (total et trainable)
- ✅ Stocke dans `self.model`

**Types de Modèles Supportés**:
- `CNN` (Classification)
- `RESNET_TRANSFER`, `EFFICIENTNET_TRANSFER` (Transfer Learning)
- `CONV_AUTOENCODER`, `VAE`, `DENOISING_AE` (Anomaly Detection)

---

#### **Étape 4: Setup Training**

```python
# Ligne 385
setup_result = self._setup_training(y_train)
```

**Actions** (`_setup_training`):

1. **Optimizer**:
   ```python
   self.optimizer = OptimizerFactory.create(self.model, self.training_config)
   # Support: ADAM, ADAMW, SGD, RMSPROP
   ```

2. **Scheduler**:
   ```python
   self.scheduler = SchedulerFactory.create(self.optimizer, self.training_config)
   # Support: REDUCE_ON_PLATEAU, COSINE_ANNEALING, STEP_LR
   ```

3. **Criterion (Loss)**:
   ```python
   # Détection type de modèle
   is_autoencoder = model_type in [CONV_AUTOENCODER, VAE, DENOISING_AE]
   
   if is_autoencoder:
       self.train_criterion = nn.MSELoss()  # Reconstruction loss
       self.val_criterion = nn.MSELoss()
   else:
       # Classification
       if use_class_weights:
           weights = compute_class_weight('balanced', classes, y_train)
           self.train_criterion = nn.CrossEntropyLoss(weight=weights_tensor)
       else:
           self.train_criterion = nn.CrossEntropyLoss()
       
       self.val_criterion = nn.CrossEntropyLoss()  # Validation sans weights
   ```

---

#### **Étape 5: Création DataLoaders**

```python
# Lignes 390-404
train_loader = DataLoaderFactory.create(
    X_train_norm, y_train,
    batch_size=self.training_config.batch_size,
    shuffle=True,  # ✅ Shuffle pour training
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoaderFactory.create(
    X_val_norm, y_val,
    batch_size=self.training_config.batch_size,
    shuffle=False,  # ✅ Pas de shuffle pour validation
    num_workers=0,
    pin_memory=False
)
```

---

#### **Étape 6: Boucle d'Entraînement**

```python
# Ligne 407
train_result = self._training_loop(train_loader, val_loader, y_val)
```

**Pipeline de `_training_loop`**:

```python
def _training_loop(self, train_loader, val_loader, y_val) -> Result:
    # Initialisation
    best_val_metric = float('inf') if is_autoencoder else 0.0
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    
    # Callbacks: on_train_begin()
    
    for epoch in range(self.training_config.epochs):
        # Callbacks: on_epoch_begin(epoch)
        
        # === PHASE TRAIN ===
        train_loss = self._train_epoch(train_loader, is_autoencoder)
        
        # === PHASE VALIDATION ===
        if is_autoencoder:
            val_loss = self._validate_epoch_autoencoder(val_loader)
            val_metrics = {'loss': val_loss}
        else:
            val_loss, val_metrics = self._validate_epoch(val_loader, y_val)
            # val_metrics = {'accuracy', 'f1', 'loss'}
        
        # === MISE À JOUR HISTORIQUE ===
        self.history['train_loss'].append(float(train_loss))
        self.history['val_loss'].append(float(val_loss))
        if not is_autoencoder:
            self.history['val_accuracy'].append(float(val_metrics['accuracy']))
            self.history['val_f1'].append(float(val_metrics['f1']))
        self.history['learning_rates'].append(current_lr)
        
        # === SCHEDULER STEP ===
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)  # Step avec métrique
            else:
                self.scheduler.step()  # Step simple
        
        # === EARLY STOPPING ===
        if is_autoencoder:
            improved = val_loss < best_val_metric  # Minimiser loss
        else:
            improved = val_metrics['f1'] > best_val_metric  # Maximiser F1
        
        if improved:
            best_model_state = copy.deepcopy(self.model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
            # Checkpoint si configuré
        else:
            patience_counter += 1
        
        # Check early stopping
        if patience_counter >= self.training_config.early_stopping_patience:
            logger.info(f"Early stopping déclenché à l'epoch {epoch+1}")
            break
        
        # Callbacks: on_epoch_end(epoch, logs)
    
    # === RESTAURATION MEILLEUR MODÈLE ===
    if best_model_state is not None:
        self.model.load_state_dict(best_model_state)
    
    # Callbacks: on_train_end({'training_time': ...})
    
    return Result.ok({...}, training_time=..., best_epoch=...)
```

**Détails `_train_epoch`** (Phase Train):
```python
def _train_epoch(self, train_loader, is_autoencoder):
    self.model.train()  # Mode training
    total_loss = 0.0
    
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        
        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(data)
        
        if is_autoencoder:
            loss = self.train_criterion(output, data)  # Reconstruction
        else:
            loss = self.train_criterion(output, target)  # Classification
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.training_config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.gradient_clip
            )
        
        self.optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)
```

**Détails `_validate_epoch`** (Phase Validation):
```python
def _validate_epoch(self, val_loader, y_val):
    self.model.eval()  # Mode evaluation
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():  # Pas de gradients
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            
            output = self.model(data)
            loss = self.val_criterion(output, target)
            
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calcul métriques
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, {'accuracy': accuracy, 'f1': f1}
```

---

#### **Étape 7: Retour Résultats Structurés**

```python
# Ligne 421
return Result.ok(
    self._build_training_result(train_result),
    training_time=total_time
)
```

**Structure Retournée** (`_build_training_result`):
```python
{
    'model': self.model,  # Modèle PyTorch entraîné
    'preprocessor': self.preprocessor,  # Preprocessor fitté
    'history': {
        'success': True,  # Bool (exception)
        'model_type': 'conv_autoencoder',
        'is_autoencoder': True,
        
        # Métriques (LISTES de float)
        'train_loss': [0.5, 0.4, 0.3, ...],
        'val_loss': [0.6, 0.5, 0.4, ...],
        'val_accuracy': [...],  # Vide si autoencoder
        'val_f1': [...],  # Vide si autoencoder
        'learning_rates': [1e-4, 1e-4, 9e-5, ...],
        
        # Résumé
        'best_epoch': 45,
        'best_val_loss': 0.35,
        'final_train_loss': 0.32,
        'training_time': 1200.5,
        'total_epochs_trained': 50,
        'early_stopping_triggered': False,
        
        # Config
        'input_shape': (3, 256, 256),
        'output_format': 'channels_first',
        'training_config': {...},
        
        # Métadonnées
        'metadata': {...}
    }
}
```

---

### **Classe**: `AnomalyAwareTrainer`

**Wrapper** autour de `ComputerVisionTrainer` pour les tâches d'anomalie avec taxonomy.

**Méthode**: `train(X_train, y_train, X_val, y_val) -> Result`

```python
def train(self, X_train, y_train, X_val, y_val, callbacks=None) -> Result:
    # Création trainer standard
    trainer = ComputerVisionTrainer(
        model_config=self.model_config,
        training_config=self.training_config,
        callbacks=active_callbacks
    )
    
    # Délégation à fit standard
    result = trainer.fit(X_train, y_train, X_val, y_val)
    
    # Copie attributs pour compatibilité
    if result.success:
        self.model = trainer.model
        self.preprocessor = trainer.preprocessor
        self.history = result.data['history']
    
    return result
```

**Note**: `AnomalyAwareTrainer` utilise le **même pipeline** que `ComputerVisionTrainer`. La différence est principalement au niveau de la **configuration** (taxonomy, anomaly_type).

---

## 🔮 Pipeline de Prédiction

### **Méthode**: `predict(X, return_reconstructed=False, batch_size=None) -> Result`

#### **Flux Complet**:

```python
def predict(self, X, return_reconstructed=False, batch_size=None) -> Result:
    """
    Pipeline de prédiction:
    
    1. VALIDATION (modèle et preprocessor disponibles)
    2. PREPROCESSING (transform uniquement, pas fit!)
    3. DÉTECTION TYPE MODÈLE
    4. CRÉATION DATALOADER
    5. PRÉDICTION (classifier ou autoencoder)
    6. RETOUR RÉSULTATS STRUCTURÉS
    """
```

---

#### **Étape 1: Validation**

```python
# Lignes 1031-1034
if self.model is None:
    return Result.err("Modèle non entraîné")
if self.preprocessor is None:
    return Result.err("Preprocessor non disponible")
```

---

#### **Étape 2: Preprocessing**

```python
# Ligne 1037
X_processed = self.preprocessor.transform(X, output_format="channels_first")
```

**⚠️ IMPORTANT**: `transform()` uniquement, **JAMAIS** `fit_transform()` sur données de test!

---

#### **Étape 3: Détection Type Modèle**

```python
# Lignes 1040-1042
is_autoencoder = self.model_config.model_type in [
    ModelType.CONV_AUTOENCODER, ModelType.VAE, ModelType.DENOISING_AE
]
```

---

#### **Étape 4: Création DataLoader**

```python
# Lignes 1045-1054
batch_size = batch_size or self.training_config.batch_size
dummy_labels = np.zeros(len(X_processed))  # Labels dummy (non utilisés)

test_loader = DataLoaderFactory.create(
    X_processed, dummy_labels,
    batch_size=batch_size,
    shuffle=False,  # ✅ Pas de shuffle pour prédictions
    num_workers=0,
    pin_memory=False
)
```

---

#### **Étape 5: Prédiction**

```python
# Ligne 1057
self.model.eval()  # Mode evaluation

if is_autoencoder:
    return self._predict_autoencoder(test_loader, X_processed, return_reconstructed)
else:
    return self._predict_classifier(test_loader)
```

---

### **Prédiction Autoencoder** (`_predict_autoencoder`)

```python
def _predict_autoencoder(self, test_loader, X_processed, return_reconstructed) -> Result:
    reconstruction_errors = []
    reconstructed_images = [] if return_reconstructed else None
    error_maps_list = []  # ✅ Cartes d'erreur spatiales
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            reconstructed = self.model(data)
            
            # === ERREUR PAR ÉCHANTILLON ===
            errors = torch.mean(
                (data - reconstructed) ** 2,
                dim=tuple(range(1, data.ndim))  # Moyenne sur C, H, W
            ).cpu().numpy()
            reconstruction_errors.extend(errors)
            
            # === CARTES D'ERREUR SPATIALES ===
            if hasattr(self.model, 'get_reconstruction_error_map'):
                batch_error_maps = self.model.get_reconstruction_error_map(data)
                batch_error_maps_np = batch_error_maps[:, 0, :, :].cpu().numpy()  # (B, H, W)
            else:
                # Fallback: calcul manuel
                batch_error_maps = torch.mean((data - reconstructed) ** 2, dim=1, keepdim=True)
                batch_error_maps_np = batch_error_maps[:, 0, :, :].cpu().numpy()
            
            error_maps_list.append(batch_error_maps_np)
            
            if return_reconstructed:
                reconstructed_images.append(reconstructed.cpu().numpy())
    
    reconstruction_errors = np.array(reconstruction_errors)
    
    # === SEUIL AUTOMATIQUE (95ème percentile) ===
    threshold = np.percentile(reconstruction_errors, 95)
    predictions = (reconstruction_errors > threshold).astype(int)
    
    # === GÉNÉRATION HEATMAPS ===
    if error_maps_list:
        error_maps = np.concatenate(error_maps_list, axis=0)  # (N, H, W)
        
        # Normalisation pour heatmaps
        heatmaps = []
        for error_map in error_maps:
            if error_map.max() > error_map.min():
                normalized = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)
            else:
                normalized = np.zeros_like(error_map)
            heatmaps.append(normalized)
        heatmaps = np.array(heatmaps)  # (N, H, W)
    
    # === CONSTRUCTION RÉSULTAT ===
    result_data = {
        'reconstruction_errors': reconstruction_errors,  # (N,)
        'predictions': predictions,  # (N,) binaires (0=normal, 1=anomaly)
        'threshold': float(threshold),
        'error_maps': error_maps,  # (N, H, W)
        'heatmaps': heatmaps,  # (N, H, W) normalisées [0, 1]
    }
    
    if return_reconstructed:
        result_data['reconstructed'] = np.concatenate(reconstructed_images, axis=0)
    
    return Result.ok(result_data)
```

**Sorties**:
- `reconstruction_errors`: Erreur moyenne par image (scalaire)
- `predictions`: Prédictions binaires (0=normal, 1=anomaly)
- `threshold`: Seuil utilisé (95ème percentile)
- `error_maps`: Cartes d'erreur brutes (H, W)
- `heatmaps`: Cartes d'erreur normalisées [0, 1] pour visualisation

---

### **Prédiction Classifier** (`_predict_classifier`)

```python
def _predict_classifier(self, test_loader) -> Result:
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = self.model(data)
            
            # Softmax pour probabilités
            probs = torch.softmax(output, dim=1).cpu().numpy()  # (B, num_classes)
            preds = output.argmax(dim=1).cpu().numpy()  # (B,) classes prédites
            
            all_probs.append(probs)
            all_preds.extend(preds)
    
    return Result.ok({
        'probabilities': np.concatenate(all_probs, axis=0),  # (N, num_classes)
        'predictions': np.array(all_preds)  # (N,) classes prédites
    })
```

**Sorties**:
- `probabilities`: Probabilités par classe (N, num_classes)
- `predictions`: Classes prédites (N,)

---

## 📊 Pipeline d'Évaluation

### **Méthode**: `evaluate(X_test, y_test) -> Result`

#### **Flux Complet**:

```python
def evaluate(self, X_test, y_test) -> Result:
    """
    Pipeline d'évaluation:
    
    1. VALIDATION TEST SET
    2. PREPROCESSING (transform uniquement!)
    3. DÉTECTION TYPE MODÈLE
    4. ÉVALUATION (classifier ou autoencoder)
    5. RETOUR MÉTRIQUES COMPLÈTES
    """
```

---

#### **Étape 1: Validation Test Set**

```python
# Ligne 1189
test_val = DataValidator.validate_input_data(X_test, y_test, "test")
if not test_val.success:
    return test_val
```

---

#### **Étape 2: Preprocessing**

```python
# Ligne 1197
X_test_norm = self.preprocessor.transform(X_test, output_format="channels_first")
```

**⚠️ CRITIQUE**: `transform()` uniquement, **JAMAIS** `fit()` sur test set!

---

#### **Étape 3: Détection Type Modèle**

```python
# Lignes 1200-1202
is_autoencoder = self.model_config.model_type in [
    ModelType.CONV_AUTOENCODER, ModelType.VAE, ModelType.DENOISING_AE
]

if is_autoencoder:
    return self._evaluate_autoencoder(X_test_norm, y_test)
else:
    return self._evaluate_classifier(X_test_norm, y_test)
```

---

### **Évaluation Classifier** (`_evaluate_classifier`)

```python
def _evaluate_classifier(self, X_test_norm, y_test) -> Result:
    # Création DataLoader
    test_loader = DataLoaderFactory.create(...)
    
    # Prédictions
    self.model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = self.model(data)
            
            probs = torch.softmax(output, dim=1).cpu().numpy()
            preds = output.argmax(dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(target.numpy())
    
    # Calcul métriques
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds, average='weighted'),
        'recall': recall_score(all_targets, all_preds, average='weighted'),
        'f1': f1_score(all_targets, all_preds, average='weighted'),
        'confusion_matrix': confusion_matrix(all_targets, all_preds).tolist(),
        'n_samples': len(X_test_norm),
        'n_classes': len(np.unique(y_test))
    }
    
    # AUC-ROC si binaire
    if self.model_config.num_classes == 2:
        metrics['auc_roc'] = roc_auc_score(all_targets, all_probs[:, 1])
    
    # Classification report
    metrics['classification_report'] = classification_report(
        all_targets, all_preds, output_dict=True
    )
    
    return Result.ok(metrics)
```

**Métriques Retournées**:
- `accuracy`: Précision globale
- `precision`: Précision moyenne pondérée
- `recall`: Rappel moyen pondéré
- `f1`: F1-score moyen pondéré
- `confusion_matrix`: Matrice de confusion (list)
- `auc_roc`: AUC-ROC (si binaire)
- `classification_report`: Rapport détaillé par classe
- `n_samples`, `n_classes`: Métadonnées

---

### **Évaluation Autoencoder** (`_evaluate_autoencoder`)

```python
def _evaluate_autoencoder(self, X_test_norm, y_test) -> Result:
    # Création DataLoader
    test_loader = DataLoaderFactory.create(...)
    
    # Calcul erreurs de reconstruction
    self.model.eval()
    reconstruction_errors = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            reconstructed = self.model(data)
            
            errors = torch.mean(
                (data - reconstructed) ** 2,
                dim=tuple(range(1, data.ndim))
            ).cpu().numpy()
            
            reconstruction_errors.extend(errors)
            all_targets.extend(target.numpy())
    
    reconstruction_errors = np.array(reconstruction_errors)
    all_targets = np.array(all_targets)
    
    # === CALCUL SEUIL OPTIMAL (sur test set) ===
    # Méthode 1: Percentile 95 des erreurs normales
    normal_errors = reconstruction_errors[all_targets == 0]
    if len(normal_errors) > 0:
        threshold = np.percentile(normal_errors, 95)
    else:
        threshold = np.percentile(reconstruction_errors, 95)
    
    # Prédictions binaires
    predictions = (reconstruction_errors > threshold).astype(int)
    
    # === MÉTRIQUES ===
    metrics = {
        'accuracy': accuracy_score(all_targets, predictions),
        'precision': precision_score(all_targets, predictions, zero_division=0),
        'recall': recall_score(all_targets, predictions, zero_division=0),
        'f1': f1_score(all_targets, predictions, zero_division=0),
        'confusion_matrix': confusion_matrix(all_targets, predictions).tolist(),
        'auc_roc': roc_auc_score(all_targets, reconstruction_errors),
        'threshold': float(threshold),
        'reconstruction_errors': reconstruction_errors.tolist(),
        'n_samples': len(X_test_norm)
    }
    
    return Result.ok(metrics)
```

**Métriques Retournées**:
- `accuracy`, `precision`, `recall`, `f1`: Métriques binaires
- `confusion_matrix`: Matrice de confusion
- `auc_roc`: AUC-ROC (utilise `reconstruction_errors` comme scores)
- `threshold`: Seuil utilisé pour binarisation
- `reconstruction_errors`: Liste des erreurs (pour analyse)
- `n_samples`: Nombre d'échantillons

---

## 🔄 Flux Complet de Bout en Bout

### **1. Chargement des Données (Home Page)**

```
ui/home.py
  └─> load_images_flexible(data_dir)
      └─> Retourne: X, X_norm, y, y_train
      
  └─> STATE.set_images(X, X_norm, y, dir_path, structure, info, y_train=y_train)
      └─> Détection automatique tâche (supervised/unsupervised)
      └─> Stockage dans STATE.data
```

---

### **2. Lancement Training (Page Training)**

```
src/app/pages/4_training_computer.py
  └─> handle_training_success(training_result)
      └─> Création TrainingContext:
          • X_train, y_train, X_val, y_val
          • model_config (dict)
          • training_config (dict)
          • preprocessing_config
      
      └─> training_orchestrator.train(context)
          │
          ├─> 1. Validation configs
          ├─> 2. Démarrage MLflow run
          ├─> 3. Création trainer (ComputerVisionTrainer ou AnomalyAwareTrainer)
          ├─> 4. trainer.fit(X_train, y_train, X_val, y_val)
          │     │
          │     ├─> 1. Validation données
          │     ├─> 2. Setup preprocessing (fit sur train, transform sur val)
          │     ├─> 3. Construction modèle
          │     ├─> 4. Setup optimizer/scheduler/criterion
          │     ├─> 5. Boucle entraînement (epochs)
          │     │     ├─> Phase train
          │     │     ├─> Phase validation
          │     │     ├─> Early stopping check
          │     │     └─> Checkpointing
          │     └─> 6. Retour Result avec model, history, preprocessor
          │
          ├─> 5. Log métriques MLflow
          ├─> 6. Log artifacts MLflow (modèle, preprocessor, config)
          └─> 7. Retour TrainingResult
      
      └─> Stockage dans STATE.training_results:
          • model
          • history
          • preprocessor
          • model_config
          • mlflow_run_id
```

---

### **3. Évaluation (Page Evaluation)**

```
src/app/pages/5_anomaly_evaluation.py
  └─> Récupération depuis STATE.training_results:
      • model = STATE.training_results["model"]
      • history = STATE.training_results["history"]
      • preprocessor = STATE.training_results["preprocessor"]
      • model_config = STATE.training_results["model_config"]
  
  └─> Création trainer (si nécessaire):
      trainer = ComputerVisionTrainer(model_config, training_config)
      trainer.model = model
      trainer.preprocessor = preprocessor
  
  └─> Prédictions sur X_test:
      pred_result = trainer.predict(X_test, return_reconstructed=True, return_localization=True)
      
      Retourne:
      • Pour autoencoder:
        - reconstruction_errors
        - predictions (binaires)
        - error_maps (spatiales)
        - heatmaps (normalisées)
        - binary_masks (si return_localization=True)
      • Pour classifier:
        - probabilities
        - predictions (classes)
  
  └─> Évaluation complète:
      eval_result = trainer.evaluate(X_test, y_test)
      
      Retourne:
      • Métriques (accuracy, precision, recall, f1, auc_roc)
      • Confusion matrix
      • Classification report
  
  └─> Visualisations:
      • Courbes d'entraînement (loss, accuracy)
      • Matrice de confusion
      • Heatmaps d'anomalies
      • ROC curve, Precision-Recall curve
      • Analyse erreurs (false positives, false negatives)
```

---

### **Schéma Visuel du Flux Complet**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           1. HOME PAGE                                  │
│  Upload images → load_images_flexible() → STATE.set_images()           │
│  Détection automatique: supervised/unsupervised                         │
└────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        2. TRAINING PAGE                                 │
│  Configuration → TrainingContext → training_orchestrator.train()       │
│                                                                         │
│  ORCHESTRATEUR:                                                         │
│    ├─ Validation configs                                               │
│    ├─ MLflow run start                                                 │
│    ├─ Création trainer                                                 │
│    │                                                                    │
│    └─ TRAINER.fit():                                                   │
│         ├─ Preprocessing (fit sur train, transform sur val)            │
│         ├─ Construction modèle                                         │
│         ├─ Setup optimizer/scheduler/criterion                         │
│         └─ Boucle entraînement (epochs)                                │
│              ├─ Train epoch                                            │
│              ├─ Validate epoch                                         │
│              ├─ Early stopping                                         │
│              └─ Checkpointing                                          │
│                                                                         │
│    ├─ Log MLflow (métriques, artifacts)                                │
│    └─ Retour TrainingResult → STATE.training_results                   │
└────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        3. EVALUATION PAGE                               │
│  Récupération depuis STATE.training_results                            │
│                                                                         │
│  PRÉDICTIONS:                                                           │
│    trainer.predict(X_test)                                             │
│      ├─ Preprocessing (transform uniquement!)                          │
│      ├─ Forward pass (model.eval())                                    │
│      └─ Retour: predictions, probabilities, error_maps, heatmaps       │
│                                                                         │
│  ÉVALUATION:                                                            │
│    trainer.evaluate(X_test, y_test)                                    │
│      ├─ Prédictions                                                    │
│      ├─ Calcul métriques (accuracy, precision, recall, f1, auc_roc)   │
│      └─ Retour: métriques complètes                                    │
│                                                                         │
│  VISUALISATIONS:                                                        │
│    • Courbes d'entraînement                                            │
│    • Matrice de confusion                                              │
│    • Heatmaps d'anomalies                                              │
│    • ROC/PR curves                                                     │
│    • Analyse erreurs                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Points Critiques

### **1. Pas de Fuite de Données**

✅ **Preprocessing**:
- `fit_transform()` sur `X_train` uniquement
- `transform()` sur `X_val` et `X_test`
- Le preprocessor est fitté une seule fois sur le train

✅ **Early Stopping**:
- Basé sur `val_loss` ou `val_f1` (pas sur test)

✅ **Checkpointing**:
- Meilleur modèle basé sur validation uniquement

---

### **2. Gestion Formats**

✅ **Channels First/Last**:
- Auto-détection dans `DataPreprocessor`
- Conversion automatique vers `channels_first` pour PyTorch
- Gestion cohérente dans tout le pipeline

✅ **Tensor ↔ NumPy**:
- Conversion explicite avec `.cpu().numpy()`
- Gestion device (CPU/GPU) avec `DeviceManager`

---

### **3. Robustesse**

✅ **Gestion Erreurs**:
- Try/catch à chaque étape critique
- Retour `Result.err()` avec message explicite
- Logging détaillé pour debugging

✅ **Validation**:
- Validation données avant processing
- Validation configs (types, ranges)
- Validation résultats (non-None, shapes cohérentes)

✅ **Fallbacks**:
- Preprocessor fallback si manquant
- Historique minimal si erreur
- Default configs si invalides

---

## 📝 Résumé des Méthodes Clés

| Méthode | Classe | Entrée | Sortie | Rôle |
|---------|--------|--------|--------|------|
| `train()` | Orchestrateur | `TrainingContext` | `TrainingResult` | Orchestration complète |
| `fit()` | `ComputerVisionTrainer` | `X_train, y_train, X_val, y_val` | `Result` | Entraînement modèle |
| `predict()` | `ComputerVisionTrainer` | `X` | `Result` | Prédictions |
| `evaluate()` | `ComputerVisionTrainer` | `X_test, y_test` | `Result` | Métriques complètes |
| `_training_loop()` | `ComputerVisionTrainer` | `train_loader, val_loader` | `Result` | Boucle epochs |
| `_predict_autoencoder()` | `ComputerVisionTrainer` | `test_loader` | `Result` | Prédictions autoencoder |
| `_predict_classifier()` | `ComputerVisionTrainer` | `test_loader` | `Result` | Prédictions classifier |
| `_evaluate_autoencoder()` | `ComputerVisionTrainer` | `X_test_norm, y_test` | `Result` | Évaluation autoencoder |
| `_evaluate_classifier()` | `ComputerVisionTrainer` | `X_test_norm, y_test` | `Result` | Évaluation classifier |

---

## ✅ Conclusion

Cette architecture garantit:
- ✅ **Séparation des responsabilités**: Orchestrateur ≠ Trainer
- ✅ **Pas de fuite de données**: Preprocessing strict
- ✅ **Robustesse**: Validation et gestion erreurs complètes
- ✅ **Observabilité**: Logging et MLflow intégrés
- ✅ **Flexibilité**: Support classification et anomaly detection
- ✅ **Production-ready**: Gestion configs, checkpointing, early stopping

**Le système est prêt pour la production !** 🚀


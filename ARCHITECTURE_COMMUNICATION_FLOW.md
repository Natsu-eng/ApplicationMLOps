# 🔗 Architecture & Communication Flow - DataLab Pro

## ✅ Vérification Complète de la Communication Inter-Modules

### 📊 Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FLUX COMPLET                                │
└─────────────────────────────────────────────────────────────────────┘

    📄 Page 4 (Training)          →    🎯 Orchestrateur         →    ⚙️ Training Bas Niveau
    ───────────────────                ──────────────────            ───────────────────
    • UI Streamlit                   • Coordination                  • ComputerVisionTrainer
    • Configuration                  • MLflow                        • AnomalyAwareTrainer  
    • Validation                     • Preprocessing                 • Training logique
    • Callbacks                      • Gestion erreurs               • Métriques
                                      • Normalisation résultats

    ⬇️ Sauvegarde dans STATE.training_results
    ⬇️
    📄 Page 5 (Evaluation)           →    🔧 Helpers                →    📈 Visualisation
    ───────────────────                  ────────────                  ───────────────
    • Récupération résultats          • Prédictions                   • Dashboard Premium
    • Calcul métriques                • Analyse erreurs               • Graphiques
    • Génération rapports             • Recommandations               • Export
```

---

## 🔄 FLUX 1 : Page Training → Orchestrateur → Training Bas Niveau

### **Page 4 : `src/app/pages/4_training_computer.py`**

#### 1. Préparation des Données
```python
# Ligne 243-251 : Split des données
STATE.data.X_train = split_result["X_train"]
STATE.data.X_val = split_result["X_val"]
STATE.data.X_test = split_result["X_test"]
STATE.data.y_train = split_result["y_train"]
STATE.data.y_val = split_result["y_val"]
STATE.data.y_test = split_result["y_test"]
```

#### 2. Configuration du Modèle
```python
# Ligne 563-566 : Sélection modèle
STATE.model_config = {
    "model_type": model["id"],
    "model_params": self.get_default_model_params(model["id"])
}
```

#### 3. Création du Contexte et Appel Orchestrateur
```python
# Ligne 1141-1158 : train_with_metier_logic()
context = TrainingContext(
    X_train=STATE.data.X_train,
    y_train=STATE.data.y_train,
    X_val=STATE.data.X_val,
    y_val=STATE.data.y_val,
    model_config=STATE.model_config,
    training_config=STATE.training_config,
    preprocessing_config=STATE.preprocessing_config,
    callbacks=self._create_callbacks(streamlit_components),
    anomaly_type=anomaly_type,
    metadata={...}
)

# ✅ DÉLÉGATION À L'ORCHESTRATEUR
result = training_orchestrator.train(context)
```

#### 4. Sauvegarde des Résultats
```python
# Ligne 1195-1204 : handle_training_success()
STATE.training_results = {
    "model": model,
    "history": history,
    "training_config": getattr(STATE, 'training_config', {}),
    "model_config": getattr(STATE, 'model_config', {}),  # ✅ Sauvegardé
    "preprocessing_config": getattr(STATE, 'preprocessing_config', {}),
    "imbalance_config": getattr(STATE, 'imbalance_config', {}),
    "preprocessor": preprocessor,  # ✅ Sauvegardé
    "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
}
```

---

### **Orchestrateur : `orchestrators/visio_training_orchestrator.py`**

#### 1. Validation et Conversion des Configs
```python
# Ligne 83 : Conversion robuste TrainingConfig
context.training_config = self._ensure_training_config_object(context.training_config)

# Ligne 88 : Validation données
self._validate_training_context(context)
```

#### 2. Création du Trainer
```python
# Ligne 362-389 : _create_trainer()
if context.anomaly_type:
    return AnomalyAwareTrainer(
        anomaly_type=context.anomaly_type,
        model_config=model_config,
        training_config=context.training_config,
        ...
    )
else:
    return ComputerVisionTrainer(
        model_config=model_config,
        training_config=context.training_config,
        ...
    )
```

#### 3. Exécution de l'Entraînement
```python
# Ligne 391-537 : _execute_training()
# Appelle trainer.fit() ou trainer.train()
# Normalise le résultat en dict standardisé
result = {
    'success': bool,
    'data': {
        'history': {...}
    },
    'error': None | str,
    'metadata': {...}
}
```

#### 4. Retour du Résultat
```python
# Ligne 139-150 : Retour TrainingResult
return TrainingResult(
    success=True,
    model=trainer.model,
    history=final_history,
    preprocessor=preprocessor,
    mlflow_run_id=run_id,
    metadata={...}
)
```

---

### **Training Bas Niveau : `src/models/computer_vision_training.py`**

#### Classes Principales
- **`ComputerVisionTrainer`** : Entraînement supervisé standard
- **`AnomalyAwareTrainer`** : Entraînement avec awareness des types d'anomalies

#### Méthodes Clés
- `fit(X_train, y_train, X_val, y_val)` → Retourne Result avec history
- `train(...)` → Alternative pour AnomalyAwareTrainer
- Le trainer gère automatiquement le preprocessing via `DataPreprocessor`

---

## 🔄 FLUX 2 : Page Evaluation → Helpers → Visualisation

### **Page 5 : `src/app/pages/5_anomaly_evaluation.py`**

#### 1. Vérifications Initiales
```python
# Ligne 102-117 : Validations
✅ STATE.training_results existe
✅ STATE.training_results est un dict
✅ STATE.training_results["model"] existe
```

#### 2. Récupération des Données
```python
# Ligne 121-138 : Récupération avec fallbacks
model = STATE.training_results["model"]
history = safe_convert_history(STATE.training_results.get("history", {}))

# ✅ FALLBACK ROBUSTE pour model_config
if not hasattr(STATE, 'model_config') or STATE.model_config is None:
    if "model_config" in STATE.training_results:
        STATE.model_config = STATE.training_results["model_config"]

model_type = STATE.model_config.get("model_type", "autoencoder")
preprocessor = STATE.training_results.get("preprocessor")

# Données test depuis STATE.data
X_test = STATE.data.X_test
y_test = STATE.data.y_test
```

#### 3. Prédictions via Helper
```python
# Ligne 219-224 : Prédictions centralisées
prediction_results = robust_predict_with_preprocessor(
    model, X_test, preprocessor, model_type,
    return_localization=True, STATE=STATE
)
```

---

## ✅ Points de Vérification

### **1. Communication Page 4 → Orchestrateur**
- ✅ **TrainingContext** correctement construit avec toutes les données STATE
- ✅ **Callbacks** créés et passés au contexte
- ✅ **Résultat** récupéré depuis `training_orchestrator.train()`
- ✅ **Preprocessor** sauvegardé dans `STATE.preprocessor` ET `training_results`

### **2. Communication Orchestrateur → Training Bas Niveau**
- ✅ **Trainer** créé selon le type (ComputerVisionTrainer ou AnomalyAwareTrainer)
- ✅ **TrainingConfig** converti robustement (dict → objet)
- ✅ **Résultat** normalisé en dict standardisé
- ✅ **Preprocessor** récupéré depuis le trainer

### **3. Communication Page 5 → Résultats**
- ✅ **Modèle** récupéré depuis `STATE.training_results["model"]`
- ✅ **History** récupéré et normalisé avec `safe_convert_history()`
- ✅ **Model_config** récupéré avec fallback robuste (STATE → training_results)
- ✅ **Preprocessor** récupéré depuis `training_results`
- ✅ **Données test** récupérées depuis `STATE.data.X_test/y_test`

### **4. Helpers Centralisés**
- ✅ **`helpers/anomaly_prediction_helpers.py`** : `robust_predict_with_preprocessor()`
- ✅ **`helpers/ui_components/anomaly_evaluation.py`** : Toutes les fonctions d'analyse
- ✅ **`ui/anomaly_evaluation_styles.py`** : CSS centralisé

---

## 🎯 Structure des Données dans STATE

### **Après Training (Page 4)**
```python
STATE = {
    # Données
    data: {
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,  # ✅ Disponible pour évaluation
        ...
    },
    
    # Configuration
    model_config: {...},  # ✅ Défini lors sélection modèle
    training_config: {...},
    preprocessing_config: {...},
    
    # Résultats
    training_results: {  # ✅ Dict complet sauvegardé
        "model": <PyTorch Model>,
        "history": {...},
        "model_config": {...},  # ✅ Dupliqué pour sécurité
        "training_config": {...},
        "preprocessing_config": {...},
        "preprocessor": <DataPreprocessor>,  # ✅ Sauvegardé
        "trained_at": "2025-..."
    },
    
    # Références directes
    trained_model: <PyTorch Model>,
    training_history: {...},
    preprocessor: <DataPreprocessor>
}
```

---

## ✅ Validation Finale

### **Tous les Flux sont Corrects :**

1. ✅ **Page 4 → Orchestrateur** : TrainingContext correctement passé
2. ✅ **Orchestrateur → Training** : Trainer créé et entraîné
3. ✅ **Training → Orchestrateur** : Résultat normalisé en dict
4. ✅ **Orchestrateur → Page 4** : TrainingResult avec tous les éléments
5. ✅ **Page 4 → STATE** : `training_results` complet sauvegardé
6. ✅ **Page 5 → STATE** : Toutes les données récupérées avec fallbacks
7. ✅ **Page 5 → Helpers** : Prédictions et analyses via modules centralisés

---

## 🔧 Points d'Amélioration Réalisés

1. ✅ **Fallback robuste** pour `model_config` dans Page 5
2. ✅ **Normalisation** des résultats dans l'orchestrateur
3. ✅ **Validation** exhaustive des données à chaque étape
4. ✅ **Centralisation** des helpers pour réutilisabilité
5. ✅ **Gestion d'erreurs** complète avec messages clairs

---

## 📝 Conclusion

**Tout est correctement connecté !** 🎉

- ✅ Les pages communiquent bien avec l'orchestrateur
- ✅ L'orchestrateur délègue correctement au training bas niveau
- ✅ Les résultats sont sauvegardés de manière robuste dans STATE
- ✅ La page d'évaluation récupère tout avec des fallbacks sécurisés
- ✅ Les helpers sont centralisés et réutilisables

**Architecture production-ready !** 🚀


# 🔍 AUDIT PAGES STREAMLIT - Computer Vision

## 📊 OBJECTIF
Vérifier la cohérence des pages `4_training_computer.py` et `5_anomaly_evaluation.py` avec l'orchestrateur et le pipeline complet jusqu'au niveau le plus bas.

---

## ✅ POINTS CONFORMES

### 1. **Intégration avec l'Orchestrateur** ✅
- ✅ Page `4_training_computer.py` utilise correctement `training_orchestrator.train(context)`
- ✅ `TrainingContext` est correctement construit avec toutes les données nécessaires
- ✅ Propagation de `split_config` dans les métadonnées du contexte
- ✅ Résultats correctement stockés dans `STATE.training_results`

### 2. **Propagation des Données** ✅
- ✅ `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test` correctement stockés dans `STATE.data`
- ✅ `model_config`, `training_config`, `preprocessing_config` propagés
- ✅ `preprocessor` récupéré depuis l'orchestrateur et stocké

### 3. **Récupération dans Page Évaluation** ✅
- ✅ Page `5_anomaly_evaluation.py` récupère correctement `model`, `history`, `preprocessor` depuis `STATE.training_results`
- ✅ Fallback pour `model_config` depuis `training_results` si absent dans `STATE.model_config`
- ✅ Validation robuste des données avant utilisation

### 4. **Gestion des Configurations** ✅
- ✅ `split_config` propagé via `context_metadata` puis ajouté directement au `TrainingContext`
- ✅ Mode détecté et stocké dans `split_config`
- ✅ Métadonnées complètes sauvegardées dans `training_results`

---

## ⚠️ PROBLÈMES IDENTIFIÉS

### ❌ P0 - Blocage Mode Non Supervisé dans Étape 2

**Fichier**: `src/app/pages/4_training_computer.py`  
**Ligne**: 279

**Problème**:
```python
if not STATE.loaded or STATE.data.y_train is None:
    st.error("❌ Données d'entraînement manquantes")
    ...
```

**Impact**: 
- En mode non supervisé, `y_train` peut être `None` ou un array vide
- Cette vérification bloque l'accès à l'étape 2 pour le mode non supervisé
- L'utilisateur ne peut pas continuer le workflow

**Correction proposée**:
```python
# Vérifier si données chargées ET (y_train présent OU mode non supervisé)
if not STATE.loaded:
    st.error("❌ Aucun dataset chargé")
    ...
    
# Récupération mode depuis split_config
split_config = getattr(STATE.data, 'split_config', {})
mode = split_config.get('mode', 'supervised')

# Vérification conditionnelle selon mode
if mode == "supervised" and (not hasattr(STATE.data, 'y_train') or STATE.data.y_train is None):
    st.error("❌ Données d'entraînement manquantes (mode supervisé)")
    ...
elif mode == "unsupervised" and not hasattr(STATE.data, 'X_train'):
    st.error("❌ Données d'entraînement manquantes (mode non supervisé)")
    ...
```

---

### ⚠️ P1 - Affichage Badge Mode dans Header

**Fichier**: `src/app/pages/4_training_computer.py`  
**Ligne**: 88

**Problème**:
```python
if STATE.loaded and STATE.data.y is not None:
    mode, _ = detect_training_mode(STATE.data.y)
    ...
```

**Impact**:
- En mode non supervisé, `STATE.data.y` peut être `None`
- Le badge de mode n'est pas affiché pour le mode non supervisé
- L'utilisateur ne voit pas clairement le mode détecté

**Correction proposée**:
```python
# Détection mode avec fallback
mode = None
if STATE.loaded:
    # Priorité 1: split_config (après split)
    if hasattr(STATE.data, 'split_config') and STATE.data.split_config:
        mode = STATE.data.split_config.get('mode')
    
    # Priorité 2: y_train (si splitté)
    if mode is None and hasattr(STATE.data, 'y_train') and STATE.data.y_train is not None:
        try:
            mode, _ = detect_training_mode(STATE.data.y_train)
        except ValueError:
            pass
    
    # Priorité 3: y global (avant split)
    if mode is None and STATE.data.y is not None:
        try:
            mode, _ = detect_training_mode(STATE.data.y)
        except ValueError:
            mode = "unsupervised"  # Fallback si y=None
    
    # Fallback final
    if mode is None:
        mode = "unsupervised"  # Par défaut si aucune info
    
    # Affichage badge
    badge_color = "#4facfe" if mode == "supervised" else "#f5576c"
    st.markdown(
        f"<div style='background:{badge_color};color:white;padding:0.3rem;border-radius:5px;text-align:center;font-size:0.7rem;'>"
        f"{'🎯 SUPERVISÉ' if mode == 'supervised' else '🔍 ANOMALIES'}"
        f"</div>",
        unsafe_allow_html=True
    )
```

---

### ⚠️ P1 - Calcul Class Weights sans Vérification

**Fichier**: `src/app/pages/4_training_computer.py`  
**Ligne**: 320

**Problème**:
```python
if use_weights:
    classes = np.unique(y_train)  # ❌ Peut échouer si y_train est None
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    ...
```

**Impact**:
- Si `y_train` est `None` (mode non supervisé), `np.unique(y_train)` échoue
- Même si le code est dans un bloc `if mode == "supervised"`, une erreur de logique pourrait permettre l'exécution

**Correction proposée**:
```python
if use_weights and y_train is not None:
    try:
        classes = np.unique(y_train)
        if len(classes) > 0:
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
            
            st.info("**Poids calculés:**")
            for cls, weight in weight_dict.items():
                st.write(f"- Classe {cls}: `{weight:.3f}`")
            
            STATE.class_weights = weight_dict
        else:
            st.warning("⚠️ Aucune classe détectée dans y_train")
    except Exception as e:
        logger.warning(f"Erreur calcul class weights: {e}")
        st.warning("⚠️ Impossible de calculer les poids de classe")
```

---

### ⚠️ P1 - Validation y_test dans Page Évaluation

**Fichier**: `src/app/pages/5_anomaly_evaluation.py`  
**Ligne**: 148

**Problème**:
```python
if not hasattr(STATE.data, 'y_test') or STATE.data.y_test is None:
    st.error("❌ Labels de test (y_test) manquants")
    st.stop()
```

**Impact**:
- En mode non supervisé, `y_test` peut être `None` ou un array vide
- La page d'évaluation bloque même si on peut évaluer un autoencoder sans labels (reconstruction error)
- L'utilisateur ne peut pas voir les résultats d'entraînement

**Correction proposée**:
```python
# Récupération mode depuis model_config ou training_results
model_type = STATE.model_config.get("model_type", "autoencoder") if isinstance(STATE.model_config, dict) else getattr(STATE.model_config, "model_type", "autoencoder")
is_autoencoder = model_type in ["autoencoder", "conv_autoencoder", "variational_autoencoder", "denoising_autoencoder"]

# Vérification conditionnelle selon type de modèle
if not hasattr(STATE.data, 'y_test') or STATE.data.y_test is None:
    if is_autoencoder:
        # Mode non supervisé: y_test peut être None, on utilisera reconstruction error
        logger.info("⚠️ y_test manquant, évaluation basée sur reconstruction error uniquement")
        y_test = None  # ou np.zeros(len(X_test)) pour compatibilité
    else:
        # Mode supervisé: y_test obligatoire
        st.error("❌ Labels de test (y_test) manquants (mode supervisé)")
        st.stop()
else:
    y_test = STATE.data.y_test
```

---

### ⚠️ P2 - Gestion y_val dans Validation

**Fichier**: `src/app/pages/4_training_computer.py`  
**Ligne**: 1304-1306

**Observation**:
```python
context = TrainingContext(
    X_train=STATE.data.X_train,
    y_train=STATE.data.y_train,  # ✅ Peut être None
    X_val=STATE.data.X_val, 
    y_val=STATE.data.y_val,  # ✅ Peut être None
    ...
)
```

**État**: ✅ **CORRECT**
- L'orchestrateur et le trainer gèrent déjà `y_train=None` et `y_val=None`
- Pas de correction nécessaire

---

## 📋 CORRECTIONS À APPLIQUER

### Priorité P0 (Bloquant)
1. ✅ Corriger vérification `y_train` dans étape 2 (ligne 279)

### Priorité P1 (Important)
2. ✅ Améliorer affichage badge mode dans header (ligne 88)
3. ✅ Ajouter vérification avant calcul class weights (ligne 320)
4. ✅ Adapter validation `y_test` dans page évaluation (ligne 148)

---

## ✅ CORRECTIONS APPLIQUÉES

### ✅ 1. Vérification y_train dans Étape 2 - CORRIGÉ
- ✅ Vérification conditionnelle selon mode (supervisé vs non supervisé)
- ✅ Mode non supervisé: vérifie uniquement `X_train`
- ✅ Mode supervisé: vérifie `y_train`

### ✅ 2. Affichage Badge Mode dans Header - CORRIGÉ
- ✅ Détection mode avec fallbacks multiples (split_config → y_train → y global)
- ✅ Badge affiché même en mode non supervisé
- ✅ Fallback vers "unsupervised" si aucune info disponible

### ✅ 3. Calcul Class Weights - CORRIGÉ
- ✅ Vérification `y_train is not None` avant calcul
- ✅ Gestion d'erreurs avec try/except
- ✅ Warning si aucune classe détectée

### ✅ 4. Validation y_test dans Page Évaluation - CORRIGÉ
- ✅ Vérification conditionnelle selon type de modèle (autoencoder vs classification)
- ✅ Mode non supervisé: `y_test` peut être `None`, évaluation basée sur reconstruction error
- ✅ Mode supervisé: `y_test` obligatoire

### ✅ 5. Calcul Métriques avec y_test=None - CORRIGÉ
- ✅ Toutes les métriques de classification conditionnées à `y_test is not None`
- ✅ Métriques de reconstruction toujours calculées pour autoencoders
- ✅ Sections UI adaptées (courbes ROC/PR, distributions) selon présence de labels

---

## ✅ CONCLUSION FINALE

**Cohérence globale**: ✅ **EXCELLENTE** - Toutes les corrections appliquées

**Points forts**:
- ✅ Intégration orchestrateur correcte
- ✅ Propagation données complète
- ✅ Récupération résultats robuste
- ✅ Gestion configurations cohérente
- ✅ Support complet mode non supervisé dans UI
- ✅ Validation conditionnelle selon mode

**Corrections appliquées**:
- ✅ Gestion mode non supervisé dans UI (4 points)
- ✅ Validation conditionnelle selon mode (1 point)
- ✅ Calcul métriques adaptatif (1 point)

**Impact**: 
- ✅ Support complet du mode non supervisé dans l'interface utilisateur
- ✅ Aligné avec le pipeline backend
- ✅ Expérience utilisateur cohérente pour tous les modes
- ✅ Aucun blocage pour les workflows non supervisés

**Tous les linters passent sans erreur** ✅


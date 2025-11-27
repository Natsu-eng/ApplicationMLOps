# 🗺️ Guide de Navigation - DataLab Pro

## 📋 Vue d'Ensemble

Ce guide documente le système de navigation entre pages dans DataLab Pro.

---

## 🎯 Règle d'Or

**TOUJOURS utiliser `st.rerun()` après `STATE.switch()`**

```python
# ❌ FAUX (ne marchera pas)
if st.button("Go to Training"):
    STATE.switch(AppPage.ML_TRAINING)

# ✅ CORRECT
if st.button("Go to Training"):
    if STATE.switch(AppPage.ML_TRAINING):
        st.rerun()
    else:
        st.error("Navigation échouée")

# 🌟 MEILLEUR (avec helper)
if st.button("Go to Training"):
    navigate_to(AppPage.ML_TRAINING)
```

---

## 🔀 Flux de Navigation

### **1. Dashboard → ML Training**

```python
from helpers.navigation_validator import navigate_to

# Reset workflow + navigation
if st.button("🎯 ML Training"):
    navigate_to(AppPage.ML_TRAINING, reset_workflow=True)
```

**Ce qui se passe :**
1. ✅ Validation : données tabulaires chargées ?
2. ✅ Reset : `STATE.current_step = 0`
3. ✅ Switch : `STATE.switch(AppPage.ML_TRAINING)`
4. ✅ Rerun : `st.rerun()` (forcé par `navigate_to`)

---

### **2. ML Training → ML Evaluation**

```python
# Dans 2_training.py après entraînement
if st.button("📈 Analyse Détaillée"):
    # 1. Sauvegarder résultats
    STATE.ml_results = result.results
    STATE.training_results = result
    
    # 2. Naviguer
    navigate_to(AppPage.ML_EVALUATION, reset_workflow=False)
```

**Important :**
- ❌ Ne PAS reset le workflow (on garde les résultats)
- ✅ Sauvegarder `STATE.ml_results` ET `STATE.training_results`

---

### **3. ML Evaluation → Dashboard**

```python
if st.button("📊 Retour Dashboard"):
    navigate_to(AppPage.DASHBOARD, reset_workflow=False)
```

---

## 🛡️ Validation de Navigation

### **Vérifier avant de naviguer**

```python
from helpers.navigation_validator import NavigationValidator

# Vérifier si ML Training accessible
can_access, reason = NavigationValidator.validate_ml_training_access()

if can_access:
    navigate_to(AppPage.ML_TRAINING)
else:
    st.error(f"❌ {reason}")
```

### **Obtenir un rapport complet**

```python
report = NavigationValidator.get_navigation_report()

# Afficher dans debug panel
st.json(report)
```

---

## 📊 État du Workflow

### **Workflow ML Training (6 étapes)**

```python
# État workflow
STATE.current_step       # 0-5 (étape courante)
STATE.workflow_complete  # True si terminé

# Reset avant nouvelle session
STATE.current_step = 0
STATE.workflow_complete = False
```

### **Données requises par étape**

| Étape | État Requis |
|-------|-------------|
| 1. Dataset | `STATE.tabular == True` |
| 2. Cible | `STATE.target_column != None` |
| 3. Déséquilibre | `STATE.imbalance_config` configuré |
| 4. Prétraitement | `STATE.preprocessing_config` configuré |
| 5. Modèles | `len(STATE.selected_models) > 0` |
| 6. Lancement | Tous les précédents validés |

---

## 🔍 Debug Navigation

### **Activer le panneau de debug**

```python
# Dans la sidebar
if st.sidebar.checkbox("🐛 Mode Debug"):
    NavigationValidator.render_debug_panel()
```

### **Affichage :**
- 📄 Page courante
- 📊 Type de données
- ✅/❌ Validations par page
- 🔄 État du workflow
- 📈 Résultats ML disponibles

---

## ⚠️ Pièges Courants

### **1. Oublier `st.rerun()`**

```python
# ❌ Symptôme : Bouton cliqué mais rien ne se passe
STATE.switch(AppPage.ML_TRAINING)  # Manque st.rerun()

# ✅ Solution
if STATE.switch(AppPage.ML_TRAINING):
    st.rerun()
```

### **2. Navigation sans validation**

```python
# ❌ Navigation directe sans vérifier
STATE.switch(AppPage.ML_EVALUATION)  # Peut échouer silencieusement

# ✅ Avec validation
can_access, reason = NavigationValidator.validate_ml_evaluation_access()
if can_access:
    navigate_to(AppPage.ML_EVALUATION)
else:
    st.error(reason)
```

### **3. Résultats ML non sauvegardés**

```python
# ❌ Naviguer sans sauvegarder
navigate_to(AppPage.ML_EVALUATION)  # Échec: pas de résultats

# ✅ Sauvegarder avant navigation
STATE.ml_results = result.results
STATE.training_results = result
navigate_to(AppPage.ML_EVALUATION)
```

---

## 🧪 Tester la Navigation

### **Lancer les tests**

```bash
python tests/test_navigation_flow.py
```

### **Tests inclus :**
1. ✅ État initial
2. ✅ Prérequis ML Training
3. ✅ Reset workflow
4. ✅ Pages autorisées
5. ✅ Rapport complet
6. ✅ Protection ML Evaluation

---

## 📚 Référence API

### **NavigationValidator**

```python
# Validation par page
validate_ml_training_access() -> (bool, str)
validate_ml_evaluation_access() -> (bool, str)
validate_cv_training_access() -> (bool, str)
validate_dashboard_access() -> (bool, str)

# Rapport complet
get_navigation_report() -> Dict

# Navigation sécurisée
safe_navigate(target_page, reset_workflow=True) -> bool

# Debug UI
render_debug_panel()
```

### **Helper navigate_to()**

```python
from helpers.navigation_validator import navigate_to

# Signature
navigate_to(page: AppPage, reset_workflow: bool = True) -> bool

# Exemples
navigate_to(AppPage.ML_TRAINING)  # Reset automatique
navigate_to(AppPage.ML_EVALUATION, reset_workflow=False)  # Sans reset
```

---

## 🔗 Pages et Autorisations

| Page | Prérequis | Autorisation |
|------|-----------|--------------|
| **HOME** | Aucun | Toujours |
| **DASHBOARD** | Données chargées | Si `STATE.loaded` |
| **ML_TRAINING** | Données tabulaires | Si `STATE.tabular` |
| **ML_EVALUATION** | Résultats ML | Si `STATE.training_results` |
| **CV_TRAINING** | Images | Si `STATE.images` |
| **ANOMALY_EVAL** | Images + résultats CV | Si résultats CV |

---

## 🎯 Bonnes Pratiques

### **1. Toujours valider avant navigation**

```python
can_access, reason = NavigationValidator.validate_ml_training_access()
if not can_access:
    st.warning(reason)
    return

navigate_to(AppPage.ML_TRAINING)
```

### **2. Utiliser le helper `navigate_to()`**

```python
# Au lieu de STATE.switch() + st.rerun()
navigate_to(AppPage.ML_TRAINING)
```

### **3. Gérer les erreurs**

```python
try:
    navigate_to(AppPage.ML_TRAINING)
except Exception as e:
    st.error(f"Erreur navigation: {e}")
    logger.error(f"Navigation failed: {e}", exc_info=True)
```

### **4. Logger les navigations**

```python
logger.info(f"Navigation: {STATE.page.value} → {target_page.value}")
navigate_to(target_page)
```

---

## 📖 Exemples Complets

### **Exemple 1: Dashboard vers Training**

```python
# pages/1_dashboard.py
from helpers.navigation_validator import navigate_to, NavigationValidator

# Vérifier accès
can_access, reason = NavigationValidator.validate_ml_training_access()

# Bouton avec validation
if st.button("🎯 ML Training", disabled=not can_access):
    navigate_to(AppPage.ML_TRAINING, reset_workflow=True)
```

### **Exemple 2: Training vers Evaluation**

```python
# pages/2_training.py (après entraînement)

# Sauvegarder résultats
STATE.ml_results = result.results
STATE.training_results = result
STATE.workflow_complete = True

# Naviguer
if st.button("📈 Voir les Résultats"):
    navigate_to(AppPage.ML_EVALUATION, reset_workflow=False)
```

### **Exemple 3: Evaluation vers Dashboard**

```python
# pages/3_evaluation.py

if st.button("📊 Retour Dashboard"):
    # Pas de reset nécessaire
    navigate_to(AppPage.DASHBOARD, reset_workflow=False)
```

---

## 🆘 Troubleshooting

### **Navigation ne fonctionne pas**

1. Vérifier les logs : `logger.info()` dans `state_managers.py`
2. Activer debug panel : Checkbox "🐛 Mode Debug"
3. Vérifier autorisations : `report['authorized_pages']`
4. Tester avec : `python tests/test_navigation_flow.py`

### **Page blanche après navigation**

- ✅ Vérifier que `st.rerun()` est appelé
- ✅ Vérifier que la page existe dans `pages/`
- ✅ Vérifier les imports dans la page cible

### **Workflow se reset involontairement**

- ❌ Utiliser `reset_workflow=False` si on veut garder l'état
- ✅ Exemple : `navigate_to(AppPage.ML_EVALUATION, reset_workflow=False)`

---

## 📞 Support

Pour toute question sur la navigation :
1. Consulter ce guide
2. Lancer les tests de navigation
3. Activer le mode debug
4. Vérifier les logs dans le terminal

---

**Version:** 1.0.0  
**Dernière mise à jour:** 2025-01-21
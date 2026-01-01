# ✅ Vérification Page Home - Upload Images & Exemples MVTec

## 📋 Structure de la Page Home

### **3 Onglets Principaux**

```
┌─────────────────────────────────────────────────────────────┐
│                    Page Home (ui/home.py)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📊 Tab 1: Données Tabulaires                               │
│     └─ Upload CSV, Excel, Parquet, JSON                     │
│                                                              │
│  🖼️ Tab 2: Données Images                                   │
│     ├─ 📁 Sous-tab 1: Dossier (Chemin)                      │
│     │   └─ Input texte pour chemin dossier                  │
│     │   └─ Support: MVTec AD, dossiers par classe, plat    │
│     │                                                       │
│     └─ 📤 Sous-tab 2: Fichiers Multiples                    │
│         └─ Upload multiple de fichiers images               │
│         └─ Création dossier temporaire                      │
│                                                              │
│  📦 Tab 3: Exemples MVTec                                   │
│     └─ Boutons pour datasets MVTec pré-configurés          │
│     └─ bottle, cable, capsule, metal_nut, etc.             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ Fonctionnalités Vérifiées

### **1. Upload via Chemin Dossier** (`_render_folder_upload`)

#### Fonctionnalités :
- ✅ Input texte pour chemin de dossier (Windows/Unix compatible)
- ✅ Validation de l'existence du dossier
- ✅ Détection automatique de la structure (MVTec, par classe, plat)
- ✅ Chargement avec `load_images_flexible()` qui retourne `y_train`
- ✅ Transmission `y_train` à `STATE.set_images()` pour détection unsupervised
- ✅ Détection automatique du mode (Unsupervised vs Supervised)
- ✅ Redirection vers Dashboard après chargement

#### Code clé :
```python
# Ligne 203
X, X_norm, y, y_train = load_images_flexible(data_dir, target_size=(256, 256))

# Ligne 213
if self.state.set_images(X, X_norm, y, data_dir, structure, info, y_train=y_train):
    # Mode détecté automatiquement
    mode_icon = "🔍" if y_train is not None and len(np.unique(y_train)) == 1 else "🎯"
```

---

### **2. Upload Fichiers Multiples** (`_render_multiple_files_upload`)

#### Fonctionnalités :
- ✅ Upload multiple de fichiers images (png, jpg, jpeg, bmp, tiff)
- ✅ Création d'un dossier temporaire
- ✅ Sauvegarde des fichiers uploadés
- ✅ Chargement comme dossier plat (structure "flat")
- ✅ Transmission `y_train` à STATE (même si None pour fichiers uploadés)
- ✅ Validation des images
- ✅ Redirection vers Dashboard

#### Code clé :
```python
# Ligne 254-261 : Création dossier temporaire
with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = Path(temp_dir)
    for uploaded_file in uploaded_files:
        file_path = temp_path / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

# Ligne 272 : Chargement
X, X_norm, y, y_train = load_images_flexible(data_dir, target_size=(256, 256))
```

---

### **3. Exemples MVTec** (`_render_mvtec_examples`)

#### Fonctionnalités :
- ✅ 9 datasets MVTec pré-configurés (bottle, cable, capsule, etc.)
- ✅ Boutons en grille 3x3
- ✅ Vérification de l'existence du dataset
- ✅ Chargement avec détection automatique de la structure
- ✅ **Transmission `y_train`** pour détection unsupervised correcte
- ✅ Message de confirmation avec mode détecté
- ✅ Redirection vers Dashboard

#### Datasets disponibles :
- 🍾 bottle (Bouteilles)
- 🔌 cable (Câbles)
- 💊 capsule (Capsules)
- 🔩 metal_nut (Écrous)
- 💊 pill (Pilules)
- 🔧 screw (Vis)
- 🪥 toothbrush (Brosses à dents)
- ⚡ transistor (Transistors)
- 🔗 zipper (Fermetures éclair)

#### Code clé :
```python
# Ligne 348
X, X_norm, y, y_train = load_images_flexible(path, target_size=(256, 256))

# Ligne 353
if self.state.set_images(X, X_norm, y, path, structure, info, y_train=y_train):
    # Vérification mode
    if y_train is not None and len(np.unique(y_train)) == 1:
        mode_msg = "🔍 Mode Unsupervised détecté (train = only normal)"
```

---

## 🔄 Flux de Données

### **Chargement → STATE**

```
┌─────────────────────────────────────────────────────────────┐
│                    CHARGEUR D'IMAGES                        │
│  load_images_flexible(data_dir, target_size=(256, 256))    │
├─────────────────────────────────────────────────────────────┤
│  Retourne:                                                  │
│  • X: Images brutes (uint8)                                 │
│  • X_norm: Images normalisées (float32)                     │
│  • y: Labels complets (train + test)                        │
│  • y_train: Labels du TRAIN uniquement (⚠️ CRITIQUE)        │
└─────────────────────────────────────────────────────────────┘
                           ⬇️
┌─────────────────────────────────────────────────────────────┐
│                  STATE MANAGER                              │
│  STATE.set_images(X, X_norm, y, dir_path, structure,       │
│                   info, y_train=y_train)                    │
├─────────────────────────────────────────────────────────────┤
│  Actions:                                                   │
│  1. Sauvegarde dans STATE.data                              │
│  2. Détection automatique de la tâche                       │
│     → detect_cv_task(y_train si disponible, sinon y)        │
│  3. Détermine: UNSUPERVISED / SUPERVISED                    │
│  4. Met à jour task_metadata                                │
└─────────────────────────────────────────────────────────────┘
                           ⬇️
┌─────────────────────────────────────────────────────────────┐
│                    REDIRECTION                              │
│  STATE.switch(AppPage.DASHBOARD)                            │
│  → st.rerun()                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ Points Critiques Validés

### **1. Transmission `y_train` ✅**
- ✅ Toutes les méthodes de chargement passent `y_train` à `set_images()`
- ✅ `y_train` est utilisé pour détection unsupervised (MVTec AD)
- ✅ Si `y_train` contient uniquement des labels "normal" (0) → UNSUPERVISED
- ✅ Sinon → SUPERVISED

### **2. Détection de Structure ✅**
- ✅ `detect_dataset_structure()` identifie automatiquement :
  - Structure MVTec AD
  - Dossiers par classe
  - Dossier plat
- ✅ Validation avec messages d'erreur clairs

### **3. Gestion des Erreurs ✅**
- ✅ Validation chemin dossier existe
- ✅ Validation est un dossier (pas un fichier)
- ✅ Validation images trouvées
- ✅ Validation structure valide
- ✅ Try/catch avec logging détaillé
- ✅ Messages utilisateur clairs

### **4. Redirection ✅**
- ✅ Après chargement réussi → Dashboard
- ✅ Utilise `STATE.switch(AppPage.DASHBOARD)`
- ✅ `st.rerun()` pour actualiser

---

## 📊 Formats Supportés

### **Structure MVTec AD**
```
dataset/
  ├── train/
  │   └── good/          # Images normales uniquement
  │       └── *.png
  └── test/
      ├── good/          # Images normales
      │   └── *.png
      └── defect/        # Images avec défauts
          └── *.png
```
✅ Détecté automatiquement → `y_train` = uniquement labels 0 (normal)

### **Structure par Classe**
```
dataset/
  ├── class1/           # Classe 1
  │   └── *.png
  ├── class2/           # Classe 2
  │   └── *.png
  └── class3/           # Classe 3
      └── *.png
```
✅ Détecté automatiquement → Classification multi-classes

### **Dossier Plat**
```
dataset/
  ├── image1.png
  ├── image2.png
  └── image3.png
```
✅ Détecté automatiquement → Toutes images dans même dossier

### **Fichiers Uploadés**
```
Fichiers uploadés → Dossier temporaire → Structure "flat"
```
✅ Tous les fichiers traités comme un dataset plat

---

## ✅ Validation Finale

### **Tout fonctionne correctement :**

1. ✅ **Upload via chemin** : Fonctionnel avec validation robuste
2. ✅ **Upload fichiers multiples** : Fonctionnel avec dossier temporaire
3. ✅ **Exemples MVTec** : 9 datasets disponibles et fonctionnels
4. ✅ **Transmission `y_train`** : Tous les chemins passent `y_train`
5. ✅ **Détection mode** : Automatique et correcte (Unsupervised/Supervised)
6. ✅ **Gestion erreurs** : Complète avec messages clairs
7. ✅ **Redirection** : Vers Dashboard après chargement réussi
8. ✅ **Performance Logger** : Initialisé et utilisé

---

## 🎯 Conclusion

**La page home est production-ready !** ✅

- ✅ **3 onglets principaux** (Tabulaire, Images, MVTec)
- ✅ **2 sous-onglets** dans Images (Dossier, Fichiers)
- ✅ **Tous les chemins** passent `y_train` correctement
- ✅ **Détection automatique** du mode (Unsupervised/Supervised)
- ✅ **Validation robuste** à chaque étape
- ✅ **Messages utilisateur** clairs et informatifs

**Tout est conforme !** 🚀




# 🚗 Compteur de Véhicules — Analyse de Trafic

Analyse vidéo pour compter les véhicules passant sur une rue, en utilisant **YOLOv8** (détection + tracking) et **OpenCV** (gestion vidéo).

---

## 📦 Installation

```bash
# 1. Crée un environnement virtuel (recommandé)
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Installe les dépendances
pip install -r requirements.txt
```

> **Note :** Le modèle `yolov8n.pt` sera téléchargé automatiquement au premier lancement (~6 Mo).

---

## 🎯 Configuration de la ligne de comptage

### Étape 1 : Trouver les coordonnées

1. Place ta vidéo (`video.mp4`) dans le dossier du projet.
2. Lance l'outil de calibration :

```bash
python trouver_coordonnees.py
```

3. **Clique sur l'image** aux points où tu veux placer ta ligne :
   - Un clic sur le **bord gauche** de la route → note le `x` = `LIGNE_X1`
   - Un clic sur le **bord droit** de la route → note le `x` = `LIGNE_X2`
   - Le `y` de ces clics = `LIGNE_Y` (la hauteur de la ligne)

4. Appuie sur `q` pour quitter.

### Étape 2 : Modifier main.py

Ouvre `main.py` et modifie ces 3 variables :

```python
LIGNE_Y  = 400    # ← Remplace par ton y
LIGNE_X1 = 100    # ← Remplace par ton x gauche
LIGNE_X2 = 1180   # ← Remplace par ton x droit
```

---

## ▶️ Utilisation

```bash
python main.py
```

- La vidéo s'affiche avec les détections en temps réel.
- Appuie sur **`q`** pour arrêter.
- À la fin, un résumé s'affiche dans le terminal.

---

## 📊 Résultats

| Fichier              | Description                                    |
|----------------------|------------------------------------------------|
| `traffic_data.csv`   | Données détaillées (timestamp, type, direction) |
| `output_tracked.mp4` | Vidéo annotée avec les bounding boxes          |

### Format du CSV

```
Timestamp,Type_Vehicule,Direction,ID_Tracking
0:00:12,voiture,DESCENDANT,3
0:00:18,camion,MONTANT,7
```

---

## ⚙️ Paramètres ajustables (dans main.py)

| Variable                | Description                          | Défaut   |
|-------------------------|--------------------------------------|----------|
| `VIDEO_PATH`            | Chemin vers la vidéo source          | `video.mp4` |
| `CONFIDENCE_THRESHOLD`  | Seuil de confiance YOLO (0.0-1.0)   | `0.35`   |
| `LIGNE_Y`               | Position Y de la ligne de comptage   | `400`    |
| `LIGNE_X1` / `LIGNE_X2` | Début/fin X de la ligne              | `100` / `1180` |
| `TOLERANCE`             | Marge de détection autour de la ligne | `10` px  |

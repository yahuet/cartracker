# 🚗 Compteur de Véhicules — Analyse de Trafic

Analyse vidéo pour compter les véhicules passant sur une rue, en utilisant **YOLOv8** (détection + tracking) et **OpenCV** (gestion vidéo).

**Deux modes disponibles :**
- 🖥️ **Mode GUI** (recommandé) — Interface graphique moderne avec calibration visuelle
- ⌨️ **Mode CLI** — Script en ligne de commande

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

## 🖥️ Mode GUI (recommandé)

```bash
python gui.py
```

### Utilisation :
1. **Parcourir** — Sélectionne ta vidéo via le bouton
2. **Calibrer** — Clique sur 2 points de l'image pour placer la ligne de comptage
3. **Ajuster** — Modifie le seuil de confiance et les types de véhicules
4. **Démarrer** — Lance l'analyse avec le bouton vert
5. **Résultats** — Les stats s'affichent en direct, le CSV est généré automatiquement

---

## ⌨️ Mode CLI

### Étape 1 : Trouver les coordonnées de la ligne

```bash
python trouver_coordonnees.py
```

Clique sur l'image pour obtenir les coordonnées, puis modifie `main.py` :

```python
LIGNE_Y  = 400    # ← Ton y
LIGNE_X1 = 100    # ← Ton x gauche
LIGNE_X2 = 1180   # ← Ton x droit
```

### Étape 2 : Lancer l'analyse

```bash
python main.py
```

Appuie sur **`q`** pour arrêter.

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

## ⚙️ Paramètres ajustables

| Variable                | Description                          | Défaut   |
|-------------------------|--------------------------------------|----------|
| `VIDEO_PATH`            | Chemin vers la vidéo source          | `video.mp4` |
| `CONFIDENCE_THRESHOLD`  | Seuil de confiance YOLO (0.0-1.0)   | `0.35`   |
| `LIGNE_Y`               | Position Y de la ligne de comptage   | `400`    |
| `LIGNE_X1` / `LIGNE_X2` | Début/fin X de la ligne              | `100` / `1180` |

> En mode GUI, tous ces paramètres sont ajustables visuellement.

---

## 📁 Architecture

```
cartracker/
├── gui.py                  ← Interface graphique (CustomTkinter)
├── counter.py              ← Moteur de comptage (classe VehicleCounter)
├── main.py                 ← Mode CLI
├── trouver_coordonnees.py  ← Outil de calibration CLI
├── requirements.txt        ← Dépendances Python
└── README.md               ← Ce fichier
```

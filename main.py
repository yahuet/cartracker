"""
=============================================================================
 COMPTEUR DE VÉHICULES PAR ANALYSE VIDÉO
 ─────────────────────────────────────────
 Utilise YOLOv8 (nano) pour la détection + tracking d'objets
 et OpenCV pour la gestion vidéo.

 Auteur  : Généré par IA
 Date    : 2026-02-07
 Licence : MIT
=============================================================================
"""

import csv
import os
from datetime import datetime, timedelta

import cv2
from ultralytics import YOLO

# =============================================================================
# ──── CONFIGURATION ────
# =============================================================================

# --- Fichiers d'entrée / sortie ---
VIDEO_PATH = "video.mp4"                  # Chemin vers la vidéo à analyser
OUTPUT_VIDEO = "output_tracked.mp4"       # Vidéo de sortie avec annotations
CSV_PATH = "traffic_data.csv"             # Fichier CSV de résultats

# --- Modèle YOLO ---
MODEL_PATH = "yolov8n.pt"                # Modèle nano (rapide sur CPU)
CONFIDENCE_THRESHOLD = 0.35              # Seuil de confiance minimum

# --- Classes COCO à détecter (indices) ---
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
CLASSES_CIBLES = {
    2: "voiture",
    3: "moto",
    5: "bus",
    7: "camion",
}

# --- Ligne virtuelle de comptage ---
# ⚠️  IMPORTANT : Ajuste ces coordonnées selon ta vidéo !
#     Utilise le script `trouver_coordonnees.py` pour les déterminer.
#     Format : (x1, y1) → (x2, y2)  (point gauche → point droit)
LIGNE_Y = 400                             # Hauteur (y) de la ligne horizontale
LIGNE_X1 = 100                            # Début de la ligne (x gauche)
LIGNE_X2 = 1180                           # Fin de la ligne (x droit)

# --- Tolérance de traversée (en pixels) ---
# Un véhicule est considéré comme "traversant" la ligne s'il passe
# dans une bande de ± TOLERANCE pixels autour de LIGNE_Y.
TOLERANCE = 10

# --- Affichage ---
COULEUR_LIGNE = (0, 255, 255)             # Jaune (BGR)
COULEUR_COMPTEUR = (0, 255, 0)            # Vert (BGR)
COULEUR_BBOX_ACTIVE = (255, 0, 0)         # Bleu (BGR)
EPAISSEUR_LIGNE = 2


# =============================================================================
# ──── FONCTIONS UTILITAIRES ────
# =============================================================================

def calculer_centre(bbox):
    """
    Calcule le centre (cx, cy) d'une bounding box.

    Args:
        bbox: tuple (x1, y1, x2, y2) — coins supérieur-gauche et inférieur-droit.

    Returns:
        tuple (cx, cy) — coordonnées du centre.
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


def determiner_direction(cy_precedent, cy_actuel, ligne_y):
    """
    Détermine la direction de déplacement d'un véhicule par rapport
    à la ligne de comptage.

    Args:
        cy_precedent: position y précédente du centre.
        cy_actuel   : position y actuelle du centre.
        ligne_y     : position y de la ligne virtuelle.

    Returns:
        str: "MONTANT" (vers le haut) ou "DESCENDANT" (vers le bas).
    """
    if cy_precedent < ligne_y and cy_actuel >= ligne_y:
        return "DESCENDANT"
    elif cy_precedent > ligne_y and cy_actuel <= ligne_y:
        return "MONTANT"
    return None


def initialiser_csv(chemin_csv):
    """
    Crée le fichier CSV avec l'en-tête s'il n'existe pas déjà.

    Args:
        chemin_csv: chemin vers le fichier CSV.
    """
    if not os.path.exists(chemin_csv):
        with open(chemin_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Type_Vehicule", "Direction", "ID_Tracking"])


def enregistrer_passage(chemin_csv, timestamp, type_vehicule, direction, track_id):
    """
    Ajoute une ligne au fichier CSV pour chaque véhicule détecté
    traversant la ligne.

    Args:
        chemin_csv    : chemin vers le fichier CSV.
        timestamp     : moment du passage (format ISO).
        type_vehicule : type de véhicule (voiture, camion, bus, moto).
        direction     : direction du véhicule (MONTANT / DESCENDANT).
        track_id      : identifiant unique de tracking YOLO.
    """
    with open(chemin_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, type_vehicule, direction, track_id])


def dessiner_interface(frame, compteur_total, compteurs_par_type,
                       compteur_montant, compteur_descendant):
    """
    Dessine la ligne de comptage et les statistiques sur l'image.

    Args:
        frame              : l'image courante (numpy array BGR).
        compteur_total     : nombre total de véhicules comptés.
        compteurs_par_type : dict {type: count} pour chaque type.
        compteur_montant   : nombre de véhicules montants.
        compteur_descendant: nombre de véhicules descendants.
    """
    # --- Ligne de comptage ---
    cv2.line(frame, (LIGNE_X1, LIGNE_Y), (LIGNE_X2, LIGNE_Y),
             COULEUR_LIGNE, EPAISSEUR_LIGNE)

    # --- Panneau de statistiques (fond semi-transparent) ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # --- Texte des compteurs ---
    y_texte = 40
    cv2.putText(frame, f"TOTAL : {compteur_total}",
                (20, y_texte), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                COULEUR_COMPTEUR, 2)

    y_texte += 30
    cv2.putText(frame, f"Montant: {compteur_montant}  |  Descendant: {compteur_descendant}",
                (20, y_texte), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)

    # --- Détail par type ---
    y_texte += 30
    for type_v, count in compteurs_par_type.items():
        cv2.putText(frame, f"  {type_v}: {count}",
                    (20, y_texte), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        y_texte += 25

    return frame


# =============================================================================
# ──── FONCTION PRINCIPALE ────
# =============================================================================

def compter_vehicules():
    """
    Fonction principale qui :
      1. Charge le modèle YOLOv8 nano.
      2. Ouvre la vidéo source.
      3. Effectue la détection + tracking frame par frame.
      4. Compte les véhicules traversant la ligne virtuelle.
      5. Sauvegarde les résultats dans un CSV.
      6. Affiche la vidéo annotée en temps réel.
    """

    # ── 1. Chargement du modèle ──
    print("=" * 60)
    print(" COMPTEUR DE VÉHICULES — Démarrage")
    print("=" * 60)
    print(f"[INFO] Chargement du modèle : {MODEL_PATH}")
    modele = YOLO(MODEL_PATH)

    # ── 2. Ouverture de la vidéo ──
    print(f"[INFO] Ouverture de la vidéo : {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la vidéo : {VIDEO_PATH}")
        print("         Vérifie que le fichier existe et que le chemin est correct.")
        return

    # Propriétés de la vidéo
    largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Résolution : {largeur}x{hauteur} | FPS : {fps:.1f} | Frames : {total_frames}")
    print(f"[INFO] Ligne de comptage : y={LIGNE_Y}, x=[{LIGNE_X1} → {LIGNE_X2}]")
    print(f"[INFO] Classes détectées : {list(CLASSES_CIBLES.values())}")
    print("-" * 60)

    # ── 3. Initialisation du CSV ──
    initialiser_csv(CSV_PATH)

    # ── 4. Initialisation de l'enregistrement vidéo (optionnel) ──
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (largeur, hauteur))

    # ── 5. Variables de suivi ──
    compteur_total = 0                      # Total de véhicules comptés
    compteur_montant = 0                    # Véhicules allant vers le haut
    compteur_descendant = 0                 # Véhicules allant vers le bas
    compteurs_par_type = {                  # Compteur par type de véhicule
        "voiture": 0,
        "moto": 0,
        "bus": 0,
        "camion": 0,
    }
    vehicules_comptes = set()               # IDs déjà comptés (évite les doublons)
    positions_precedentes = {}              # {track_id: cy_precedent}
    numero_frame = 0                        # Compteur de frames
    heure_debut = datetime.now()            # Heure de début du traitement

    # ── 6. Boucle principale frame par frame ──
    print("[INFO] Traitement en cours... Appuie sur 'q' pour quitter.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Fin de la vidéo
            print("\n[INFO] Fin de la vidéo atteinte.")
            break

        numero_frame += 1

        # Calcul du timestamp relatif (basé sur le FPS)
        secondes_ecoulees = numero_frame / fps if fps > 0 else 0
        timestamp_relatif = str(timedelta(seconds=int(secondes_ecoulees)))

        # ── Détection + Tracking avec YOLOv8 ──
        # persist=True maintient les IDs entre les frames
        resultats = modele.track(
            frame,
            persist=True,
            conf=CONFIDENCE_THRESHOLD,
            classes=list(CLASSES_CIBLES.keys()),
            verbose=False,
            tracker="bytetrack.yaml",       # Tracker performant par défaut
        )

        # ── Traitement des détections ──
        if resultats and resultats[0].boxes is not None and resultats[0].boxes.id is not None:
            boxes = resultats[0].boxes

            for i in range(len(boxes)):
                # Récupération des données de la détection
                bbox = boxes.xyxy[i].cpu().numpy()         # Coordonnées [x1, y1, x2, y2]
                track_id = int(boxes.id[i].cpu().numpy())  # ID unique de tracking
                classe_id = int(boxes.cls[i].cpu().numpy()) # Classe COCO
                confiance = float(boxes.conf[i].cpu().numpy())

                # Vérifier que la classe est dans notre filtre
                if classe_id not in CLASSES_CIBLES:
                    continue

                type_vehicule = CLASSES_CIBLES[classe_id]

                # Calcul du centre de la bounding box
                cx, cy = calculer_centre(bbox)

                # ── Dessiner la bounding box et le label ──
                x1, y1, x2, y2 = map(int, bbox)
                label = f"ID:{track_id} {type_vehicule} {confiance:.0%}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), COULEUR_BBOX_ACTIVE, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COULEUR_BBOX_ACTIVE, 1)

                # Point central du véhicule
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # ── Logique de comptage par traversée de ligne ──
                if track_id in positions_precedentes and track_id not in vehicules_comptes:
                    cy_precedent = positions_precedentes[track_id]

                    # Vérifier si le véhicule traverse la ligne
                    # (il était d'un côté et il est maintenant de l'autre)
                    direction = determiner_direction(cy_precedent, cy, LIGNE_Y)

                    if direction is not None:
                        # ✅ Nouveau passage détecté !
                        compteur_total += 1
                        compteurs_par_type[type_vehicule] += 1
                        vehicules_comptes.add(track_id)

                        if direction == "MONTANT":
                            compteur_montant += 1
                        else:
                            compteur_descendant += 1

                        # Enregistrement dans le CSV
                        enregistrer_passage(
                            CSV_PATH,
                            timestamp_relatif,
                            type_vehicule,
                            direction,
                            track_id,
                        )

                        print(f"  🚗 #{compteur_total:03d} | {timestamp_relatif} | "
                              f"{type_vehicule:<8} | {direction:<11} | ID: {track_id}")

                # Mémoriser la position actuelle pour la prochaine frame
                positions_precedentes[track_id] = cy

        # ── Dessiner l'interface sur la frame ──
        frame = dessiner_interface(
            frame, compteur_total, compteurs_par_type,
            compteur_montant, compteur_descendant,
        )

        # Écriture dans la vidéo de sortie
        writer.write(frame)

        # ── Affichage en temps réel ──
        cv2.imshow("Compteur de Vehicules", frame)

        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Arrêt demandé par l'utilisateur (touche 'q').")
            break

    # ── 7. Nettoyage et fermeture ──
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # ── 8. Résumé final ──
    duree = datetime.now() - heure_debut
    print("\n" + "=" * 60)
    print(" RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"  Frames traitées      : {numero_frame}/{total_frames}")
    print(f"  Durée du traitement  : {duree}")
    print(f"  ──────────────────────────────────")
    print(f"  TOTAL véhicules      : {compteur_total}")
    print(f"  ├─ Montants          : {compteur_montant}")
    print(f"  └─ Descendants       : {compteur_descendant}")
    print(f"  ──────────────────────────────────")
    for type_v, count in compteurs_par_type.items():
        print(f"  {type_v:<10} : {count}")
    print(f"  ──────────────────────────────────")
    print(f"  Données sauvegardées : {CSV_PATH}")
    print(f"  Vidéo annotée        : {OUTPUT_VIDEO}")
    print("=" * 60)


# =============================================================================
# ──── POINT D'ENTRÉE ────
# =============================================================================

if __name__ == "__main__":
    compter_vehicules()

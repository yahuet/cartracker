"""
=============================================================================
 MOTEUR DE COMPTAGE DE VÉHICULES (Classe réutilisable)
 ─────────────────────────────────────────────────────
 Contient la logique de détection, tracking et comptage de véhicules.
 Utilisable depuis le CLI (main.py) ou le GUI (gui.py).

 Auteur  : Généré par IA
 Date    : 2026-02-08
 Licence : MIT
=============================================================================
"""

import csv
import os
from datetime import timedelta

import cv2
from ultralytics import YOLO


# =============================================================================
# ──── CONSTANTES PAR DÉFAUT ────
# =============================================================================

# Classes COCO pertinentes pour les véhicules
CLASSES_VEHICULES = {
    2: "voiture",
    3: "moto",
    5: "bus",
    7: "camion",
}

# Couleurs d'affichage (BGR)
COULEUR_LIGNE = (0, 255, 255)          # Jaune
COULEUR_COMPTEUR = (0, 255, 0)         # Vert
COULEUR_BBOX = (255, 0, 0)            # Bleu
COULEUR_CENTRE = (0, 0, 255)          # Rouge


# =============================================================================
# ──── CLASSE PRINCIPALE ────
# =============================================================================

class VehicleCounter:
    """
    Moteur de comptage de véhicules par analyse vidéo.

    Utilise YOLOv8 pour la détection + tracking et une ligne virtuelle
    pour compter les traversées.

    Attributs:
        modele          : modèle YOLO chargé
        cap             : capture vidéo OpenCV
        en_cours        : True si l'analyse est en cours
        compteur_total  : nombre total de véhicules comptés
    """

    def __init__(self, video_path, model_path="yolov8n.pt",
                 confidence=0.35, ligne_y=400, ligne_x1=100, ligne_x2=1180,
                 classes_actives=None, csv_path="traffic_data.csv",
                 output_video_path="output_tracked.mp4"):
        """
        Initialise le compteur.

        Args:
            video_path       : chemin vers le fichier vidéo.
            model_path       : chemin vers le modèle YOLO (.pt).
            confidence       : seuil de confiance (0.0 → 1.0).
            ligne_y          : position Y de la ligne de comptage.
            ligne_x1         : position X gauche de la ligne.
            ligne_x2         : position X droite de la ligne.
            classes_actives  : dict {id_coco: nom} des classes à détecter.
            csv_path         : chemin du fichier CSV de sortie.
            output_video_path: chemin de la vidéo annotée de sortie.
        """
        # --- Configuration ---
        self.video_path = video_path
        self.model_path = model_path
        self.confidence = confidence
        self.ligne_y = ligne_y
        self.ligne_x1 = ligne_x1
        self.ligne_x2 = ligne_x2
        self.classes_actives = classes_actives or dict(CLASSES_VEHICULES)
        self.csv_path = csv_path
        self.output_video_path = output_video_path

        # --- État interne ---
        self.modele = None
        self.cap = None
        self.writer = None
        self.en_cours = False

        # --- Compteurs ---
        self.compteur_total = 0
        self.compteur_montant = 0
        self.compteur_descendant = 0
        self.compteurs_par_type = {nom: 0 for nom in CLASSES_VEHICULES.values()}

        # --- Tracking ---
        self.vehicules_comptes = set()       # IDs déjà comptés
        self.positions_precedentes = {}      # {track_id: cy_precedent}
        self.numero_frame = 0

        # --- Infos vidéo ---
        self.largeur = 0
        self.hauteur = 0
        self.fps = 0
        self.total_frames = 0

    # ─────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────

    def initialiser(self):
        """
        Charge le modèle YOLO et ouvre la vidéo.

        Returns:
            bool: True si l'initialisation a réussi.

        Raises:
            FileNotFoundError: si le fichier vidéo n'existe pas.
        """
        # Charger le modèle
        self.modele = YOLO(self.model_path)

        # Ouvrir la vidéo
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(
                f"Impossible d'ouvrir la vidéo : {self.video_path}"
            )

        # Récupérer les propriétés
        self.largeur = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.hauteur = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialiser l'enregistrement vidéo
        if self.output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                self.output_video_path, fourcc, self.fps,
                (self.largeur, self.hauteur)
            )

        # Initialiser le CSV
        self._initialiser_csv()

        # Réinitialiser les compteurs
        self._reinitialiser_compteurs()
        self.en_cours = True

        return True

    def _reinitialiser_compteurs(self):
        """Remet tous les compteurs à zéro."""
        self.compteur_total = 0
        self.compteur_montant = 0
        self.compteur_descendant = 0
        self.compteurs_par_type = {nom: 0 for nom in CLASSES_VEHICULES.values()}
        self.vehicules_comptes = set()
        self.positions_precedentes = {}
        self.numero_frame = 0

    # ─────────────────────────────────────────────────────────────
    # Traitement frame par frame
    # ─────────────────────────────────────────────────────────────

    def traiter_frame_suivante(self):
        """
        Lit et traite la prochaine frame de la vidéo.

        Returns:
            tuple (frame_annotee, nouveau_passage) ou (None, None) si fin de vidéo.
                - frame_annotee  : numpy array BGR avec annotations dessinées.
                - nouveau_passage: dict si un véhicule vient de traverser la ligne,
                                   None sinon.
                  Format du dict: {
                      "timestamp": "0:01:23",
                      "type": "voiture",
                      "direction": "MONTANT",
                      "track_id": 5
                  }
        """
        if not self.en_cours or self.cap is None:
            return None, None

        ret, frame = self.cap.read()
        if not ret:
            self.en_cours = False
            return None, None

        self.numero_frame += 1

        # Timestamp relatif
        secondes = self.numero_frame / self.fps if self.fps > 0 else 0
        timestamp = str(timedelta(seconds=int(secondes)))

        # Détection + tracking
        resultats = self.modele.track(
            frame,
            persist=True,
            conf=self.confidence,
            classes=list(self.classes_actives.keys()),
            verbose=False,
            tracker="bytetrack.yaml",
        )

        nouveau_passage = None

        # Traitement des détections
        if (resultats and resultats[0].boxes is not None
                and resultats[0].boxes.id is not None):

            boxes = resultats[0].boxes

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                track_id = int(boxes.id[i].cpu().numpy())
                classe_id = int(boxes.cls[i].cpu().numpy())
                confiance = float(boxes.conf[i].cpu().numpy())

                if classe_id not in self.classes_actives:
                    continue

                type_vehicule = self.classes_actives[classe_id]
                cx, cy = self._calculer_centre(bbox)

                # Dessiner la bbox et le label
                x1, y1, x2, y2 = map(int, bbox)
                label = f"ID:{track_id} {type_vehicule} {confiance:.0%}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), COULEUR_BBOX, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COULEUR_BBOX, 1)
                cv2.circle(frame, (cx, cy), 5, COULEUR_CENTRE, -1)

                # Logique de traversée de ligne
                if (track_id in self.positions_precedentes
                        and track_id not in self.vehicules_comptes):

                    cy_prec = self.positions_precedentes[track_id]
                    direction = self._determiner_direction(cy_prec, cy)

                    if direction is not None:
                        self.compteur_total += 1
                        self.compteurs_par_type[type_vehicule] = \
                            self.compteurs_par_type.get(type_vehicule, 0) + 1
                        self.vehicules_comptes.add(track_id)

                        if direction == "MONTANT":
                            self.compteur_montant += 1
                        else:
                            self.compteur_descendant += 1

                        # Enregistrer dans le CSV
                        self._enregistrer_passage(
                            timestamp, type_vehicule, direction, track_id
                        )

                        nouveau_passage = {
                            "timestamp": timestamp,
                            "type": type_vehicule,
                            "direction": direction,
                            "track_id": track_id,
                        }

                self.positions_precedentes[track_id] = cy

        # Dessiner l'interface sur la frame
        self._dessiner_overlay(frame)

        # Écrire dans la vidéo de sortie
        if self.writer:
            self.writer.write(frame)

        return frame, nouveau_passage

    # ─────────────────────────────────────────────────────────────
    # Accesseurs
    # ─────────────────────────────────────────────────────────────

    def obtenir_stats(self):
        """
        Retourne un dictionnaire des statistiques actuelles.

        Returns:
            dict avec les clés: total, montant, descendant, par_type, progression.
        """
        progression = 0
        if self.total_frames > 0:
            progression = (self.numero_frame / self.total_frames) * 100

        return {
            "total": self.compteur_total,
            "montant": self.compteur_montant,
            "descendant": self.compteur_descendant,
            "par_type": dict(self.compteurs_par_type),
            "progression": progression,
            "frame": self.numero_frame,
            "total_frames": self.total_frames,
        }

    def obtenir_premiere_frame(self):
        """
        Lit la première frame de la vidéo sans démarrer l'analyse.

        Returns:
            numpy array BGR de la première frame, ou None si erreur.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    # ─────────────────────────────────────────────────────────────
    # Arrêt / Nettoyage
    # ─────────────────────────────────────────────────────────────

    def arreter(self):
        """Arrête proprement l'analyse et libère les ressources."""
        self.en_cours = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.writer:
            self.writer.release()
            self.writer = None

    # ─────────────────────────────────────────────────────────────
    # Méthodes privées
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _calculer_centre(bbox):
        """Calcule le centre (cx, cy) d'une bounding box."""
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _determiner_direction(self, cy_precedent, cy_actuel):
        """Détermine la direction de traversée de la ligne."""
        if cy_precedent < self.ligne_y and cy_actuel >= self.ligne_y:
            return "DESCENDANT"
        elif cy_precedent > self.ligne_y and cy_actuel <= self.ligne_y:
            return "MONTANT"
        return None

    def _initialiser_csv(self):
        """Crée le fichier CSV avec l'en-tête s'il n'existe pas."""
        if self.csv_path and not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Type_Vehicule", "Direction", "ID_Tracking"
                ])

    def _enregistrer_passage(self, timestamp, type_vehicule, direction, track_id):
        """Ajoute une ligne au CSV pour un passage détecté."""
        if not self.csv_path:
            return
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, type_vehicule, direction, track_id])

    def _dessiner_overlay(self, frame):
        """Dessine la ligne de comptage et les statistiques sur la frame."""
        # Ligne de comptage
        cv2.line(frame, (self.ligne_x1, self.ligne_y),
                 (self.ligne_x2, self.ligne_y), COULEUR_LIGNE, 2)

        # Panneau semi-transparent
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Compteurs
        y_t = 40
        cv2.putText(frame, f"TOTAL : {self.compteur_total}",
                    (20, y_t), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    COULEUR_COMPTEUR, 2)

        y_t += 30
        cv2.putText(
            frame,
            f"Montant: {self.compteur_montant}  |  "
            f"Descendant: {self.compteur_descendant}",
            (20, y_t), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

        y_t += 30
        for type_v, count in self.compteurs_par_type.items():
            cv2.putText(frame, f"  {type_v}: {count}",
                        (20, y_t), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            y_t += 25

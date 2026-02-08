"""
=============================================================================
 OUTIL : Trouver les coordonnées de la ligne de comptage
 ─────────────────────────────────────────────────────────
 Ce petit script ouvre la première frame de ta vidéo et te permet
 de cliquer dessus pour lire les coordonnées (x, y) dans le terminal.

 Utilisation :
   1. Lance ce script : python trouver_coordonnees.py
   2. Clique sur la vidéo aux endroits où tu veux placer ta ligne.
   3. Note les coordonnées affichées dans le terminal.
   4. Reporte-les dans main.py (LIGNE_Y, LIGNE_X1, LIGNE_X2).
   5. Appuie sur 'q' pour quitter.
=============================================================================
"""

import cv2

# ── Chemin vers ta vidéo (le même que dans main.py) ──
VIDEO_PATH = "video.mp4"


def clic_souris(event, x, y, flags, param):
    """
    Callback appelé à chaque clic de souris sur l'image.
    Affiche les coordonnées et dessine un point rouge.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"  📍 Coordonnées cliquées : x={x}, y={y}")

        # Dessiner un point rouge sur l'image
        cv2.circle(param["frame"], (x, y), 5, (0, 0, 255), -1)
        cv2.putText(param["frame"], f"({x},{y})", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("Calibration - Cliquez pour les coordonnees", param["frame"])


def main():
    """
    Ouvre la première frame de la vidéo et attend les clics utilisateur.
    """
    print("=" * 60)
    print(" OUTIL DE CALIBRATION — Ligne de comptage")
    print("=" * 60)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir : {VIDEO_PATH}")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERREUR] Impossible de lire la première frame.")
        return

    hauteur, largeur = frame.shape[:2]
    print(f"[INFO] Résolution de la vidéo : {largeur}x{hauteur}")
    print()
    print("  → Clique sur l'image pour obtenir les coordonnées (x, y).")
    print("  → Note le 'y' pour LIGNE_Y et les 'x' pour LIGNE_X1 / LIGNE_X2.")
    print("  → Appuie sur 'q' pour quitter.")
    print()

    # Contexte partagé avec le callback
    contexte = {"frame": frame.copy()}

    cv2.imshow("Calibration - Cliquez pour les coordonnees", frame)
    cv2.setMouseCallback("Calibration - Cliquez pour les coordonnees",
                         clic_souris, contexte)

    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("[INFO] Calibration terminée.")


if __name__ == "__main__":
    main()

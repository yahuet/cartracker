"""
=============================================================================
 INTERFACE GRAPHIQUE — COMPTEUR DE VÉHICULES
 ────────────────────────────────────────────
 Interface CustomTkinter (thème sombre) pour l'analyse vidéo
 de trafic avec YOLOv8.

 Fonctionnalités :
   - Sélection de vidéo via dialogue fichier OU lien YouTube
   - Calibration visuelle de la ligne de comptage (clic)
   - Paramètres ajustables (confiance, types de véhicules)
   - Vidéo en temps réel avec détections
   - Statistiques en direct
   - Export CSV automatique

 Auteur  : Généré par IA
 Date    : 2026-02-08
 Licence : MIT
=============================================================================
"""

import os
import re
import threading
import subprocess
from tkinter import filedialog

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

from counter import VehicleCounter, CLASSES_VEHICULES


# =============================================================================
# ──── CONFIGURATION CUSTOMTKINTER ────
# =============================================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# =============================================================================
# ──── FENÊTRE PRINCIPALE ────
# =============================================================================

class App(ctk.CTk):
    """
    Fenêtre principale de l'application de comptage de véhicules.
    """

    # --- Dimensions ---
    LARGEUR_FENETRE = 1100
    HAUTEUR_FENETRE = 700
    LARGEUR_VIDEO = 680
    HAUTEUR_VIDEO = 480

    def __init__(self):
        super().__init__()

        # ── Configuration de la fenêtre ──
        self.title("🚗 Compteur de Véhicules — Analyse de Trafic")
        self.geometry(f"{self.LARGEUR_FENETRE}x{self.HAUTEUR_FENETRE}")
        self.minsize(900, 600)

        # ── Variables ──
        self.video_path = None
        self.counter = None
        self.analyse_thread = None
        self.en_cours = False

        # Variables de calibration (ligne de comptage)
        self.points_ligne = []        # Liste de 2 points [(x1,y1), (x2,y2)]
        self.mode_calibration = False
        self.premiere_frame = None     # Frame originale pour la calibration
        self.photo_image = None        # Référence pour éviter le garbage collection

        # Variables des cases à cocher (types de véhicules)
        self.var_voiture = ctk.BooleanVar(value=True)
        self.var_camion = ctk.BooleanVar(value=True)
        self.var_bus = ctk.BooleanVar(value=True)
        self.var_moto = ctk.BooleanVar(value=True)

        # Variable du slider de confiance
        self.var_confiance = ctk.DoubleVar(value=0.35)

        # ── Construction de l'interface ──
        self._construire_interface()

        # ── Gestion de la fermeture ──
        self.protocol("WM_DELETE_WINDOW", self._quitter)

    # ─────────────────────────────────────────────────────────────
    # Construction de l'interface
    # ─────────────────────────────────────────────────────────────

    def _construire_interface(self):
        """Construit tous les widgets de l'interface."""

        # Layout principal : 2 colonnes
        self.grid_columnconfigure(0, weight=3)   # Colonne vidéo (plus large)
        self.grid_columnconfigure(1, weight=1)   # Colonne contrôles
        self.grid_rowconfigure(0, weight=1)

        # ── PANNEAU GAUCHE : Vidéo ──
        self._construire_panneau_video()

        # ── PANNEAU DROIT : Contrôles ──
        self._construire_panneau_controles()

    def _construire_panneau_video(self):
        """Construit le panneau gauche contenant la vidéo."""

        frame_video = ctk.CTkFrame(self, corner_radius=10)
        frame_video.grid(row=0, column=0, padx=(15, 7), pady=15, sticky="nsew")
        frame_video.grid_rowconfigure(0, weight=1)
        frame_video.grid_columnconfigure(0, weight=1)

        # Canvas pour la vidéo
        self.canvas_video = ctk.CTkCanvas(
            frame_video,
            bg="#1a1a2e",
            highlightthickness=0,
        )
        self.canvas_video.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Texte par défaut sur le canvas
        self.canvas_video.bind("<Configure>", self._on_canvas_resize)

        # Clic pour la calibration
        self.canvas_video.bind("<Button-1>", self._on_clic_canvas)

        # Barre de progression en bas
        self.label_progression = ctk.CTkLabel(
            frame_video, text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#00bfff", height=20,
        )
        self.label_progression.grid(row=1, column=0, padx=10, pady=(5, 0), sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(frame_video, height=12)
        self.progress_bar.grid(row=2, column=0, padx=10, pady=(2, 10), sticky="ew")
        self.progress_bar.set(0)

    def _construire_panneau_controles(self):
        """Construit le panneau droit avec les contrôles."""

        # Frame scrollable pour le panneau droit
        self.frame_controles = ctk.CTkScrollableFrame(
            self, corner_radius=10, label_text="Contrôles"
        )
        self.frame_controles.grid(
            row=0, column=1, padx=(7, 15), pady=15, sticky="nsew"
        )

        # ── Section : Fichier vidéo ──
        self._section_fichier()

        # ── Section : Ligne de comptage ──
        self._section_calibration()

        # ── Section : Paramètres ──
        self._section_parametres()

        # ── Section : Statistiques ──
        self._section_statistiques()

        # ── Section : Actions ──
        self._section_actions()

    # ─────────────────────────────────────────────────────────────
    # Sections du panneau de contrôles
    # ─────────────────────────────────────────────────────────────

    def _section_fichier(self):
        """Section pour la sélection du fichier vidéo (local ou YouTube)."""

        label = ctk.CTkLabel(
            self.frame_controles, text="📂 Fichier vidéo",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        label.pack(anchor="w", padx=5, pady=(10, 5))

        # Affichage du chemin
        self.label_chemin = ctk.CTkLabel(
            self.frame_controles, text="Aucune vidéo sélectionnée",
            font=ctk.CTkFont(size=11), text_color="gray",
            wraplength=250,
        )
        self.label_chemin.pack(anchor="w", padx=10, pady=2)

        btn_parcourir = ctk.CTkButton(
            self.frame_controles, text="📁 Parcourir...",
            command=self._ouvrir_video, height=32
        )
        btn_parcourir.pack(fill="x", padx=10, pady=5)

        # ── Lien YouTube ──
        ctk.CTkLabel(
            self.frame_controles, text="ou coller un lien YouTube :",
            font=ctk.CTkFont(size=11), text_color="gray70",
        ).pack(anchor="w", padx=10, pady=(5, 2))

        self.entry_url = ctk.CTkEntry(
            self.frame_controles,
            placeholder_text="https://www.youtube.com/watch?v=...",
            height=32,
        )
        self.entry_url.pack(fill="x", padx=10, pady=2)

        self.btn_telecharger = ctk.CTkButton(
            self.frame_controles, text="⬇️  Télécharger",
            command=self._telecharger_youtube, height=32,
            fg_color="#6f42c1", hover_color="#5a32a3",
        )
        self.btn_telecharger.pack(fill="x", padx=10, pady=5)

        # ── Connexion YouTube ──
        cookies_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cookies.txt"
        )
        btn_connect_text = (
            "✅ YouTube connecté" if os.path.exists(cookies_path)
            else "🔑 Connecter YouTube"
        )
        self.btn_connecter_yt = ctk.CTkButton(
            self.frame_controles, text=btn_connect_text,
            command=self._connecter_youtube, height=28,
            fg_color="#2d6a4f" if os.path.exists(cookies_path) else "#555",
            hover_color="#1b4332" if os.path.exists(cookies_path) else "#666",
            font=ctk.CTkFont(size=11),
        )
        self.btn_connecter_yt.pack(fill="x", padx=10, pady=(0, 5))

        # Séparateur
        sep = ctk.CTkFrame(self.frame_controles, height=2, fg_color="gray30")
        sep.pack(fill="x", padx=5, pady=10)

    def _section_calibration(self):
        """Section pour la calibration de la ligne de comptage."""

        label = ctk.CTkLabel(
            self.frame_controles, text="📏 Ligne de comptage",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        label.pack(anchor="w", padx=5, pady=(5, 5))

        self.label_calibration = ctk.CTkLabel(
            self.frame_controles,
            text="Sélectionnez une vidéo puis\ncliquez sur l'image pour\nplacer 2 points.",
            font=ctk.CTkFont(size=11), text_color="gray",
            justify="left",
        )
        self.label_calibration.pack(anchor="w", padx=10, pady=2)

        self.btn_reinitialiser = ctk.CTkButton(
            self.frame_controles, text="🔄 Réinitialiser la ligne",
            command=self._reinitialiser_ligne, height=28,
            fg_color="gray30", hover_color="gray40",
        )
        self.btn_reinitialiser.pack(fill="x", padx=10, pady=5)

        sep = ctk.CTkFrame(self.frame_controles, height=2, fg_color="gray30")
        sep.pack(fill="x", padx=5, pady=10)

    def _section_parametres(self):
        """Section pour les paramètres de détection."""

        label = ctk.CTkLabel(
            self.frame_controles, text="⚙️ Paramètres",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        label.pack(anchor="w", padx=5, pady=(5, 5))

        # Slider de confiance
        frame_conf = ctk.CTkFrame(self.frame_controles, fg_color="transparent")
        frame_conf.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(
            frame_conf, text="Confiance :", font=ctk.CTkFont(size=12)
        ).pack(side="left")

        self.label_confiance_val = ctk.CTkLabel(
            frame_conf, text="0.35", font=ctk.CTkFont(size=12, weight="bold"),
            width=40
        )
        self.label_confiance_val.pack(side="right")

        self.slider_confiance = ctk.CTkSlider(
            self.frame_controles, from_=0.1, to=1.0,
            variable=self.var_confiance,
            command=self._on_confiance_change,
        )
        self.slider_confiance.pack(fill="x", padx=10, pady=(0, 8))

        # Cases à cocher pour les types
        ctk.CTkLabel(
            self.frame_controles, text="Types détectés :",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=10, pady=(5, 2))

        frame_types = ctk.CTkFrame(self.frame_controles, fg_color="transparent")
        frame_types.pack(fill="x", padx=10)

        ctk.CTkCheckBox(
            frame_types, text="Voiture", variable=self.var_voiture,
            font=ctk.CTkFont(size=12)
        ).grid(row=0, column=0, padx=5, pady=2, sticky="w")

        ctk.CTkCheckBox(
            frame_types, text="Camion", variable=self.var_camion,
            font=ctk.CTkFont(size=12)
        ).grid(row=0, column=1, padx=5, pady=2, sticky="w")

        ctk.CTkCheckBox(
            frame_types, text="Bus", variable=self.var_bus,
            font=ctk.CTkFont(size=12)
        ).grid(row=1, column=0, padx=5, pady=2, sticky="w")

        ctk.CTkCheckBox(
            frame_types, text="Moto", variable=self.var_moto,
            font=ctk.CTkFont(size=12)
        ).grid(row=1, column=1, padx=5, pady=2, sticky="w")

        sep = ctk.CTkFrame(self.frame_controles, height=2, fg_color="gray30")
        sep.pack(fill="x", padx=5, pady=10)

    def _section_statistiques(self):
        """Section pour les statistiques en temps réel."""

        label = ctk.CTkLabel(
            self.frame_controles, text="📊 Statistiques",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        label.pack(anchor="w", padx=5, pady=(5, 5))

        # Frame pour les stats
        self.frame_stats = ctk.CTkFrame(
            self.frame_controles, fg_color="gray15", corner_radius=8
        )
        self.frame_stats.pack(fill="x", padx=10, pady=5)

        # Labels de statistiques
        self.label_total = ctk.CTkLabel(
            self.frame_stats, text="Total : 0",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#00ff88",
        )
        self.label_total.pack(anchor="w", padx=15, pady=(10, 2))

        self.label_directions = ctk.CTkLabel(
            self.frame_stats,
            text="↑ Montant : 0    ↓ Descendant : 0",
            font=ctk.CTkFont(size=12), text_color="gray70",
        )
        self.label_directions.pack(anchor="w", padx=15, pady=2)

        # Séparateur fin
        sep_stats = ctk.CTkFrame(self.frame_stats, height=1, fg_color="gray30")
        sep_stats.pack(fill="x", padx=10, pady=5)

        self.label_voiture = ctk.CTkLabel(
            self.frame_stats, text="🚗 Voiture : 0",
            font=ctk.CTkFont(size=12)
        )
        self.label_voiture.pack(anchor="w", padx=15, pady=1)

        self.label_camion = ctk.CTkLabel(
            self.frame_stats, text="🚛 Camion  : 0",
            font=ctk.CTkFont(size=12)
        )
        self.label_camion.pack(anchor="w", padx=15, pady=1)

        self.label_bus = ctk.CTkLabel(
            self.frame_stats, text="🚌 Bus     : 0",
            font=ctk.CTkFont(size=12)
        )
        self.label_bus.pack(anchor="w", padx=15, pady=1)

        self.label_moto = ctk.CTkLabel(
            self.frame_stats, text="🏍️ Moto    : 0",
            font=ctk.CTkFont(size=12)
        )
        self.label_moto.pack(anchor="w", padx=15, pady=(1, 10))

        sep = ctk.CTkFrame(self.frame_controles, height=2, fg_color="gray30")
        sep.pack(fill="x", padx=5, pady=10)

    def _section_actions(self):
        """Section pour les boutons d'action."""

        # Boutons Démarrer / Arrêter
        frame_btns = ctk.CTkFrame(self.frame_controles, fg_color="transparent")
        frame_btns.pack(fill="x", padx=10, pady=5)
        frame_btns.grid_columnconfigure(0, weight=1)
        frame_btns.grid_columnconfigure(1, weight=1)

        self.btn_demarrer = ctk.CTkButton(
            frame_btns, text="▶  Démarrer",
            command=self._demarrer_analyse,
            fg_color="#28a745", hover_color="#218838",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=40,
        )
        self.btn_demarrer.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        self.btn_arreter = ctk.CTkButton(
            frame_btns, text="⏹  Arrêter",
            command=self._arreter_analyse,
            fg_color="#dc3545", hover_color="#c82333",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=40,
            state="disabled",
        )
        self.btn_arreter.grid(row=0, column=1, padx=(5, 0), sticky="ew")

        # Bouton Ouvrir CSV
        self.btn_csv = ctk.CTkButton(
            self.frame_controles, text="📊 Ouvrir le CSV",
            command=self._ouvrir_csv,
            fg_color="gray30", hover_color="gray40",
            height=32,
        )
        self.btn_csv.pack(fill="x", padx=10, pady=(10, 5))

        # Label de statut
        self.label_statut = ctk.CTkLabel(
            self.frame_controles, text="En attente...",
            font=ctk.CTkFont(size=11), text_color="gray50",
        )
        self.label_statut.pack(anchor="w", padx=10, pady=5)

    # ─────────────────────────────────────────────────────────────
    # Gestion de la vidéo et calibration
    # ─────────────────────────────────────────────────────────────

    def _ouvrir_video(self):
        """Ouvre un dialogue pour sélectionner un fichier vidéo."""
        chemin = filedialog.askopenfilename(
            title="Sélectionner une vidéo",
            filetypes=[
                ("Fichiers vidéo", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("Tous les fichiers", "*.*"),
            ]
        )
        if not chemin:
            return

        self._charger_video(chemin)

    def _telecharger_youtube(self):
        """Télécharge une vidéo YouTube via yt-dlp dans un thread."""
        url = self.entry_url.get().strip()
        if not url:
            self.label_statut.configure(
                text="⚠️ Collez un lien YouTube d'abord !",
                text_color="#ff6b6b",
            )
            return

        # Validation basique de l'URL
        pattern = r"(youtube\.com|youtu\.be)"
        if not re.search(pattern, url):
            self.label_statut.configure(
                text="⚠️ Ce n'est pas un lien YouTube valide",
                text_color="#ff6b6b",
            )
            return

        # Désactiver le bouton pendant le téléchargement
        self.btn_telecharger.configure(state="disabled", text="⏳ Téléchargement...")
        self.label_statut.configure(
            text="⬇️ Téléchargement en cours...", text_color="#00bfff"
        )
        self.label_chemin.configure(text="Téléchargement...", text_color="#ffcc00")
        self.progress_bar.set(0)

        # Lancer le téléchargement dans un thread
        thread = threading.Thread(
            target=self._thread_telecharger, args=(url,), daemon=True
        )
        thread.start()

    def _maj_progression_download(self, pourcentage, texte_statut):
        """Met à jour la barre de progression et le statut (thread-safe via after)."""
        self.progress_bar.set(pourcentage / 100.0)
        self.label_progression.configure(text=texte_statut)
        self.label_statut.configure(
            text=texte_statut, text_color="#00bfff"
        )

    def _thread_telecharger(self, url):
        """Thread de téléchargement YouTube (ne bloque pas le GUI)."""
        try:
            import yt_dlp

            output_dir = os.path.dirname(os.path.abspath(__file__))
            output_template = os.path.join(output_dir, "youtube_video.%(ext)s")
            output_path = os.path.join(output_dir, "youtube_video.mp4")

            # Supprimer l'ancien fichier s'il existe
            if os.path.exists(output_path):
                os.remove(output_path)

            def hook_progression(d):
                """Callback appelé par yt-dlp pour signaler la progression."""
                if d["status"] == "downloading":
                    total = d.get("total_bytes") or d.get("total_bytes_estimate", 0)
                    downloaded = d.get("downloaded_bytes", 0)
                    speed = d.get("speed", 0) or 0

                    pct = (downloaded / total) * 100 if total > 0 else 0

                    if speed > 1_000_000:
                        vitesse_txt = f"{speed / 1_000_000:.1f} Mo/s"
                    elif speed > 1_000:
                        vitesse_txt = f"{speed / 1_000:.0f} Ko/s"
                    else:
                        vitesse_txt = "..."

                    taille_txt = f"{downloaded / 1_000_000:.1f}"
                    total_txt = f"{total / 1_000_000:.1f}" if total else "?"

                    statut = f"⬇️ {pct:.0f}% — {taille_txt}/{total_txt} Mo ({vitesse_txt})"
                    self.after(0, lambda p=pct, s=statut: self._maj_progression_download(p, s))

                elif d["status"] == "finished":
                    self.after(0, lambda: self._maj_progression_download(
                        100, "✅ Téléchargement terminé"
                    ))

            ydl_opts = {
                "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
                "outtmpl": output_template,
                "merge_output_format": "mp4",
                "quiet": True,
                "no_warnings": True,
                "noprogress": True,
                "progress_hooks": [hook_progression],
            }

            # Utiliser cookies.txt s'il existe (exporté via Connecter YouTube)
            cookies_path = os.path.join(output_dir, "cookies.txt")
            if os.path.exists(cookies_path):
                ydl_opts["cookiefile"] = cookies_path

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                titre = info.get("title", "Vidéo YouTube")

            # Revenir sur le thread principal pour mettre à jour le GUI
            self.after(0, lambda: self._youtube_termine(output_path, titre))

        except ImportError:
            self.after(0, lambda: self._youtube_erreur(
                "yt-dlp non installé. Lancez : pip install yt-dlp"
            ))
        except Exception as e:
            msg = str(e)
            self.after(0, lambda: self._youtube_erreur(msg))

    # ── Connexion YouTube via Selenium ──────────────────────────────

    def _connecter_youtube(self):
        """Ouvre Chrome pour se connecter à YouTube et exporter les cookies."""
        self.btn_connecter_yt.configure(
            state="disabled", text="🔄 Ouverture de Chrome..."
        )
        threading.Thread(
            target=self._thread_connecter_youtube, daemon=True
        ).start()

    def _thread_connecter_youtube(self):
        """Thread : ouvre Chrome, attend la connexion, exporte les cookies."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options
            from webdriver_manager.chrome import ChromeDriverManager

            self.after(0, lambda: self.label_statut.configure(
                text="🔄 Démarrage de Chrome...", text_color="#ffaa00"
            ))

            options = Options()
            options.add_argument("--start-maximized")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])

            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)

            driver.get("https://accounts.google.com/ServiceLogin?continue=https://www.youtube.com")

            self.after(0, lambda: self.label_statut.configure(
                text="🔑 Connectez-vous à YouTube puis fermez la fenêtre Chrome",
                text_color="#ffaa00"
            ))

            # Attendre que l'utilisateur ferme la fenêtre Chrome
            import time
            while True:
                try:
                    _ = driver.window_handles
                    time.sleep(1)
                except Exception:
                    break  # Fenêtre fermée

            # Exporter les cookies au format Netscape
            cookies = driver.get_cookies()
            output_dir = os.path.dirname(os.path.abspath(__file__))
            cookies_path = os.path.join(output_dir, "cookies.txt")

            with open(cookies_path, "w", encoding="utf-8") as f:
                f.write("# Netscape HTTP Cookie File\n")
                for c in cookies:
                    domain = c.get("domain", "")
                    flag = "TRUE" if domain.startswith(".") else "FALSE"
                    path = c.get("path", "/")
                    secure = "TRUE" if c.get("secure", False) else "FALSE"
                    expiry = str(int(c.get("expiry", 0)))
                    name = c.get("name", "")
                    value = c.get("value", "")
                    f.write(f"{domain}\t{flag}\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n")

            try:
                driver.quit()
            except Exception:
                pass

            self.after(0, self._connexion_youtube_ok)

        except Exception as e:
            msg = str(e)
            self.after(0, lambda: self._connexion_youtube_erreur(msg))

    def _connexion_youtube_ok(self):
        """Appelé après connexion YouTube réussie."""
        self.btn_connecter_yt.configure(
            state="normal", text="✅ YouTube connecté",
            fg_color="#2d6a4f", hover_color="#1b4332"
        )
        self.label_statut.configure(
            text="✅ Cookies YouTube sauvegardés !", text_color="#00ff88"
        )

    def _connexion_youtube_erreur(self, message):
        """Appelé en cas d'erreur de connexion YouTube."""
        self.btn_connecter_yt.configure(
            state="normal", text="🔑 Connecter YouTube",
            fg_color="#555", hover_color="#666"
        )
        self.label_statut.configure(
            text=f"❌ Erreur connexion : {message[:60]}", text_color="#ff4444"
        )

    def _youtube_termine(self, chemin, titre):
        """Appelé quand le téléchargement YouTube est terminé."""
        self.btn_telecharger.configure(state="normal", text="⬇️  Télécharger")
        self.label_statut.configure(
            text=f"✅ Téléchargé : {titre[:40]}", text_color="#00ff88"
        )
        self._charger_video(chemin)

    def _youtube_erreur(self, message):
        """Appelé en cas d'erreur de téléchargement YouTube."""
        self.btn_telecharger.configure(state="normal", text="⬇️  Télécharger")
        self.label_statut.configure(
            text=f"⚠️ {message[:60]}", text_color="#ff6b6b"
        )
        self.label_chemin.configure(text="Erreur de téléchargement", text_color="#ff6b6b")

    def _charger_video(self, chemin):
        """Charge une vidéo (locale ou téléchargée) et affiche la première frame."""
        self.video_path = chemin
        nom_fichier = os.path.basename(chemin)
        self.label_chemin.configure(text=nom_fichier, text_color="white")

        # Charger et afficher la première frame
        cap = cv2.VideoCapture(chemin)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                self.premiere_frame = frame.copy()
                self.points_ligne = []
                self.mode_calibration = True
                self._afficher_frame(frame)
                self.label_calibration.configure(
                    text="Cliquez sur 2 points de\nl'image pour définir la ligne.",
                    text_color="#ffcc00",
                )
                self.label_statut.configure(
                    text="📏 Mode calibration — Cliquez sur l'image"
                )

    def _on_canvas_resize(self, event):
        """Redessinée quand le canvas change de taille."""
        if self.premiere_frame is not None and not self.en_cours:
            frame = self.premiere_frame.copy()
            if len(self.points_ligne) >= 2:
                self._dessiner_ligne_sur_frame(frame)
            self._afficher_frame(frame)
        elif not self.en_cours:
            # Texte par défaut
            self.canvas_video.delete("all")
            w = event.width
            h = event.height
            self.canvas_video.create_text(
                w // 2, h // 2,
                text="Sélectionnez une vidéo\npour commencer",
                fill="gray50", font=("Segoe UI", 16),
                justify="center",
            )

    def _on_clic_canvas(self, event):
        """Gère les clics sur le canvas pour la calibration."""
        if not self.mode_calibration or self.premiere_frame is None:
            return
        if self.en_cours:
            return

        # Convertir les coordonnées canvas → coordonnées image réelles
        canvas_w = self.canvas_video.winfo_width()
        canvas_h = self.canvas_video.winfo_height()
        img_h, img_w = self.premiere_frame.shape[:2]

        # Calcul du ratio et de l'offset (car l'image est centrée)
        ratio = min(canvas_w / img_w, canvas_h / img_h)
        affiche_w = int(img_w * ratio)
        affiche_h = int(img_h * ratio)
        offset_x = (canvas_w - affiche_w) // 2
        offset_y = (canvas_h - affiche_h) // 2

        # Vérifier que le clic est dans l'image
        rel_x = event.x - offset_x
        rel_y = event.y - offset_y
        if rel_x < 0 or rel_y < 0 or rel_x >= affiche_w or rel_y >= affiche_h:
            return

        # Convertir en coordonnées de l'image originale
        x_reel = int(rel_x / ratio)
        y_reel = int(rel_y / ratio)

        self.points_ligne.append((x_reel, y_reel))

        if len(self.points_ligne) == 1:
            # Premier point placé
            self.label_calibration.configure(
                text=f"Point 1 : ({x_reel}, {y_reel})\n"
                     f"Cliquez pour placer le 2ᵉ point.",
                text_color="#ffcc00",
            )
            # Dessiner le premier point
            frame = self.premiere_frame.copy()
            cv2.circle(frame, (x_reel, y_reel), 8, (0, 255, 255), -1)
            self._afficher_frame(frame)

        elif len(self.points_ligne) >= 2:
            # Deux points → ligne définie
            p1, p2 = self.points_ligne[0], self.points_ligne[1]
            self.mode_calibration = False
            self.label_calibration.configure(
                text=f"✅ Ligne définie :\n"
                     f"  ({p1[0]}, {p1[1]}) → ({p2[0]}, {p2[1]})",
                text_color="#00ff88",
            )
            self.label_statut.configure(text="✅ Ligne calibrée — Prêt à démarrer")

            # Dessiner la ligne sur la frame d'aperçu
            frame = self.premiere_frame.copy()
            self._dessiner_ligne_sur_frame(frame)
            self._afficher_frame(frame)

    def _dessiner_ligne_sur_frame(self, frame):
        """Dessine la ligne de comptage et les points sur une frame."""
        if len(self.points_ligne) >= 2:
            p1, p2 = self.points_ligne[0], self.points_ligne[1]
            cv2.line(frame, p1, p2, (0, 255, 255), 3)
            cv2.circle(frame, p1, 8, (0, 255, 255), -1)
            cv2.circle(frame, p2, 8, (0, 255, 255), -1)

    def _reinitialiser_ligne(self):
        """Réinitialise les points de la ligne de comptage."""
        self.points_ligne = []
        self.mode_calibration = True
        self.label_calibration.configure(
            text="Cliquez sur 2 points de\nl'image pour définir la ligne.",
            text_color="#ffcc00",
        )
        if self.premiere_frame is not None and not self.en_cours:
            self._afficher_frame(self.premiere_frame.copy())
        self.label_statut.configure(
            text="📏 Mode calibration — Cliquez sur l'image"
        )

    def _afficher_frame(self, frame):
        """
        Affiche une frame OpenCV (BGR) sur le canvas tkinter.

        Args:
            frame: numpy array BGR.
        """
        # Convertir BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Redimensionner pour le canvas
        canvas_w = self.canvas_video.winfo_width()
        canvas_h = self.canvas_video.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            return

        img_h, img_w = frame_rgb.shape[:2]
        ratio = min(canvas_w / img_w, canvas_h / img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)

        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        img_pil = Image.fromarray(frame_resized)
        self.photo_image = ImageTk.PhotoImage(image=img_pil)

        # Centrer l'image dans le canvas
        x_center = canvas_w // 2
        y_center = canvas_h // 2

        self.canvas_video.delete("all")
        self.canvas_video.create_image(
            x_center, y_center, anchor="center", image=self.photo_image
        )

    # ─────────────────────────────────────────────────────────────
    # Analyse vidéo
    # ─────────────────────────────────────────────────────────────

    def _obtenir_classes_actives(self):
        """Retourne le dict des classes actives selon les cases cochées."""
        classes = {}
        if self.var_voiture.get():
            classes[2] = "voiture"
        if self.var_moto.get():
            classes[3] = "moto"
        if self.var_bus.get():
            classes[5] = "bus"
        if self.var_camion.get():
            classes[7] = "camion"
        return classes

    def _demarrer_analyse(self):
        """Démarre l'analyse vidéo dans un thread séparé."""
        if not self.video_path:
            self.label_statut.configure(
                text="⚠️ Sélectionnez d'abord une vidéo !", text_color="#ff6b6b"
            )
            return

        if len(self.points_ligne) < 2:
            self.label_statut.configure(
                text="⚠️ Calibrez la ligne d'abord !", text_color="#ff6b6b"
            )
            return

        classes = self._obtenir_classes_actives()
        if not classes:
            self.label_statut.configure(
                text="⚠️ Sélectionnez au moins un type !", text_color="#ff6b6b"
            )
            return

        # Extraire les coordonnées de la ligne
        p1, p2 = self.points_ligne[0], self.points_ligne[1]

        # Utiliser le Y moyen des deux points pour la ligne horizontale
        ligne_y = (p1[1] + p2[1]) // 2
        ligne_x1 = min(p1[0], p2[0])
        ligne_x2 = max(p1[0], p2[0])

        # Créer le compteur
        self.counter = VehicleCounter(
            video_path=self.video_path,
            confidence=self.var_confiance.get(),
            ligne_y=ligne_y,
            ligne_x1=ligne_x1,
            ligne_x2=ligne_x2,
            classes_actives=classes,
        )

        # Initialiser
        try:
            self.counter.initialiser()
        except FileNotFoundError as e:
            self.label_statut.configure(
                text=f"⚠️ {e}", text_color="#ff6b6b"
            )
            return

        # Mettre à jour l'interface
        self.en_cours = True
        self.btn_demarrer.configure(state="disabled")
        self.btn_arreter.configure(state="normal")
        self.label_statut.configure(
            text="🔄 Analyse en cours...", text_color="#00bfff"
        )

        # Lancer la boucle de traitement
        self._boucle_analyse()

    def _boucle_analyse(self):
        """
        Boucle de traitement frame par frame.
        Utilise after() pour rester dans le thread principal (GUI).
        """
        if not self.en_cours or self.counter is None:
            return

        frame, passage = self.counter.traiter_frame_suivante()

        if frame is None:
            # Fin de la vidéo
            self._fin_analyse()
            return

        # Afficher la frame
        self._afficher_frame(frame)

        # Mettre à jour les statistiques
        self._maj_statistiques()

        # Planifier la prochaine frame (1ms de délai pour que le GUI respire)
        self.after(1, self._boucle_analyse)

    def _arreter_analyse(self):
        """Arrête l'analyse en cours."""
        self.en_cours = False
        if self.counter:
            self.counter.arreter()
        self._fin_analyse()

    def _fin_analyse(self):
        """Finalise l'analyse et remet l'interface en état initial."""
        self.en_cours = False

        if self.counter:
            stats = self.counter.obtenir_stats()
            self.counter.arreter()
            self.label_statut.configure(
                text=f"✅ Terminé — {stats['total']} véhicules détectés",
                text_color="#00ff88",
            )
        else:
            self.label_statut.configure(
                text="✅ Analyse terminée", text_color="#00ff88"
            )

        self.btn_demarrer.configure(state="normal")
        self.btn_arreter.configure(state="disabled")

        # Réafficher la première frame
        if self.premiere_frame is not None:
            frame = self.premiere_frame.copy()
            if len(self.points_ligne) >= 2:
                self._dessiner_ligne_sur_frame(frame)
            self._afficher_frame(frame)

    # ─────────────────────────────────────────────────────────────
    # Mise à jour des statistiques
    # ─────────────────────────────────────────────────────────────

    def _maj_statistiques(self):
        """Met à jour tous les labels de statistiques."""
        if not self.counter:
            return

        stats = self.counter.obtenir_stats()

        self.label_total.configure(text=f"Total : {stats['total']}")
        self.label_directions.configure(
            text=f"↑ Montant : {stats['montant']}    "
                 f"↓ Descendant : {stats['descendant']}"
        )

        par_type = stats["par_type"]
        self.label_voiture.configure(
            text=f"🚗 Voiture : {par_type.get('voiture', 0)}"
        )
        self.label_camion.configure(
            text=f"🚛 Camion  : {par_type.get('camion', 0)}"
        )
        self.label_bus.configure(
            text=f"🚌 Bus     : {par_type.get('bus', 0)}"
        )
        self.label_moto.configure(
            text=f"🏍️ Moto    : {par_type.get('moto', 0)}"
        )

        # Progression
        progression = stats["progression"] / 100.0
        self.progress_bar.set(min(progression, 1.0))

    # ─────────────────────────────────────────────────────────────
    # Utilitaires
    # ─────────────────────────────────────────────────────────────

    def _on_confiance_change(self, value):
        """Met à jour le label de confiance quand le slider bouge."""
        self.label_confiance_val.configure(text=f"{value:.2f}")

    def _ouvrir_csv(self):
        """Ouvre le fichier CSV dans l'application par défaut."""
        csv_path = "traffic_data.csv"
        if os.path.exists(csv_path):
            os.startfile(csv_path)
        else:
            self.label_statut.configure(
                text="⚠️ Pas de fichier CSV trouvé", text_color="#ff6b6b"
            )

    def _quitter(self):
        """Ferme proprement l'application."""
        self.en_cours = False
        if self.counter:
            self.counter.arreter()
        self.destroy()


# =============================================================================
# ──── POINT D'ENTRÉE ────
# =============================================================================

if __name__ == "__main__":
    app = App()
    app.mainloop()

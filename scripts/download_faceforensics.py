# scripts/download_faceforensics.py
"""
Script simplifié pour télécharger FaceForensics++
Version adaptée pour Windows/Mac/Linux
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
import requests
import zipfile
import gdown
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def check_prerequisites():
    """Vérifie les prérequis"""
    print("🔍 Vérification des prérequis...")

    # Vérifier Python
    python_version = sys.version_info
    if python_version.major < 3 or python_version.minor < 8:
        print("❌ Python 3.8+ requis")
        return False

    # Vérifier l'espace disque
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (2 ** 30)

    print(f"💾 Espace disponible : {free_gb} GB")

    if free_gb < 20:
        print("⚠️  Espace insuffisant. 20GB minimum recommandé")
        return False

    print("✅ Tous les prérequis sont satisfaits")
    return True


def download_with_gdown():
    """
    Télécharge un SUBSET léger via Google Drive
    C'est la méthode la plus simple !
    """
    print("\n📥 Téléchargement d'un subset léger...")

    # Créer la structure des dossiers
    raw_dir = Path("data/raw/faceforensicspp")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # URLs Google Drive (exemples publics - À ADAPTER)
    # Ces URLs sont fictives, il faut les vraies URLs

    print("⚠️  Les URLs Google Drive nécessitent un accès")
    print("💡 Pour l'instant, je vais te montrer comment")
    print("   utiliser le script officiel ci-dessous")

    return False


def download_official_subset():
    """
    Utilise le script officiel de FaceForensics
    pour télécharger un petit subset
    """
    print("\n🎯 Utilisation du script officiel...")

    # 1. Clone le repository officiel
    repo_dir = Path("external/faceforensics")

    if not repo_dir.exists():
        print("📥 Clonage du repo FaceForensics...")
        subprocess.run([
            "git", "clone", "https://github.com/ondyari/FaceForensics.git",
            str(repo_dir)
        ], check=True)

    # 2. Naviguer dans le dossier
    os.chdir(repo_dir)

    # 3. Installer les dépendances
    print("📦 Installation des dépendances...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ], check=True)

    # 4. Télécharger un TRÈS PETIT subset
    print("🎬 Téléchargement de 5 vidéos seulement...")

    command = [
        sys.executable, "download.py",
        "--dataset", "FaceForensics++",
        "--compression", "c23",  # Qualité moyenne
        "--methods", "original", "DeepFakes",  # Seulement DeepFakes
        "--num_videos", "5",  # SEULEMENT 5 VIDÉOS
        "--videos"
    ]

    try:
        subprocess.run(command, check=True)
        print("✅ Téléchargement réussi !")

        # Retourner au dossier original
        os.chdir("..")

        # Organiser les fichiers dans notre structure
        organize_downloaded_files()

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du téléchargement : {e}")
        os.chdir("..")
        return False


def organize_downloaded_files():
    """Organise les fichiers téléchargés dans notre structure"""
    print("\n📁 Organisation des fichiers...")

    source_dir = Path("external/faceforensics/dataset/FaceForensics++")
    dest_dir = Path("data/raw/faceforensicspp")

    if not source_dir.exists():
        print("❌ Dossier source introuvable")
        return

    # Créer la structure de destination
    (dest_dir / "original_sequences/youtube/c23/videos").mkdir(parents=True, exist_ok=True)
    (dest_dir / "manipulated_sequences/DeepFakes/c23/videos").mkdir(parents=True, exist_ok=True)

    # Copier les fichiers
    import shutil

    # Original videos
    original_source = source_dir / "original_sequences/youtube/c23/videos"
    if original_source.exists():
        for video_file in original_source.glob("*.mp4"):
            shutil.copy2(video_file, dest_dir / "original_sequences/youtube/c23/videos")
            print(f"  📹 Copié : {video_file.name}")

    # DeepFakes videos
    fake_source = source_dir / "manipulated_sequences/DeepFakes/c23/videos"
    if fake_source.exists():
        for video_file in fake_source.glob("*.mp4"):
            shutil.copy2(video_file, dest_dir / "manipulated_sequences/DeepFakes/c23/videos")
            print(f"  🎭 Copié : {video_file.name}")

    print(f"\n✅ Fichiers organisés dans : {dest_dir}")


def create_dummy_videos():
    """
    Crée des vidéos factices pour tester le pipeline
    Sans télécharger 100GB !
    """
    print("\n🎥 Création de vidéos factices pour test...")

    raw_dir = Path("data/raw/faceforensicspp")

    # Structure des dossiers
    (raw_dir / "original_sequences/youtube/c23/videos").mkdir(parents=True, exist_ok=True)
    (raw_dir / "manipulated_sequences/DeepFakes/c23/videos").mkdir(parents=True, exist_ok=True)

    # Créer quelques fichiers .txt pour simuler (plus tard tu mettras des vraies vidéos)
    for i in range(5):
        # "Vidéos" réelles
        real_file = raw_dir / "original_sequences/youtube/c23/videos" / f"real_video_{i:03d}.txt"
        real_file.write_text(f"Ceci simule une vidéo réelle #{i}\nPlaceholder pour test")

        # "Vidéos" fake
        fake_file = raw_dir / "manipulated_sequences/DeepFakes/c23/videos" / f"fake_video_{i:03d}.txt"
        fake_file.write_text(f"Ceci simule un deepfake #{i}\nPlaceholder pour test")

    print("✅ 10 fichiers factices créés")
    print("💡 REMPLACE-LES plus tard par de vraies vidéos .mp4")


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Télécharger FaceForensics++")
    parser.add_argument("--mode", choices=["light", "full", "dummy"],
                        default="light", help="Mode de téléchargement")
    parser.add_argument("--videos", type=int, default=5,
                        help="Nombre de vidéos par classe")

    args = parser.parse_args()

    print("=" * 60)
    print("📥 TÉLÉCHARGEMENT FACE FORENSICS++")
    print("=" * 60)

    # Vérifier les prérequis
    if not check_prerequisites():
        print("❌ Prérequis non satisfaits")
        return

    # Choisir la méthode
    if args.mode == "dummy":
        create_dummy_videos()
        return

    elif args.mode == "light":
        print(f"\n🎯 Mode LÉGER sélectionné")
        print(f"   • {args.videos} vidéos réelles")
        print(f"   • {args.videos} vidéos fake")
        print(f"   • ~{args.videos * 2 * 100}MB estimés")

        # Essayer le téléchargement officiel
        success = download_official_subset()

        if not success:
            print("\n⚠️  Fallback : création de données factices")
            create_dummy_videos()

    elif args.mode == "full":
        print("\n⚠️  ATTENTION : Mode COMPLET sélectionné")
        print("   Cela va télécharger ~100GB de données")
        print("   Cela peut prendre plusieurs heures")

        response = input("\n❓ Continuer ? (oui/non): ")
        if response.lower() != "oui":
            print("❌ Annulé")
            return

        # Ici tu mettrais le téléchargement complet
        print("🔧 Implémentation du mode complet à venir...")

    print("\n" + "=" * 60)
    print("✅ OPÉRATION TERMINÉE")
    print("=" * 60)

    # Afficher la structure créée
    print("\n📁 Structure créée :")
    for root, dirs, files in os.walk("data/raw"):
        level = root.replace("data/raw", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")

        subindent = " " * 2 * (level + 1)
        for file in files[:3]:  # Afficher max 3 fichiers
            print(f"{subindent}{file}")
        if len(files) > 3:
            print(f"{subindent}... et {len(files) - 3} autres")


if __name__ == "__main__":
    main()
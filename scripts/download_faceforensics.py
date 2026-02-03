"""
FaceForensics++ ULTRA LIGHT DOWNLOADER
------------------------------------
✔ Télécharge de VRAIES vidéos
✔ Limité à quelques fichiers (PC faible)
✔ Fallback automatique en dummy
✔ Compatible Windows / Linux / Mac
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse


# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = Path("data/raw/faceforensicspp")
REAL_DIR = BASE_DIR / "real"
FAKE_DIR = BASE_DIR / "fake"
REPO_DIR = Path("external/faceforensics")
MIN_FREE_GB = 1


# ==========================================================
# CHECK SYSTEM
# ==========================================================
def check_system():
    print("🔍 Vérification système...")

    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ requis")
        return False

    total, used, free = shutil.disk_usage(Path.cwd().anchor)
    free_gb = free // (2 ** 30)

    print(f"💾 Espace libre : {free_gb} GB")

    if free_gb < MIN_FREE_GB:
        print("❌ Pas assez d'espace disque")
        return False

    print("✅ Système OK")
    return True


# ==========================================================
# DUMMY MODE
# ==========================================================
def create_dummy(n):
    print("\n⚠ MODE DUMMY (aucune vraie vidéo)")

    REAL_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        (REAL_DIR / f"real_{i}.txt").write_text("FAKE REAL VIDEO")
        (FAKE_DIR / f"fake_{i}.txt").write_text("FAKE DEEPFAKE VIDEO")

    print(f"✅ {n} faux réels + {n} faux deepfakes créés")


# ==========================================================
# REAL DOWNLOAD (LIGHT)
# ==========================================================
def download_real_videos(n):
    print("\n🎥 Téléchargement de vraies vidéos (MODE LÉGER)")

    REAL_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if not REPO_DIR.exists():
            print("📥 Clonage FaceForensics...")
            subprocess.run(
                ["git", "clone", "https://github.com/ondyari/FaceForensics.git", str(REPO_DIR)],
                check=True
            )

        os.chdir(REPO_DIR)

        print("📦 Installation dépendances minimales...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True
        )

        print("⬇ Téléchargement subset réel + deepfake...")
        subprocess.run(
            [
                sys.executable, "download.py",
                "--dataset", "FaceForensics++",
                "--compression", "c23",
                "--methods", "original", "DeepFakes",
                "--num_videos", str(n),
                "--videos"
            ],
            check=True
        )

        os.chdir("..")

    except Exception as e:
        print("❌ Échec du téléchargement :", e)
        os.chdir("../../")
        create_dummy(n)
        return

    organize_dataset()


# ==========================================================
# ORGANISATION
# ==========================================================
def organize_dataset():
    print("\n📁 Organisation des vidéos...")

    src = REPO_DIR / "dataset/FaceForensics++"

    real_src = src / "original_sequences/youtube/c23/videos"
    fake_src = src / "manipulated_sequences/DeepFakes/c23/videos"

    for f in real_src.glob("*.mp4"):
        shutil.copy2(f, REAL_DIR)

    for f in fake_src.glob("*.mp4"):
        shutil.copy2(f, FAKE_DIR)

    print("✅ Vidéos prêtes")
    print(f"📂 REAL : {len(list(REAL_DIR.glob('*.mp4')))}")
    print(f"📂 FAKE : {len(list(FAKE_DIR.glob('*.mp4')))}")


# ==========================================================
# MAIN
# ==========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=int, default=5)
    parser.add_argument("--mode", choices=["real", "dummy"], default="real")
    args = parser.parse_args()

    print("=" * 60)
    print("🧠 FACEFORENSICS++ — ULTRA LIGHT VERSION")
    print("=" * 60)

    if not check_system():
        return

    if args.mode == "dummy":
        create_dummy(args.videos)
    else:
        download_real_videos(args.videos)

    print("\n🚀 TERMINÉ — prêt pour l'entraînement IA")


if __name__ == "__main__":
    main()

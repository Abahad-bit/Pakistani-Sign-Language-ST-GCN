import os
import sys

os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding='utf-8')
from src.train import train

if __name__ == "__main__":
    train("dataset/PakistanSignLanguageDataset")
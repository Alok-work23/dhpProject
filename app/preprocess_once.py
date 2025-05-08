# app/preprocess_once.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.data_loader import preprocess_and_save



if __name__ == "__main__":
    preprocess_and_save()

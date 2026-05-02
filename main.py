import sys
import os

# Add current directory to PYTHONPATH so imports work easily
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()

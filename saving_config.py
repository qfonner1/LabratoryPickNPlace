import os
import time

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR = os.path.join("outputs", f"run_{RUN_ID}")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Counter for plot filenames
PLOT_COUNTER = 0
import os
import logging

from pathlib import Path

# Logging String
logging.basicConfig(level = logging.INFO, format = '[%(asctime)s]: %(message)s:')

list_of_files = [
    ".github/workflows/.gitkeep",
    f"datasets/__init__.py",
    f"utils/__init__.py",
    f"pruning/__init__.py",
    f"evaluate/__init__.py",
    f"metrics/__init__.py",
    "requirements.txt"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")
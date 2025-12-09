import subprocess
import sys
from pathlib import Path

def main():
    """Launch the lettuce-ui Marimo app."""
    ui_path = Path(__file__).parent/ "ui.py"
    
    # Use subprocess to call marimo run
    result = subprocess.run(
        ["marimo", "run", str(ui_path)] + sys.argv[1:],
        check=False
    )
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()

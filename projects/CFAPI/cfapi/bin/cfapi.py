import sys
from pathlib import Path

here = Path(__file__).resolve().parent
api = here.parent / "src"

if str(api) not in sys.path:
    sys.path.insert(0, str(api))

from cfapi.cfapi_interface import main

if __name__ == "__main__":
    main(script="cfapi.py")

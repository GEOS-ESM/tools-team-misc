import sys
import importlib.util
from pathlib import Path

here = Path(__file__).parent.resolve()
CFAPI = here.parent / "cfapi" / "src"
required = {"cfapi": CFAPI}


def ensure_path(path):
    path = Path(path).resolve()
    if not path.exists():
        return
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def mod_spec(mod):
    return importlib.util.find_spec(mod)


for mod, default_path in required.items():
    spec = mod_spec(mod)
    if spec is None:
        ensure_path(default_path)
        spec = mod_spec(mod)
    if spec is None:
        raise RuntimeError(
            f"Required module '{mod}' not available " f"(tried adding {default_path})"
        )
    print(f"{mod} found at {spec.origin}")

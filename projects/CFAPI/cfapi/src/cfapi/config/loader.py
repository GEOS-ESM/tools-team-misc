from importlib.resources import files
import yaml

def load_yaml(name):
    path = files("cfapi.config").joinpath(name)
    with path.open() as f:
        return yaml.safe_load(f)
import yaml

from pathlib import Path

def load_config():
    """ Load config.yml """
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "config.yml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config.yml : {e}")
        return None
    
    abs_paths = {}
    for key, rel_path in config["paths"].items():
        abs_paths[key] = (root_dir / rel_path).resolve()
    config["paths"] = abs_paths

    return config
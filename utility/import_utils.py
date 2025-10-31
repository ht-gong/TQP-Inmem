import json
from pathlib import Path
import sys
import importlib.util

def load_vortex_lib():
    # Load configuration
    config_path = "config.json" 
    with open(config_path, "r") as f:
        config = json.load(f)

    package_path = config.get("vortex_build_path")  

    if not package_path:
        raise ValueError("'vortex_build_path' must be defined in config.json")

    # Ensure the package path is absolute
    package_path = str(Path(package_path).resolve())

    # Add package path to sys.path
    if package_path not in sys.path:
        sys.path.insert(0, package_path)
    return None
    # Dynamically import the module
    try:
        vortex = importlib.import_module("vortex")
        print(f"Successfully imported from {package_path}")
    except ModuleNotFoundError as e:
        print(f"Error importing {package_path}: {e}")
        
    return vortex

def load_hipTQP_lib():

    # Load configuration
    config_path = "config.json" 
    with open(config_path, "r") as f:
        config = json.load(f)

    package_path = config.get("vortex_build_path")  

    if not package_path:
        raise ValueError("'vortex_build_path' must be defined in config.json")

    # Ensure the package path is absolute
    package_path = str(Path(package_path).resolve())

    # Add package path to sys.path
    if package_path not in sys.path:
        sys.path.insert(0, package_path)

    # Dynamically import the module
    try:
        hipTQP = importlib.import_module("hipTQP")
        print(f"Successfully imported from {package_path}")
    except ModuleNotFoundError as e:
        print(f"Error importing {package_path}: {e}")
    
    return None
    # return hipTQP
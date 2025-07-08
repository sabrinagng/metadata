import json
import sys
from pathlib import Path

def probe_json(json_path):
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            return

    print(f"Top-level type: {type(data).__name__}")

    # Probe deeper levels (up to 2 levels deep)
    def probe_levels(obj, level=1, max_level=2):
        indent = "  " * level
        if isinstance(obj, dict):
            print(f"{indent}Level {level} keys: {list(obj.keys())}")
            if level < max_level:
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        print(f"{indent}Key '{k}': {type(v).__name__}")
                        probe_levels(v, level + 1, max_level)
        elif isinstance(obj, list):
            print(f"{indent}Level {level} list length: {len(obj)}")
            if obj and isinstance(obj[0], (dict, list)):
                print(f"{indent}First element type: {type(obj[0]).__name__}")
                if level < max_level:
                    probe_levels(obj[0], level + 1, max_level)

    probe_levels(data)
    if isinstance(data, dict):
        print(f"Top-level keys: {list(data.keys())}")
    elif isinstance(data, list):
        print(f"Top-level list length: {len(data)}")
        if data:
            print(f"First element type: {type(data[0]).__name__}")
    else:
        print("Top-level JSON is neither dict nor list.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python json_probe.py <path_to_json_file>")
    else:
        probe_json(sys.argv[1])
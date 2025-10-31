#!/usr/bin/env python3
import os
import json
import glob
import argparse

def load_json(filepath):
    """Load a JSON file and return its dict, or None on error."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  Skipping {filepath}: cannot read/parse JSON ({e})")
        return None

def compute_proportions_for_pair(json1, json2, filename):
    """
    Given two dicts (json1, json2) with (ideally) identical keys,
    compute for each key: json1[key] / json2[key], or None if not numeric or division by zero.
    Prints a header for the file pair and then each key's proportion.
    """
    keys1 = set(json1.keys())
    keys2 = set(json2.keys())
    if keys1 != keys2:
        print(f"  Warning: Key-sets differ in file '{filename}':")
        only1 = sorted(keys1 - keys2)
        only2 = sorted(keys2 - keys1)
        if only1:
            print(f"    Keys only in dir1/{filename}: {only1}")
        if only2:
            print(f"    Keys only in dir2/{filename}: {only2}")
        # We'll proceed with the intersection of keys1∩keys2
    common_keys = sorted(keys1 & keys2)

    print(f"\n--- Proportions for '{filename}' (dir1 / dir2) ---")
    for key in common_keys:
        v1 = json1.get(key)['total']
        v2 = json2.get(key)['total']

        # Only compute if both are numeric
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            if v2 == 0:
                prop = None
            else:
                prop = v1 / v2
        else:
            prop = None

        if prop is None:
            print(f"{key:30} :  cannot compute (non-numeric or division by zero)")
        else:
            print(f"{key:30} : {prop:.6f}")

def main():
    parser = argparse.ArgumentParser(
        description="For every JSON file in dir1, find the JSON with the same filename in dir2, "
                    "then for each key compute (value_in_dir1 / value_in_dir2)."
    )
    parser.add_argument(
        'dir1',
        help='First directory containing JSON files.'
    )
    parser.add_argument(
        'dir2',
        help='Second directory containing JSON files (must have matching filenames).'
    )
    args = parser.parse_args()

    dir1 = args.dir1
    dir2 = args.dir2

    # Glob all JSON files in dir1
    pattern1 = os.path.join(dir1, '*.json')
    files1 = sorted(glob.glob(pattern1))

    if not files1:
        print(f"No JSON files found in {dir1}")
        return

    # For each JSON in dir1, look for a matching name in dir2
    for filepath1 in files1:
        filename = os.path.basename(filepath1)
        filepath2 = os.path.join(dir2, filename)

        if not os.path.isfile(filepath2):
            print(f"  Skipping '{filename}': no matching file in {dir2}")
            continue

        json1 = load_json(filepath1)
        json2 = load_json(filepath2)
        if json1 is None or json2 is None:
            continue

        compute_proportions_for_pair(json1, json2, filename)

    # Optionally, warn about files in dir2 that had no counterpart in dir1
    pattern2 = os.path.join(dir2, '*.json')
    files2 = sorted(glob.glob(pattern2))
    extra_in_dir2 = [
        os.path.basename(f2) for f2 in files2
        if os.path.basename(f2) not in {os.path.basename(f1) for f1 in files1}
    ]
    if extra_in_dir2:
        print("\n The following files exist in dir2 but not in dir1:")
        for fn in extra_in_dir2:
            print(f"    {fn}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Parse Micro-Manager .cfg files and print detected power properties.

For each channel config preset, shows which light-source device/property
controls the LED power (e.g. Spectra|Cyan_Level).

Usage:
    python scripts/show_power_properties.py path/to/*.cfg
    python scripts/show_power_properties.py /path/to/pertzlab_mic_configs/micromanager/
"""

from __future__ import annotations

import sys
from pathlib import Path


def parse_cfg(path: str | Path) -> tuple[
    dict[str, set[str]],            # device → {property_names}
    dict[str, list[str]],           # group → [config_names]
    dict[tuple[str, str], list[tuple[str, str, str]]],  # (group, config) → [(dev, prop, val)]
]:
    """Parse a Micro-Manager .cfg file into devices, groups, and config data."""
    devices: dict[str, set[str]] = {}
    groups: dict[str, list[str]] = {}
    config_data: dict[tuple[str, str], list[tuple[str, str, str]]] = {}

    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")

        if parts[0] == "Property" and len(parts) >= 4:
            dev, prop = parts[1], parts[2]
            devices.setdefault(dev, set()).add(prop)

        elif parts[0] == "ConfigGroup" and len(parts) >= 6:
            group, config, dev, prop, val = parts[1], parts[2], parts[3], parts[4], parts[5]
            groups.setdefault(group, [])
            if config not in groups[group]:
                groups[group].append(config)
            config_data.setdefault((group, config), []).append((dev, prop, val))
            # Also register the device property (Level props are often only here)
            devices.setdefault(dev, set()).add(prop)

    return devices, groups, config_data


def detect_power_properties_per_group(
    devices: dict[str, set[str]],
    groups: dict[str, list[str]],
    config_data: dict[tuple[str, str], list[tuple[str, str, str]]],
) -> dict[str, dict[str, tuple[str, str, str]]]:
    """Auto-detect per-channel power properties, grouped by config group.

    Returns:
        {group: {config_name: (device, property, matched_value)}}
    """
    # Find devices with *_Level properties
    level_lookup: dict[str, tuple[str, str]] = {}
    for dev, props in devices.items():
        for prop in props:
            if prop.endswith("_Level"):
                color = prop[:-6].lower()
                level_lookup[color] = (dev, prop)

    if not level_lookup:
        return {}

    result: dict[str, dict[str, tuple[str, str, str]]] = {}
    for group, configs in groups.items():
        if group == "System":
            continue  # skip System group (Startup/Shutdown)
        group_result: dict[str, tuple[str, str, str]] = {}
        for config_name in configs:
            for dev, prop, val in config_data.get((group, config_name), []):
                val_lower = val.lower()
                for color, (level_dev, level_prop) in level_lookup.items():
                    if val_lower == color or (len(color) >= 3 and val_lower.startswith(color)):
                        group_result[config_name] = (level_dev, level_prop, val)
                        break
                if config_name in group_result:
                    break
        if group_result:
            result[group] = group_result

    return result


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.cfg")))
        else:
            paths.append(p)

    for cfg_path in paths:
        if not cfg_path.exists():
            print(f"  [not found: {cfg_path}]")
            continue

        devices, groups, config_data = parse_cfg(cfg_path)
        per_group = detect_power_properties_per_group(devices, groups, config_data)

        print(f"\n{'=' * 60}")
        print(f"  {cfg_path.name}")
        print(f"{'=' * 60}")

        if not per_group:
            print("  (no power properties detected)")
            continue

        for group, mapping in sorted(per_group.items()):
            print(f"\n  [{group}]")
            max_config = max(len(c) for c in mapping)
            for config_name, (dev, prop, matched) in sorted(mapping.items()):
                print(f"    {config_name:<{max_config}}  ->  {dev}|{prop}  (matched '{matched}')")


if __name__ == "__main__":
    main()

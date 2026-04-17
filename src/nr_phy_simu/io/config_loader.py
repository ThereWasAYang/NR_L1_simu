from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

from nr_phy_simu.config import SimulationConfig, config_path


def load_simulation_config(path: str | Path) -> SimulationConfig:
    resolved = config_path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        data = json.loads(resolved.read_text())
    elif suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(resolved.read_text())
    elif suffix == ".xml":
        data = _xml_to_mapping(ET.parse(resolved).getroot())
    else:
        raise ValueError(f"Unsupported config format: {resolved.suffix}")
    _resolve_relative_paths(data, resolved.parent)
    return SimulationConfig.from_mapping(data)


def _resolve_relative_paths(data: dict, base_dir: Path) -> None:
    waveform_input = data.get("waveform_input")
    if isinstance(waveform_input, dict):
        waveform_path = waveform_input.get("waveform_path")
        if isinstance(waveform_path, str) and waveform_path.strip() != "":
            waveform_input["waveform_path"] = str(_resolve_path_string(waveform_path, base_dir))

    simulation = data.get("simulation")
    if isinstance(simulation, dict):
        result_output_path = simulation.get("result_output_path")
        if isinstance(result_output_path, str) and result_output_path.strip() != "":
            simulation["result_output_path"] = str(_resolve_path_string(result_output_path, base_dir))

    channel = data.get("channel")
    if isinstance(channel, dict):
        params = channel.get("params")
        if isinstance(params, dict):
            frequency_response_path = params.get("frequency_response_path")
            if isinstance(frequency_response_path, str) and frequency_response_path.strip() != "":
                params["frequency_response_path"] = str(_resolve_path_string(frequency_response_path, base_dir))


def _resolve_path_string(path_value: str, base_dir: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def _xml_to_mapping(element: ET.Element):
    children = list(element)
    if not children:
        return _parse_text_value(element.text)

    grouped: dict[str, list] = {}
    for child in children:
        grouped.setdefault(child.tag, []).append(_xml_to_mapping(child))

    mapping = {}
    for key, values in grouped.items():
        mapping[key] = values[0] if len(values) == 1 else values
    return mapping


def _parse_text_value(text: str | None):
    if text is None:
        return None
    stripped = text.strip()
    if stripped == "":
        return None
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return stripped

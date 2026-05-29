from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

from nr_phy_simu.config import SimulationConfig, config_path

_TEXT_FILE_READ_ENCODING = "utf-8-sig"


def load_simulation_config(path: str | Path) -> SimulationConfig:
    """Load a simulation config file using a Windows-safe UTF-8 reader.

    Args:
        path: Config file path. Supported suffixes are JSON, YAML/YML and XML.

    Returns:
        Parsed :class:`SimulationConfig` instance.
    """
    resolved = config_path(path)
    data = _load_mapping_file(resolved)
    _resolve_relative_paths(data, resolved.parent)
    return SimulationConfig.from_mapping(data)


def _load_mapping_file(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding=_TEXT_FILE_READ_ENCODING))
    elif suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding=_TEXT_FILE_READ_ENCODING))
    elif suffix == ".xml":
        with path.open("r", encoding=_TEXT_FILE_READ_ENCODING) as handle:
            data = _xml_to_mapping(ET.parse(handle).getroot())
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {path}")
    return data


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
        config_path_value = channel.get("config_path")
        if isinstance(config_path_value, str) and config_path_value.strip() != "":
            channel_config_path = _resolve_path_string(config_path_value, base_dir)
            inline_channel = dict(channel)
            external_channel = _load_mapping_file(channel_config_path)
            _resolve_channel_relative_paths(external_channel, channel_config_path.parent)
            inline_channel["config_path"] = str(channel_config_path)
            _resolve_channel_relative_paths(inline_channel, base_dir)
            channel.clear()
            channel.update(_merge_channel_config(external_channel, inline_channel))
        else:
            _resolve_channel_relative_paths(channel, base_dir)

    interference = data.get("interference")
    if isinstance(interference, dict):
        _resolve_interference_relative_paths(interference, base_dir)


def _resolve_channel_relative_paths(channel: dict, base_dir: Path) -> None:
    params = channel.get("params")
    if isinstance(params, dict):
        _resolve_channel_params_relative_paths(params, base_dir)


def _resolve_interference_relative_paths(interference: dict, base_dir: Path) -> None:
    sources = interference.get("sources", []) or []
    if not isinstance(sources, list):
        return
    for source in sources:
        if not isinstance(source, dict):
            continue
        config_path_value = source.get("config_path")
        if isinstance(config_path_value, str) and config_path_value.strip() != "":
            source["config_path"] = str(_resolve_path_string(config_path_value, base_dir))
        channel_params = source.get("channel_params")
        if isinstance(channel_params, dict):
            _resolve_channel_params_relative_paths(channel_params, base_dir)


def _resolve_channel_params_relative_paths(params: dict, base_dir: Path) -> None:
    frequency_response_path = params.get("frequency_response_path")
    if isinstance(frequency_response_path, str) and frequency_response_path.strip() != "":
        params["frequency_response_path"] = str(_resolve_path_string(frequency_response_path, base_dir))


def _merge_channel_config(external: dict, inline: dict) -> dict:
    merged = dict(external)
    for key, value in inline.items():
        if key == "params" and isinstance(value, dict) and isinstance(merged.get("params"), dict):
            params = dict(merged["params"])
            params.update(value)
            merged["params"] = params
        elif key == "geometry" and isinstance(value, dict) and isinstance(merged.get("geometry"), dict):
            geometry = dict(merged["geometry"])
            geometry.update(value)
            merged["geometry"] = geometry
        else:
            merged[key] = value
    return merged


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

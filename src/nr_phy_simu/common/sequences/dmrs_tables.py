from __future__ import annotations


def _single_symbol_offsets(channel_type: str, mapping: str, duration: int, add_pos: int) -> tuple[int, ...] | None:
    """Return 38.211 single-symbol DM-RS table entries; zero denotes ``l0``."""
    if mapping == "A":
        if duration < (4 if channel_type == "PUSCH" else 3):
            return None
        if duration <= 7 or add_pos == 0:
            return (0,)
        if duration in (8, 9):
            return (0, 7)
        if duration in (10, 11):
            return (0, 9) if add_pos == 1 else (0, 6, 9)
        if duration == 12:
            if add_pos == 1:
                return (0, 9)
            if add_pos == 2:
                return (0, 6, 9)
            return (0, 5, 8, 11)
        if add_pos == 1:
            return (0, 11)
        if add_pos == 2:
            return (0, 7, 11)
        return (0, 5, 8, 11)

    if channel_type == "PUSCH":
        if duration <= 4 or add_pos == 0:
            return (0,)
        if duration <= 7:
            return (0, 4)
        if duration <= 9:
            return (0, 6) if add_pos == 1 else (0, 3, 6)
        if duration <= 11:
            if add_pos == 1:
                return (0, 8)
            return (0, 4, 8) if add_pos == 2 else (0, 3, 6, 9)
        if add_pos == 1:
            return (0, 10)
        return (0, 5, 10) if add_pos == 2 else (0, 3, 6, 9)

    if duration == 14:
        return None
    if duration <= 4 or add_pos == 0:
        return (0,)
    if duration <= 7:
        return (0, 4)
    if duration == 8:
        return (0, 6) if add_pos == 1 else (0, 3, 6)
    if duration in (9, 10):
        return (0, 7) if add_pos == 1 else (0, 4, 7)
    if duration == 11:
        if add_pos == 1:
            return (0, 8)
        return (0, 4, 8) if add_pos == 2 else (0, 3, 6, 9)
    if add_pos == 1:
        return (0, 9)
    return (0, 5, 9) if add_pos == 2 else (0, 3, 6, 9)


def _double_symbol_offsets(channel_type: str, mapping: str, duration: int, add_pos: int) -> tuple[int, ...] | None:
    """Return first-symbol table entries for double-symbol pairs; zero denotes ``l0``."""
    if mapping == "A":
        if duration < 4:
            return None
        if duration <= 9 or add_pos == 0:
            return (0,)
        if duration <= 12:
            return (0, 8)
        return (0, 10)

    if duration < 5 or (channel_type == "PDSCH" and duration == 14):
        return None
    if duration <= 7 or add_pos == 0:
        return (0,)
    if duration <= 9:
        return (0, 5)
    if duration <= 11:
        return (0, 7)
    return (0, 9 if channel_type == "PUSCH" else 8)


def resolve_dmrs_symbol_indices(
    *,
    channel_type: str,
    start_symbol: int,
    num_symbols: int,
    mapping_type: str,
    additional_positions: int,
    max_length: int,
    type_a_position: int,
) -> tuple[int, ...]:
    """Resolve PUSCH/PDSCH DM-RS symbols from 38.211 tables.

    Mapping type A uses absolute slot entries plus the configured ``l0``;
    mapping type B entries are relative to the allocation start.
    """
    channel = channel_type.upper()
    if channel not in {"PUSCH", "PDSCH"}:
        raise ValueError(f"Unsupported shared channel type: {channel_type}")
    mapping = mapping_type.upper()
    if mapping not in {"A", "B"}:
        raise ValueError("dmrs.mapping_type must be 'A' or 'B'.")
    start = int(start_symbol)
    length = int(num_symbols)
    slot_end = start + length
    duration = slot_end if mapping == "A" else length
    add_pos = int(additional_positions)
    if add_pos not in {0, 1, 2, 3}:
        raise ValueError("dmrs.additional_positions must be in [0, 3].")
    dmrs_length = int(max_length)
    if dmrs_length not in {1, 2}:
        raise ValueError("dmrs.max_length must be 1 or 2.")
    type_a_pos = int(type_a_position)
    if mapping == "A" and type_a_pos not in {2, 3}:
        raise ValueError("dmrs.type_a_position must be 2 or 3.")
    if mapping == "A" and add_pos == 3 and type_a_pos != 2:
        raise ValueError(
            "dmrs.additional_positions=3 is only supported when "
            "dmrs.type_a_position=2."
        )
    if dmrs_length == 2 and add_pos not in {0, 1}:
        raise ValueError(
            "Double-symbol DM-RS (dmrs.max_length=2) only supports "
            "dmrs.additional_positions 0 or 1."
        )
    if mapping == "A" and not start <= type_a_pos < slot_end:
        raise ValueError(
            "Mapping type A front-loaded DM-RS must fall inside the symbol allocation: "
            f"l0={type_a_pos}, allocation=[{start}, {slot_end})."
        )

    offsets = (
        _double_symbol_offsets(channel, mapping, duration, add_pos)
        if dmrs_length == 2
        else _single_symbol_offsets(channel, mapping, duration, add_pos)
    )
    if offsets is None:
        raise ValueError(
            "The configured channel/mapping/duration combination has no DM-RS "
            f"position in 38.211: channel={channel}, mapping={mapping}, duration={duration}."
        )

    if mapping == "A":
        pair_starts = tuple(type_a_pos if position == 0 else position for position in offsets)
    else:
        pair_starts = tuple(start + offset for offset in offsets)
    if dmrs_length == 2:
        positions = tuple(
            value
            for pair_start in pair_starts
            for value in (pair_start, pair_start + 1)
        )
    else:
        positions = pair_starts

    invalid_positions = tuple(pos for pos in positions if not start <= pos < slot_end)
    if invalid_positions:
        raise ValueError(
            "Resolved DM-RS symbols fall outside the configured symbol allocation: "
            f"positions={invalid_positions}, allocation=[{start}, {slot_end})."
        )
    if not positions:
        raise ValueError("The configured DM-RS combination resolves to no symbols.")
    return positions

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
    effective_add_pos = min(add_pos, 1)
    if mapping == "A":
        if duration < 4:
            return None
        if duration <= 9 or effective_add_pos == 0:
            return (0,)
        if duration <= 12:
            return (0, 8)
        return (0, 10)

    if duration < 5 or (channel_type == "PDSCH" and duration == 14):
        return None
    if duration <= 7 or effective_add_pos == 0:
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
    duration = int(start_symbol + num_symbols) if mapping == "A" else int(num_symbols)
    add_pos = int(additional_positions)
    if add_pos not in {0, 1, 2, 3}:
        raise ValueError("dmrs.additional_positions must be in [0, 3].")
    if int(max_length) not in {1, 2}:
        raise ValueError("dmrs.max_length must be 1 or 2.")

    offsets = (
        _double_symbol_offsets(channel, mapping, duration, add_pos)
        if int(max_length) == 2
        else _single_symbol_offsets(channel, mapping, duration, add_pos)
    )
    if offsets is None:
        raise ValueError(
            "The configured channel/mapping/duration combination has no DM-RS "
            f"position in 38.211: channel={channel}, mapping={mapping}, duration={duration}."
        )

    if mapping == "A":
        pair_starts = tuple(int(type_a_position) if position == 0 else position for position in offsets)
    else:
        pair_starts = tuple(int(start_symbol) + offset for offset in offsets)
    if int(max_length) == 2:
        positions = tuple(value for start in pair_starts for value in (start, start + 1))
    else:
        positions = pair_starts

    slot_end = int(start_symbol + num_symbols)
    return tuple(pos for pos in positions if int(start_symbol) <= pos < slot_end)

from __future__ import annotations


TYPE_A_SINGLE_SYMBOL_TABLE: dict[tuple[int, int], tuple[int, ...]] = {
    (8, 0): (2, 7),
    (8, 1): (2, 7),
    (8, 2): (2, 7),
    (8, 3): (2, 7),
    (9, 0): (2, 7),
    (9, 1): (2, 7),
    (9, 2): (2, 7),
    (9, 3): (2, 7),
    (10, 0): (2,),
    (10, 1): (2, 9),
    (10, 2): (2, 6, 9),
    (10, 3): (2, 6, 9),
    (11, 0): (2,),
    (11, 1): (2, 9),
    (11, 2): (2, 6, 9),
    (11, 3): (2, 6, 9),
    (12, 0): (2,),
    (12, 1): (2, 11),
    (12, 2): (2, 7, 11),
    (12, 3): (2, 5, 8, 11),
    (13, 0): (2,),
    (13, 1): (2, 11),
    (13, 2): (2, 7, 11),
    (13, 3): (2, 5, 8, 11),
    (14, 0): (2,),
    (14, 1): (2, 11),
    (14, 2): (2, 7, 11),
    (14, 3): (2, 5, 8, 11),
}


def resolve_dmrs_symbol_indices(
    *,
    start_symbol: int,
    num_symbols: int,
    mapping_type: str,
    additional_positions: int,
    max_length: int,
    type_a_position: int,
) -> tuple[int, ...]:
    """Resolve DMRS symbol positions from protocol-style parameters.

    The current implementation keeps the existing single-slot scheduling rules
    explicit through small lookup tables so callers can reason about which R18
    mapping pattern is selected.
    """
    mapping = mapping_type.upper()
    add_pos = int(additional_positions)
    if mapping == "A":
        base = TYPE_A_SINGLE_SYMBOL_TABLE.get((int(num_symbols), add_pos), (int(type_a_position),))
        positions = tuple(int(pos) for pos in base)
    else:
        positions = (int(start_symbol),)
        extra: list[int] = []
        if int(num_symbols) >= 8 and add_pos >= 1:
            extra.append(int(start_symbol + num_symbols - 4))
        if int(num_symbols) >= 10 and add_pos >= 2:
            extra.append(int(start_symbol + num_symbols // 2))
        if int(num_symbols) >= 12 and add_pos >= 3:
            extra.append(int(start_symbol + 3))
        positions = tuple(sorted(set(positions + tuple(extra))))

    if int(max_length) == 2:
        doubled = []
        for pos in positions:
            doubled.extend((pos, pos + 1))
        positions = tuple(sorted(set(doubled)))

    slot_end = int(start_symbol + num_symbols)
    return tuple(pos for pos in positions if int(start_symbol) <= pos < slot_end)

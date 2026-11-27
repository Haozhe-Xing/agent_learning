"""A deliberately incomplete winner selection for the OpenCode workshop."""


def winner(options: list[tuple[str, int]]) -> str:
    """Return the highest-scoring label, breaking ties by ascending label."""
    return max(options, key=lambda item: item[1])[0]

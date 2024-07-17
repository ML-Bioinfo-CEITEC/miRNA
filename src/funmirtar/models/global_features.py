def get_only_positive_conservation(conservation):
    if isinstance(conservation, int):
        return []
    return [x if x > 0 else 0 for x in conservation]
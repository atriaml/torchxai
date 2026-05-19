from collections.abc import Mapping, MutableMapping


def deep_merge(base: MutableMapping, override: Mapping):
    """Recursively merges override into base.

    Parameters:
        base: the dictionary to merge into (modified in place)
        override: the dictionary to merge from (not modified)
    """
    for key in override:
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(override[key], Mapping)
        ):
            deep_merge(base[key], override[key])
        else:
            base[key] = override[key]
    return base

"""Environment-variable switches for experimental, off-by-default code paths.

These are deliberately env vars rather than CLI arguments while the paths are
experimental: they can be flipped on an existing installation / job script
without touching the argument plumbing, and with the variable unset the code
path is byte-for-byte the stock one.
"""

import os

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def env_flag(name: str, default: bool = False) -> bool:
    """Parse ``$name`` as a boolean switch.

    ``1/true/yes/on`` -> True, ``0/false/no/off`` (or empty) -> False, unset ->
    ``default``. Anything else raises so a typo cannot silently mean "on" (a
    bare presence check would treat ``NAME=0`` as enabled).
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    allowed = sorted(_TRUE | (_FALSE - {""}))
    raise ValueError(f"{name}={raw!r}: expected one of {allowed}")

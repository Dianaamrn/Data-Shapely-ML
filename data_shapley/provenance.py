"""file fingerprints for checking which code and data made the results."""
import hashlib
from pathlib import Path


def source_sha256(path):
    """fingerprint of a code file, ignoring windows or linux line endings."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def raw_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

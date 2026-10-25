"""Lifetime DLL search handles for actual native imports and isolated probes."""
import os
import sys
from importlib import import_module, metadata, util
from pathlib import Path
from threading import RLock

_DLL_HANDLES = {}
_DLL_LOCK = RLock()


def dll_directories():
    """Discover package locations without importing (and locking) Torch/CT2."""
    try:
        if "rocm" not in metadata.version("torch").lower():
            return []
    except metadata.PackageNotFoundError:
        return []
    directories = []
    for package, suffix in (("torch", "lib"), ("ctranslate2", ""),
                            ("_rocm_sdk_core", "bin"), ("_rocm_sdk_libraries_custom", "bin")):
        spec = util.find_spec(package)
        if spec and spec.submodule_search_locations:
            root = Path(next(iter(spec.submodule_search_locations)))
            directory = root / suffix
            if directory.is_dir():
                directories.append(directory.resolve())
    return directories


def initialize_dll_search():
    """Keep each AddDllDirectory handle alive for the process; idempotent."""
    if sys.platform != "win32":
        return
    with _DLL_LOCK:
        _prepare_miopen_cache()
        _prepare_miopen_headers()
        for directory in dll_directories():
            key = str(directory)
            if key not in _DLL_HANDLES:
                _DLL_HANDLES[key] = os.add_dll_directory(key)


def _prepare_miopen_cache():
    """Avoid the pinned MIOpen SQLite failure with Unicode custom cache paths."""
    cache_path = os.environ.get("MIOPEN_CUSTOM_CACHE_DIR", "")
    if cache_path.isascii() or "MIOPEN_DISABLE_CACHE" in os.environ:
        return
    try:
        if metadata.version("torch") != "2.9.1+rocm7.2.1":
            return
    except metadata.PackageNotFoundError:
        return
    os.environ.setdefault("MIOPEN_DISABLE_CACHE", "1")


def _pinned_development_sdk():
    if sys.platform != "win32":
        return False
    try:
        return (
            metadata.version("torch") == "2.9.1+rocm7.2.1"
            and metadata.version("rocm-sdk-devel") == "7.2.1"
        )
    except metadata.PackageNotFoundError:
        return False


def _prepare_miopen_headers():
    """Point HIPRTC at already expanded headers without changing package files."""
    if "ROCM_PATH" in os.environ or not _pinned_development_sdk():
        return
    specification = util.find_spec("_rocm_sdk_devel")
    if specification and specification.submodule_search_locations:
        root = Path(next(iter(specification.submodule_search_locations)))
        if (root / "include/rocrand/rocrand_xorwow.h").is_file():
            os.environ.setdefault("ROCM_PATH", str(root))


def initialize_development_sdk():
    """Expand pinned development files only during explicitly started setup."""
    if not _pinned_development_sdk():
        return
    root = Path(import_module("rocm_sdk._devel").get_devel_root())
    if not (root / "include/rocrand/rocrand_xorwow.h").is_file():
        raise RuntimeError("Pinned ROCm SDK expansion did not provide required headers")
    _prepare_miopen_headers()

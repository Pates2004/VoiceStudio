"""Windows ROCm RDNA 2 uses pinned native kernels and safe precision."""
import importlib
import json
import re
import types

import pytest


def _module(name):
    """Resolve the canonical module even when a parent package retains a stale alias."""
    return importlib.import_module(name)


def test_recipe_has_rdna2_native_variant():
    recipe = _module("core.windows_rocm.recipe")

    selected = recipe.load_recipe(architecture="gfx1031")
    assert selected["gpu_architectures"] == ["gfx1031"]
    assert selected["torch"]["hip_prefix"] == "7.15."
    assert "amd-torch-device-gfx1031==2.13.0+rocm10.0.0" in selected["runtime"]["packages"]
    assert all("gfx1030" not in p and "gfx1032" not in p for p in selected["runtime"]["packages"])


def test_rdna2_urls_are_allowlisted_and_hashed():
    install = _module("core.windows_rocm.install")
    recipe = _module("core.windows_rocm.recipe")

    selected = recipe.load_recipe(architecture="gfx1031")
    for line in install.hashed_requirements(selected).splitlines():
        url, digest = line.split(" --hash=sha256:")
        assert url.startswith("https://stable.repo.amd.com/rocm/whl-next/")
        assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_missing_rdna2_hash_is_rejected(tmp_path):
    recipe = _module("core.windows_rocm.recipe")

    data = json.loads(recipe.RECIPE_PATH.read_text(encoding="utf-8"))
    device = data["rdna2"]["devices"]["gfx1031"]
    del device["wheel_sha256"][device["wheel_files"][0]]
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="sha256"):
        recipe.load_recipe(path, architecture="gfx1031")


def test_unapproved_rdna2_source_is_rejected(tmp_path):
    recipe = _module("core.windows_rocm.recipe")

    data = json.loads(recipe.RECIPE_PATH.read_text(encoding="utf-8"))
    data["rdna2"]["torch"]["find_links"] = "https://unapproved.example/"
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source"):
        recipe.load_recipe(path, architecture="gfx1031")


def test_windows_rocm10_only_reports_installed_architectures(monkeypatch):
    device_caps = _module("core.device_caps")

    monkeypatch.setattr(device_caps.sys, "platform", "win32")
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    fake = types.SimpleNamespace(
        __version__="2.13.0+rocm10.0.0", version=types.SimpleNamespace(hip="7.15"),
        cuda=types.SimpleNamespace(is_available=lambda: True,
            get_arch_list=lambda: ["gfx1030", "gfx1031"],
            get_device_properties=lambda index: types.SimpleNamespace(gcnArchName="gfx1031")))
    monkeypatch.setattr(device_caps, "_installed_rocm_arches", lambda version: {"gfx1031"})
    assert device_caps.arch_unsupported(fake) is None
    assert device_caps.build_arch_list(fake) == ["gfx1031"]
    assert device_caps.rocm_override_target("gfx1031", device_caps.build_arch_list(fake)) is None

@pytest.mark.parametrize("pci,gfx", [("73bf", "gfx1030"), ("73df", "gfx1031"), ("73ff", "gfx1032")])
def test_pci_device_selects_the_exact_recipe(monkeypatch, pci, gfx):
    gpu_inventory = _module("core.gpu_inventory")
    recipe = _module("core.windows_rocm.recipe")

    monkeypatch.setattr(recipe.sys, "platform", "win32")
    monkeypatch.setattr(gpu_inventory, "detect_host_gpus", lambda **kwargs: (
        gpu_inventory.HostGPU(vendor="amd", name="Radeon", pci_device_id=pci),))
    assert recipe.load_recipe(for_install=True)["gpu_architectures"] == [gfx]


def test_mixed_amd_architectures_fail_closed(monkeypatch):
    gpu_inventory = _module("core.gpu_inventory")
    recipe = _module("core.windows_rocm.recipe")

    monkeypatch.setattr(recipe.sys, "platform", "win32")
    monkeypatch.setattr(gpu_inventory, "detect_host_gpus", lambda **kwargs: (
        gpu_inventory.HostGPU(vendor="amd", name="Radeon", pci_device_id="73df"),
        gpu_inventory.HostGPU(vendor="amd", name="Radeon", pci_device_id="744c")))
    with pytest.raises(RuntimeError, match="Multiple AMD"):
        recipe.load_recipe(for_install=True)


@pytest.mark.parametrize("gfx", ["gfx1030", "gfx1031", "gfx1032"])
def test_each_selected_artifact_is_pinned(gfx):
    install = _module("core.windows_rocm.install")
    recipe = _module("core.windows_rocm.recipe")

    selected = recipe.load_recipe(architecture=gfx)
    lines = install.hashed_requirements(selected).splitlines()
    assert len(lines) == 10
    assert all(re.search(r" --hash=sha256:[0-9a-f]{64}$", line) for line in lines)
    assert sum(f"device-{gfx}/" in line for line in lines) == 3


def test_the_amd_kernel_and_sdk_packages_are_both_required(monkeypatch):
    from importlib import metadata
    device_caps = _module("core.device_caps")

    class Dist:
        metadata = {"Name": "amd-torch-device-gfx1031"}
        version = "2.13.0+rocm10.0.0"
    monkeypatch.setattr(metadata, "distributions", lambda: [Dist()])
    def missing_sdk(name):
        raise metadata.PackageNotFoundError(name)
    monkeypatch.setattr(metadata, "version", missing_sdk)
    assert device_caps._installed_rocm_arches(Dist.version) == set()


def test_native_gfx1031_is_reported_as_accelerated():
    device_caps = _module("core.device_caps")
    build_gpu_report = _module("core.gpu_report").build_gpu_report

    caps = device_caps.HostCaps(family="rocm", available_families=("rocm", "cpu"), device_name="Radeon")
    report = build_gpu_report(caps, (), {"kind": "rocm", "version": "2.13.0+rocm10.0.0"}, [], [], platform="win32")
    assert report["state"] == "accelerated"
    assert report["options"] == []


def test_windows_never_sets_hsa_override(monkeypatch):
    model_manager = _module("services.model_manager")
    monkeypatch.setattr(model_manager.sys, "platform", "win32")
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    fake = types.SimpleNamespace(
        version=types.SimpleNamespace(hip="7.15"),
        cuda=types.SimpleNamespace(get_device_name=lambda i: "Radeon",
            get_device_properties=lambda i: types.SimpleNamespace(gcnArchName="gfx1031"),
            get_arch_list=lambda: ["gfx1030"]))
    model_manager._configure_rocm_if_needed(fake)
    assert "HSA_OVERRIDE_GFX_VERSION" not in model_manager.os.environ


def test_rocm10_dll_search_excludes_stale_rocm7_libraries(monkeypatch, tmp_path):
    runtime = _module("core.windows_rocm.runtime")
    for package in ("torch", "ctranslate2", "_rocm_sdk_core", "_rocm_sdk_libraries", "_rocm_sdk_libraries_custom"):
        (tmp_path / package / ("lib" if package == "torch" else "" if package == "ctranslate2" else "bin")).mkdir(parents=True)
    monkeypatch.setattr(runtime.metadata, "version", lambda name: "2.13.0+rocm10.0.0")
    monkeypatch.setattr(runtime, "package_root", lambda name: tmp_path / name)
    directories = runtime.dll_directories()
    assert (tmp_path / "_rocm_sdk_libraries" / "bin").resolve() in directories
    assert (tmp_path / "_rocm_sdk_libraries_custom" / "bin").resolve() not in directories


def test_rdna2_refuses_an_hsa_override_before_imports(monkeypatch):
    probe = _module("core.windows_rocm.probe")
    recipe = _module("core.windows_rocm.recipe")

    selected = recipe.load_recipe(architecture="gfx1031")
    monkeypatch.setattr(probe, "validate_target", lambda: None)
    monkeypatch.setattr(probe, "load_recipe", lambda: selected)
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    monkeypatch.setattr(probe.metadata, "version", lambda name: pytest.fail("unexpected package probe"))
    with pytest.raises(RuntimeError, match="Remove HSA_OVERRIDE_GFX_VERSION"):
        probe.probe()


def test_missing_rocm10_kernel_metadata_fails_closed(monkeypatch):
    device_caps = _module("core.device_caps")

    monkeypatch.setattr(device_caps.sys, "platform", "win32")
    fake = types.SimpleNamespace(__version__="2.13.0+rocm10.0.0",
        version=types.SimpleNamespace(hip="7.15"),
        cuda=types.SimpleNamespace(is_available=lambda: True,
            get_arch_list=lambda: ["gfx1031"],
            get_device_properties=lambda i: types.SimpleNamespace(gcnArchName="gfx1031")))
    monkeypatch.setattr(device_caps, "_installed_rocm_arches", lambda version: (_ for _ in ()).throw(OSError("metadata")))
    assert device_caps.arch_unsupported(fake) is not None


def test_recipe_rejects_a_requirements_file_injection(tmp_path):
    recipe = _module("core.windows_rocm.recipe")

    data = json.loads(recipe.RECIPE_PATH.read_text(encoding="utf-8"))
    section = data["rdna2"]["torch"]
    name = section["wheel_files"][0]
    injected = name + "\nhttps://unapproved.example/wheel.whl"
    section["wheel_files"][0] = injected
    section["wheel_sha256"][injected] = section["wheel_sha256"].pop(name)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsafe artifact"):
        recipe.load_recipe(path, architecture="gfx1031")


@pytest.fixture
def existing_rocm10(monkeypatch):
    """Expose installed10 metadata while refusing any real acquisition or install."""
    from unittest.mock import Mock

    install = _module("core.windows_rocm.install")
    recipe = _module("core.windows_rocm.recipe")
    VerificationFailure = _module("core.windows_rocm.failures").VerificationFailure

    installed = recipe.expected_versions(recipe.load_recipe(architecture="gfx1031"))

    def installed_version(name):
        try:
            return installed[name]
        except KeyError as exc:
            raise install.metadata.PackageNotFoundError(name) from exc

    monkeypatch.setattr(install.metadata, "version", installed_version)
    monkeypatch.setattr(install, "validate_target", lambda: None)

    def setup(category):
        """Inject the repairable child outcome without running a native probe."""
        verify = Mock(side_effect=VerificationFailure(category, "Installed ROCm10 needs verification"))
        download = Mock(side_effect=AssertionError("Unexpected download with unknown GPU"))
        install_command = Mock(side_effect=AssertionError("Unexpected package installation"))
        monkeypatch.setattr(install, "verify_environment", verify)
        monkeypatch.setattr(install, "download_archive", download)
        monkeypatch.setattr(install, "run_install", install_command)
        return types.SimpleNamespace(install=install, installed=installed, verify=verify,
                                     download=download, install_command=install_command)

    return setup


def _assert_unknown_repair_stops(runtime, *, message="identify", expected_torch="2.13.0+rocm10.0.0"):
    """Assert the actual installer preserves installed packages and all mutation boundaries."""
    VerificationFailure = _module("core.windows_rocm.failures").VerificationFailure

    before = dict(runtime.installed)
    if expected_torch is not None:
        assert runtime.install.metadata.version("torch") == expected_torch
    with pytest.raises(VerificationFailure, match=message) as failure:
        runtime.install.provision("uv-unit-fixture")
    assert failure.value.repairable is False
    runtime.verify.assert_called_once_with()
    runtime.download.assert_not_called()
    runtime.install_command.assert_not_called()
    assert runtime.installed == before


@pytest.mark.parametrize("pci", [None, "", "ffff"])
@pytest.mark.parametrize("category", ["package_missing", "package_version"])
def test_unknown_gpu_does_not_repair_an_existing_rocm10_runtime(monkeypatch, existing_rocm10, pci, category):
    """Unknown architecture must stop repair before acquisition or package mutation."""
    gpu_inventory = _module("core.gpu_inventory")
    recipe = _module("core.windows_rocm.recipe")

    runtime = existing_rocm10(category)
    monkeypatch.setattr(recipe.sys, "platform", "win32")
    inventory = () if pci is None else (
        gpu_inventory.HostGPU(vendor="amd", name="Radeon", pci_device_id=pci),
    )
    monkeypatch.setattr(gpu_inventory, "detect_host_gpus", lambda **kwargs: inventory)
    _assert_unknown_repair_stops(runtime, message="ROCm overlay" if pci else "identify")


@pytest.mark.parametrize("other_pci", ["ffff", "13c0"])
def test_unmapped_gpu_does_not_hide_among_supported_gpus(monkeypatch, existing_rocm10, other_pci):
    """A second display adapter cannot authorize replacing an existing10 overlay."""
    gpu_inventory = _module("core.gpu_inventory")
    recipe = _module("core.windows_rocm.recipe")

    runtime = existing_rocm10("package_version")
    monkeypatch.setattr(recipe.sys, "platform", "win32")
    monkeypatch.setattr(gpu_inventory, "detect_host_gpus", lambda **kwargs: tuple(
        gpu_inventory.HostGPU(vendor="amd", name="Radeon", pci_device_id=pci) for pci in ("7550", other_pci)
    ))
    _assert_unknown_repair_stops(runtime, message="ROCm overlay")


@pytest.mark.parametrize("fault", ["adapter", "identity", "enumeration"])
def test_partial_registry_does_not_repair_an_existing_rocm10_runtime(monkeypatch, existing_rocm10, fault):
    """A readable9070 must not hide an unreadable second adapter during repair."""
    import sys
    from contextlib import nullcontext

    gpu_inventory = _module("core.gpu_inventory")

    runtime = existing_rocm10("package_version")
    registry = types.SimpleNamespace(HKEY_LOCAL_MACHINE=object())

    def open_key(parent, name):
        if parent is registry.HKEY_LOCAL_MACHINE:
            return nullcontext("display-class")
        if name == "0001" and fault == "adapter":
            raise PermissionError("Unreadable second adapter")
        return nullcontext(name)

    def enum_key(key, index):
        if index == 1 and fault == "enumeration":
            raise PermissionError("Incomplete adapter enumeration")
        if index >= 2:
            error = OSError("No more adapter keys")
            error.winerror = 259
            raise error
        return f"{index:04d}"

    def query_value(key, name):
        if name == "MatchingDeviceId":
            if key == "0001" and fault == "identity":
                raise PermissionError("Unreadable second adapter identity")
            return ("PCI\\VEN_1002&DEV_" + ("7550" if key == "0000" else "73df"), 1)
        return ("Radeon" if name == "DriverDesc" else 0, 1)

    registry.OpenKey, registry.EnumKey, registry.QueryValueEx = open_key, enum_key, query_value
    monkeypatch.setitem(sys.modules, "winreg", registry)
    monkeypatch.setattr(gpu_inventory.sys, "platform", "win32")
    monkeypatch.delenv("OMNIVOICE_DISABLE_GPU_INVENTORY", raising=False)
    gpu_inventory.detect_host_gpus.cache_clear()
    try:
        assert [gpu.pci_device_id for gpu in gpu_inventory.detect_host_gpus()] == ["7550"]
        _assert_unknown_repair_stops(runtime)
        fault = None
        assert [gpu.pci_device_id for gpu in gpu_inventory.detect_host_gpus(require_complete=True)] == ["7550", "73df"]
    finally:
        gpu_inventory.detect_host_gpus.cache_clear()


@pytest.mark.parametrize("architecture", ["gfx1100", "gfx1101", "gfx1150", "gfx1151", "gfx1200", "gfx1201"])
def test_install_recipe_preserves_existing_supported_architectures(architecture):
    """Explicit supported RDNA3/3.5/4 selections retain the7.2.1 package recipe."""
    recipe = _module("core.windows_rocm.recipe")

    selected = recipe.load_recipe(architecture=architecture, for_install=True)
    assert architecture in selected["gpu_architectures"]
    assert selected["torch"]["packages"] == [
        "torch==2.9.1+rocm7.2.1",
        "torchaudio==2.9.1+rocm7.2.1",
        "torchvision==0.24.1+rocm7.2.1",
    ]


def test_confirmed_rdna4_inventory_keeps_the_seven_recipe(monkeypatch):
    """A complete9070 PCI inventory remains eligible for the existing default."""
    from unittest.mock import Mock

    gpu_inventory = _module("core.gpu_inventory")
    recipe = _module("core.windows_rocm.recipe")

    inventory = Mock(return_value=(gpu_inventory.HostGPU(vendor="amd", name="Radeon9070", pci_device_id="7550"),))
    monkeypatch.setattr(gpu_inventory, "detect_host_gpus", inventory)
    monkeypatch.setattr(recipe.sys, "platform", "win32")
    selected = recipe.load_recipe(for_install=True)
    inventory.assert_called_once_with(require_complete=True)
    assert "gfx1201" in selected["gpu_architectures"]
    assert selected["torch"]["find_links"] == "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/"


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_non_windows_recipe_does_not_query_the_windows_inventory(monkeypatch, platform):
    """Other platforms retain their recipe-read behavior without registry access."""
    from unittest.mock import Mock

    gpu_inventory = _module("core.gpu_inventory")
    recipe = _module("core.windows_rocm.recipe")

    inventory = Mock(side_effect=AssertionError("Unexpected Windows inventory"))
    monkeypatch.setattr(gpu_inventory, "detect_host_gpus", inventory)
    monkeypatch.setattr(recipe.sys, "platform", platform)
    recipe.load_recipe(for_install=True)
    inventory.assert_not_called()


@pytest.mark.parametrize("installed_state", ["fresh", "seven", "seven_with_old_libraries"])
@pytest.mark.parametrize("pcis", [("7550", "13c0"), ("13c0", "7550")])
def test_seven_setup_keeps_the_supported_gpu_with_an_advisory_igpu(monkeypatch, existing_rocm10, installed_state, pcis):
    """The real9070+iGPU topology remains eligible for fresh or same-family7 setup."""
    gpu_inventory = _module("core.gpu_inventory")
    recipe = _module("core.windows_rocm.recipe")
    runtime = existing_rocm10("package_missing")
    runtime.installed.clear()
    if installed_state != "fresh":
        runtime.installed.update(recipe.expected_versions(recipe.load_recipe(architecture="gfx1201")))
    if installed_state == "seven_with_old_libraries":
        runtime.installed["rocm-sdk-libraries"] = "10.0.0"
    monkeypatch.setattr(recipe.sys, "platform", "win32")
    monkeypatch.setattr(gpu_inventory, "detect_host_gpus", lambda **kwargs: tuple(
        gpu_inventory.HostGPU(vendor="amd", name="AMD adapter", pci_device_id=pci) for pci in pcis
    ))
    with pytest.raises(AssertionError, match="Unexpected download"):
        runtime.install.provision("uv-unit-fixture")
    runtime.download.assert_called_once()
    runtime.install_command.assert_not_called()


@pytest.mark.parametrize("pci", ["73bf", "73df", "73ff"])
@pytest.mark.parametrize("fresh", [False, True])
def test_native_rdna2_can_start_fresh_or_same_family_repair(monkeypatch, existing_rocm10, pci, fresh):
    """All three native RDNA2 targets reach the mocked checked-acquisition boundary."""
    gpu_inventory = _module("core.gpu_inventory")
    recipe = _module("core.windows_rocm.recipe")
    runtime = existing_rocm10("package_missing")
    if fresh:
        runtime.installed.clear()
    monkeypatch.setattr(recipe.sys, "platform", "win32")
    monkeypatch.setattr(gpu_inventory, "detect_host_gpus", lambda **kwargs: (
        gpu_inventory.HostGPU(vendor="amd", name="AMD adapter", pci_device_id=pci),
    ))
    with pytest.raises(AssertionError, match="Unexpected download"):
        runtime.install.provision("uv-unit-fixture")
    runtime.download.assert_called_once()
    runtime.install_command.assert_not_called()


@pytest.mark.parametrize("torch_version", [None, "2.9.1"])
def test_sdk_family_protects_existing10_when_hip_torch_metadata_is_missing(monkeypatch, existing_rocm10, torch_version):
    """An absent or non-HIP Torch wheel must not hide the existing shared SDK family."""
    gpu_inventory = _module("core.gpu_inventory")
    recipe = _module("core.windows_rocm.recipe")
    runtime = existing_rocm10("package_missing")
    runtime.installed.pop("torch")
    if torch_version is not None:
        runtime.installed["torch"] = torch_version
    monkeypatch.setattr(recipe.sys, "platform", "win32")
    monkeypatch.setattr(gpu_inventory, "detect_host_gpus", lambda **kwargs: (
        gpu_inventory.HostGPU(vendor="amd", name="AMD adapter", pci_device_id="7550"),
    ))
    assert runtime.installed["rocm-sdk-core"] == "10.0.0"
    _assert_unknown_repair_stops(runtime, message="ROCm overlay", expected_torch=torch_version)


def test_existing_seven_is_not_replaced_with10_during_repair(monkeypatch, existing_rocm10):
    """The family guard protects both directions of an automatic overlay conversion."""
    gpu_inventory = _module("core.gpu_inventory")
    recipe = _module("core.windows_rocm.recipe")
    runtime = existing_rocm10("package_version")
    runtime.installed.clear()
    runtime.installed.update(recipe.expected_versions(recipe.load_recipe(architecture="gfx1201")))
    monkeypatch.setattr(recipe.sys, "platform", "win32")
    monkeypatch.setattr(gpu_inventory, "detect_host_gpus", lambda **kwargs: (
        gpu_inventory.HostGPU(vendor="amd", name="AMD adapter", pci_device_id="73df"),
    ))
    _assert_unknown_repair_stops(runtime, message="ROCm overlay", expected_torch="2.9.1+rocm7.2.1")


def test_conflicting_sdk_families_cannot_authorize_repair_without_torch(monkeypatch, existing_rocm10):
    """Conflicting shared SDK metadata is not evidence for choosing either family."""
    gpu_inventory = _module("core.gpu_inventory")
    recipe = _module("core.windows_rocm.recipe")
    runtime = existing_rocm10("package_missing")
    runtime.installed.pop("torch")
    runtime.installed["rocm"] = "7.2.1"
    monkeypatch.setattr(recipe.sys, "platform", "win32")
    monkeypatch.setattr(gpu_inventory, "detect_host_gpus", lambda **kwargs: (
        gpu_inventory.HostGPU(vendor="amd", name="AMD adapter", pci_device_id="7550"),
    ))
    _assert_unknown_repair_stops(runtime, message="ROCm overlay", expected_torch=None)


@pytest.mark.parametrize("architecture,torch_version,sdk_version,pci", [
    ("gfx1201", "2.9.1+rocm7.2.1", "10.0.0", "7550"),
    ("gfx1031", "2.13.0+rocm10.0.0", "7.2.1", "73df"),
])
def test_conflicting_active_torch_and_sdk_families_cannot_authorize_repair(
    monkeypatch, existing_rocm10, architecture, torch_version, sdk_version, pci,
):
    """A matching Torch tag cannot authorize overwriting a different active SDK."""
    gpu_inventory = _module("core.gpu_inventory")
    recipe = _module("core.windows_rocm.recipe")
    runtime = existing_rocm10("package_version")
    runtime.installed.clear()
    runtime.installed.update(recipe.expected_versions(recipe.load_recipe(architecture=architecture)))
    runtime.installed["rocm-sdk-core"] = sdk_version
    monkeypatch.setattr(recipe.sys, "platform", "win32")
    monkeypatch.setattr(gpu_inventory, "detect_host_gpus", lambda **kwargs: (
        gpu_inventory.HostGPU(vendor="amd", name="AMD adapter", pci_device_id=pci),
    ))
    _assert_unknown_repair_stops(runtime, message="ROCm overlay", expected_torch=torch_version)

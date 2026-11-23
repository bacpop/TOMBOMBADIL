"""JAX backend configuration performed before importing JAX."""

import os
import re


SUPPORTED_PLATFORMS = ("cpu", "gpu", "tpu")


def _host_device_count_flags(existing_flags, cpus):
    flag_name = "--xla_force_host_platform_device_count"
    pattern = re.compile(
        rf"(?<!\S){re.escape(flag_name)}(?:\s*=\s*|\s+)\d+(?=\s|$)"
    )
    found = False

    def replace_count(_match):
        nonlocal found
        if found:
            return ""
        found = True
        return f"{flag_name}={cpus}"

    flags = pattern.sub(replace_count, existing_flags)
    if not found:
        flags = f"{flags} {flag_name}={cpus}"
    return re.sub(r"\s{2,}", " ", flags).strip()


def configure_platform(platform, *, cpus, force_cpu_devices=False):
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"Unsupported JAX platform: {platform!r}")
    try:
        cpus = int(cpus)
    except (TypeError, ValueError):
        raise ValueError("CPU worker count must be a positive integer") from None
    if cpus < 1:
        raise ValueError("CPU worker count must be a positive integer")

    os.environ["JAX_PLATFORMS"] = platform
    if platform == "cpu":
        os.environ["NPROC"] = str(cpus)
        if force_cpu_devices:
            os.environ["XLA_FLAGS"] = _host_device_count_flags(
                os.environ.get("XLA_FLAGS", ""), cpus
            )


def validate_platform(platform):
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"Unsupported JAX platform: {platform!r}")
    import jax
    devices = list(jax.devices(platform))
    if not devices:
        raise RuntimeError(f"No JAX devices are available for platform {platform!r}")
    return devices, jax.default_backend()

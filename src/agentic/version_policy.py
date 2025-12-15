import os
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from typing import Literal, TypeAlias

from packaging.version import Version

SDK: TypeAlias = Literal['python', 'typescript']

try:
    _SESSION_MANAGER_VERSION = get_version("agentica-server")
except PackageNotFoundError:
    _SESSION_MANAGER_VERSION = "0.0.0-dev"


class VersionStatus(Enum):
    OK = "ok"
    DEPRECATED = "deprecated"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class SDKVersionPolicy:
    min_supported: Version
    min_recommended: Version


SDK_VERSION_POLICIES: dict[SDK, SDKVersionPolicy] = {
    "python": SDKVersionPolicy(
        min_supported=Version(_SESSION_MANAGER_VERSION),
        min_recommended=Version(_SESSION_MANAGER_VERSION),
    ),
    "typescript": SDKVersionPolicy(
        min_supported=Version(_SESSION_MANAGER_VERSION),
        min_recommended=Version(_SESSION_MANAGER_VERSION),
    ),
}


UPGRADE_URL = "https://agentica.symbolica.ai/quickstart"


def _is_local_mode() -> bool:
    org_id = os.getenv("ORGANIZATION_ID", "LOCAL_SESSION_MANAGER")
    return org_id == "LOCAL_SESSION_MANAGER"


def is_disabled_version_check() -> bool:
    return os.environ.get('AGENTICA_SERVER_DISABLE_VERSION_CHECK', '0') == '1'


def check_sdk_version(sdk: SDK, version: str) -> VersionStatus:
    if is_disabled_version_check():
        return VersionStatus.OK

    # Allow 0.0.0-dev for local development
    if version == "0.0.0-dev":
        if _is_local_mode():
            return VersionStatus.OK
        else:
            return VersionStatus.UNSUPPORTED

    policy = SDK_VERSION_POLICIES.get(sdk)
    if not policy:
        return VersionStatus.OK

    try:
        v = Version(version)
        if v < policy.min_supported:
            return VersionStatus.UNSUPPORTED
        elif v < policy.min_recommended:
            return VersionStatus.DEPRECATED
        else:
            return VersionStatus.OK
    except Exception:
        return VersionStatus.UNSUPPORTED


def format_upgrade_message(sdk: SDK, version: str) -> str:
    policy = SDK_VERSION_POLICIES[sdk]
    return (
        f"SDK update recommended: "
        f"your version {version}, "
        f"recommended {policy.min_recommended.public}+. "
        f"Visit {UPGRADE_URL}"
    )


def format_unsupported_message(sdk: SDK, version: str) -> str:
    policy = SDK_VERSION_POLICIES[sdk]
    return (
        f"\n{'=' * 60}\n"
        + f"  SDK VERSION NOT SUPPORTED\n"
        + f"{'=' * 60}\n"
        + (f"  Your version: {version}\n" if version != "0.0.0-dev" else "")
        + f"  Minimum required: {policy.min_supported.public}\n"
        + f"\n"
        + f"  Please use your package manager to upgrade to the latest version.\n"
        + f"{'=' * 60}\n"
    )

"""
Extracts version numbers from the session manager's version policy
for use in TypeScript version policy tests.

Outputs shell-compatible variable assignments.

NOTE: This script MUST be run via `uv run python get_test_versions.py`
      from the server directory to ensure access to dependencies.
      Do NOT use a shebang or run directly with system python.
"""

from pathlib import Path

session_manager_dir = Path(__file__).parent
version_policy_path = session_manager_dir / "src" / "agentic" / "version_policy.py"

import importlib.util

spec = importlib.util.spec_from_file_location("version_policy", version_policy_path)
version_policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version_policy)

SDK_VERSION_POLICIES = version_policy.SDK_VERSION_POLICIES
Version = version_policy.Version


def calculate_unsupported_version(min_supported: Version) -> str:
    if min_supported >= Version("0.2.0"):
        return "0.1.99"
    else:
        return "0.0.1"


def main():
    policy = SDK_VERSION_POLICIES["typescript"]

    min_supported = policy.min_supported
    min_recommended = policy.min_recommended

    unsupported_version = calculate_unsupported_version(min_supported)
    deprecated_version = str(min_supported.public)

    has_deprecated_range = min_supported < min_recommended

    print(f"UNSUPPORTED_VERSION={unsupported_version}")
    print(f"DEPRECATED_VERSION={deprecated_version}")
    print(f"MIN_SUPPORTED={min_supported.public}")
    print(f"MIN_RECOMMENDED={min_recommended.public}")
    print(f"HAS_DEPRECATED_RANGE={'true' if has_deprecated_range else 'false'}")


if __name__ == "__main__":
    main()

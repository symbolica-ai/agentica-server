import os
import pathlib
import re
import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py

sandbox_dir = pathlib.Path(__file__).parent / "src" / "sandbox"
build_dir = sandbox_dir / "build"


def version_scheme(version):
    """Custom version scheme for external/standalone repo builds.

    Uses git tag as semver directly.
    Tag format: v0.3.2 or v0.3.2.dev2
    """
    tag = str(version.tag)
    distance = version.distance or 0

    # Tag is the semver
    tag_match = re.match(r'^v?(\d+\.\d+\.\d+)(?:\.dev(\d+))?$', tag)
    if tag_match:
        base_version = tag_match.group(1)
        tag_dev = int(tag_match.group(2)) if tag_match.group(2) else 0
        dev_num = tag_dev + distance

        if dev_num == 0:
            return base_version
        return f"{base_version}.dev{dev_num}"

    return "0.0.0.dev0"


class BuildPy(_build_py):
    def run(self):
        for script in ["build_guest.sh", "build_host.sh"]:
            result = subprocess.run(
                ["bash", script],
                cwd=build_dir,
                stdout=sys.stderr,
                stderr=sys.stderr,
                text=True,
            )
            if result.returncode != 0:
                print(f"\n{script} failed with exit code {result.returncode}", file=sys.stderr)
                os._exit(1)

        super().run()


setup(
    name="session_manager",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"session_manager": ["sandbox/env.wasm"]},
    include_package_data=True,
    cmdclass={"build_py": BuildPy},
    use_scm_version={
        "version_scheme": version_scheme,
        "local_scheme": "no-local-version",
    },
)

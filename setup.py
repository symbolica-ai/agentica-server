import os
import pathlib
import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py

sandbox_dir = pathlib.Path(__file__).parent / "src" / "sandbox"
build_dir = sandbox_dir / "build"


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
)

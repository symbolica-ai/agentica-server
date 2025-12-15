#!/usr/bin/env python3
"""Generates the OpenAPI schema for the session manager."""

import os
import subprocess
import sys


def main():
    """Generate OpenAPI schema using litestar."""
    # Change to the src directory
    src_dir = os.path.join(os.path.dirname(__file__))
    os.chdir(src_dir)

    # Run `litestar schema openapi` with explicit app path
    # This avoids autodiscovery issues with forward references in imported types
    try:
        result = subprocess.run(
            ["litestar", "--app", "application.main:app", "schema", "openapi"],
            check=True,
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error generating OpenAPI schema: {e}", file=sys.stderr)
        return e.returncode
    except FileNotFoundError:
        print(
            "Error: litestar command not found. Make sure litestar is installed.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

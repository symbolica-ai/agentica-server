import argparse
import ast
import sys
from pathlib import Path

# we can't import uuid because that somehow causes pydantic to break at runtime
DESIRED_MODULES = [
    'abc',
    'asyncio',
    'base64',
    'collections',
    'csv',
    'dataclasses',
    'datetime',
    'decimal',
    'difflib',
    'enum',
    'fractions',
    'functools',
    'hashlib',
    'inspect',
    'itertools',
    'json',
    'math',
    'os',
    'pprint',
    're',
    'random',
    'secrets',
    'statistics',
    'string',
    'sys',
    'textwrap',
    'time',
    'traceback',
    'typeguard',
    'types',
    'typing',
    'typing_extensions',
]

EXCLUDED_MODULES = {
    'subprocess',
    'socket',
    '_socket',
    'platform',
    'posix',
    'nt',
    'posixpath',
    'ntpath',
    'unix_events',
    'windows_events',
    'shutil',
    'io',
    'argparse',
    'getopt',
    'doctest',
    'pdb',
    'trace',
    '_hashlib',
    '_posixsubprocess',
    '_ssl',
}


def find_non_toplevel_imports(module_path: Path) -> dict[str, str]:
    try:
        source = module_path.read_text()
        tree = ast.parse(source)
    except Exception as e:
        print(f"  Warning: Could not parse {module_path}: {e}", file=sys.stderr)
        return {}

    top_level_imports = set()
    nested_imports = {}

    class ImportFinder(ast.NodeVisitor):
        def __init__(self):
            self.depth = 0
            self.context_stack = []

        def _extract_import_names(self, node):
            names = []
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split('.')[0]
                    names.append((alias.name, root))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split('.')[0]
                    for alias in node.names:
                        if alias.name.startswith('_'):
                            full_name = f"{node.module}.{alias.name}"
                            names.append((full_name, root))
                        else:
                            names.append((node.module, root))
                else:
                    for alias in node.names:
                        names.append((alias.name, alias.name.split('.')[0]))
            return names

        def visit_Import(self, node):
            for full_name, root_name in self._extract_import_names(node):
                if self.depth == 0:
                    top_level_imports.add(root_name)
                else:
                    context = ' -> '.join(self.context_stack)
                    nested_imports[full_name] = f"nested in {context}"
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            # Skip relative imports (from . import X) - they're package-internal
            if node.level > 0:
                self.generic_visit(node)
                return

            for full_name, root_name in self._extract_import_names(node):
                if self.depth == 0:
                    top_level_imports.add(root_name)
                else:
                    context = ' -> '.join(self.context_stack)
                    nested_imports[full_name] = f"nested in {context}"
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            self.depth += 1
            self.context_stack.append(f"function {node.name}")
            self.generic_visit(node)
            self.context_stack.pop()
            self.depth -= 1

        def visit_AsyncFunctionDef(self, node):
            self.depth += 1
            self.context_stack.append(f"async function {node.name}")
            self.generic_visit(node)
            self.context_stack.pop()
            self.depth -= 1

        def visit_ClassDef(self, node):
            self.depth += 1
            self.context_stack.append(f"class {node.name}")
            self.generic_visit(node)
            self.context_stack.pop()
            self.depth -= 1

        def visit_If(self, node):
            self.depth += 1
            self.context_stack.append("conditional")
            self.generic_visit(node)
            self.context_stack.pop()
            self.depth -= 1

        def visit_Try(self, node):
            self.depth += 1
            self.context_stack.append("try/except")
            self.generic_visit(node)
            self.context_stack.pop()
            self.depth -= 1

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == '__import__':
                if node.args and isinstance(node.args[0], ast.Constant):
                    module_name = node.args[0].value
                    if self.depth > 0:
                        context = ' -> '.join(self.context_stack)
                        nested_imports[module_name] = f"__import__ nested in {context}"
                    else:
                        top_level_imports.add(module_name.split('.')[0])

            elif isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module':
                if node.args and isinstance(node.args[0], ast.Constant):
                    module_name = node.args[0].value
                    if self.depth > 0:
                        context = ' -> '.join(self.context_stack)
                        nested_imports[module_name] = f"import_module nested in {context}"
                    else:
                        top_level_imports.add(module_name.split('.')[0])

            self.generic_visit(node)

    finder = ImportFinder()
    finder.visit(tree)

    novel_imports = {}
    for nested_module, location in nested_imports.items():
        root = nested_module.split('.')[0]
        if root in top_level_imports:
            continue
        if '.' in nested_module:
            continue
        novel_imports[nested_module] = location

    return novel_imports


def analyze_module_imports(cpython_lib_path: Path, module_name: str) -> dict[str, str]:
    module_file = cpython_lib_path / f"{module_name}.py"

    if not module_file.exists():
        module_dir = cpython_lib_path / module_name
        if module_dir.exists():
            module_file = module_dir / "__init__.py"

    if not module_file.exists():
        return {}

    nested = find_non_toplevel_imports(module_file)
    if nested:
        print(f"  {module_name}:", file=sys.stderr)
        for mod, loc in nested.items():
            print(f"    - {mod}: {loc}", file=sys.stderr)

    return {mod: f"Found in {module_name}: {loc}" for mod, loc in nested.items()}


def find_imports_via_sys_modules(module_name: str) -> set[str]:
    before = set(sys.modules.keys())
    try:
        if module_name in sys.modules:
            del sys.modules[module_name]
        __import__(module_name)
    except Exception as e:
        print(f"Warning: Could not import {module_name}: {e}", file=sys.stderr)
        return set()

    after = set(sys.modules.keys())
    new_modules = after - before
    return {m for m in new_modules if m.startswith('_') or '.' in m}


def validate_importable(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False
    except Exception as e:
        print(f"  Warning: Unexpected error importing {module_name}: {e}", file=sys.stderr)
        return False


def generate_prelude(cpython_path: Path | None = None) -> str:
    print("Analyzing module dependencies...", file=sys.stderr)

    all_deps: dict[str, str] = {}
    if cpython_path:
        print(
            f"\n1. Scanning CPython source for nested imports at {cpython_path}...", file=sys.stderr
        )
        print("   (Looking for imports inside functions/classes/conditionals)\n", file=sys.stderr)

        # Scan desired modules
        to_scan = list(DESIRED_MODULES)
        scanned = set()

        while to_scan:
            module = to_scan.pop(0)
            if module in scanned:
                continue
            scanned.add(module)

            nested = analyze_module_imports(cpython_path, module)
            for dep, reason in nested.items():
                if dep not in all_deps:
                    all_deps[dep] = reason
                    if dep.startswith('_') and '.' not in dep and dep not in scanned:
                        to_scan.append(dep)

        if not all_deps:
            print("  No nested imports found!", file=sys.stderr)
        else:
            print(f"\n  Found {len(all_deps)} nested dependencies", file=sys.stderr)
    else:
        print("\n1. Skipping CPython source analysis (no --cpython-path provided)", file=sys.stderr)
        print("   Provide --cpython-path to detect nested imports", file=sys.stderr)

    print(
        "\n2. Checking sys.modules for C extensions and immediate dependencies...", file=sys.stderr
    )
    for module in DESIRED_MODULES:
        sys_deps = find_imports_via_sys_modules(module)
        for dep in sys_deps:
            if dep.startswith('_') and '.' not in dep and dep not in all_deps:
                all_deps[dep] = f"C extension or immediate dependency of {module}"
                print(f"  {dep}: immediate dep of {module}", file=sys.stderr)

    print("\n3. Validating that discovered modules are importable...", file=sys.stderr)
    invalid_modules = []
    for module_name in list(all_deps.keys()):
        if not validate_importable(module_name):
            invalid_modules.append(module_name)
            print(f"  ✗ {module_name}: cannot import, excluding", file=sys.stderr)
            del all_deps[module_name]

    if invalid_modules:
        print(f"\n  Excluded {len(invalid_modules)} non-importable modules", file=sys.stderr)
    else:
        print("  ✓ All discovered modules are importable", file=sys.stderr)

    content = '''"""
Prelude module for REPL environment.

This module imports all standard library modules that should be available
in the REPL namespace. Imports must be at the top level (not exec'd) to
satisfy componentize-py's build-time dependency resolution requirement.

See: https://github.com/bytecodealliance/componentize-py?tab=readme-ov-file#known-limitations

## Auto-generated

This file is AUTO-GENERATED by generate_prelude.py. Do not edit manually!
To regenerate: uv run generate_prelude.py
"""

# ruff: noqa: F401, I001
# fmt: off

'''

    filtered_deps = {
        k: v for k, v in all_deps.items() if k not in DESIRED_MODULES and k not in EXCLUDED_MODULES
    }

    if filtered_deps:
        content += (
            "# ============================================================================\n"
        )
        content += "# Nested/Lazy Dependencies (Non-Top-Level Imports)\n"
        content += "# These imports are NOT at module top-level - they're inside functions,\n"
        content += "# classes, or conditionals. Must be explicit for componentize-py.\n"
        content += (
            "# ============================================================================\n\n"
        )
        for dep in sorted(filtered_deps.keys()):
            reason = filtered_deps[dep]
            content += f"import {dep}  # {reason}\n"
        content += "\n"

    content += "# ============================================================================\n"
    content += "# Main Stdlib Modules\n"
    content += "# These are the modules available in the REPL namespace\n"
    content += "# ============================================================================\n\n"
    for module in sorted(DESIRED_MODULES):
        content += f"import {module}\n"

    content += "\n"
    content += "# Reference desired modules\n"
    content += "__modules__ = [\n"
    for module in sorted(DESIRED_MODULES):
        content += f"    {module},\n"
    content += "]\n"

    content += "\n"
    content += "# Export all imported modules\n"
    content += "__all__ = [\n"
    content += "    '__modules__',\n"

    # Export all nested dependencies
    for dep in sorted(filtered_deps.keys()):
        content += f"    '{dep}',\n"

    # Export all main modules
    for module in sorted(DESIRED_MODULES):
        content += f"    '{module}',\n"

    content += "]\n"

    return content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cpython-path',
        type=Path,
        help='Path to CPython Lib directory for source analysis (auto-detected if not provided)',
        default=None,
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file path',
        default=Path(__file__).parent / 'the_prelude.py',
    )

    args = parser.parse_args()

    # Auto-detect CPython path if not provided
    cpython_path = args.cpython_path
    if not cpython_path:
        import os

        cpython_path = Path(os.path.dirname(os.__file__))
        print(f"Auto-detected CPython path: {cpython_path}", file=sys.stderr)

    if cpython_path:
        if not cpython_path.exists():
            print(f"Error: CPython path does not exist: {cpython_path}", file=sys.stderr)
            print(
                "Try: python generate_prelude.py --cpython-path /usr/lib/python3.11",
                file=sys.stderr,
            )
            sys.exit(1)
        if not (cpython_path / 'os.py').exists():
            print(
                f"Error: {cpython_path} doesn't look like Python's Lib directory",
                file=sys.stderr,
            )
            print("Should contain files like os.py, sys.py, etc.", file=sys.stderr)
            sys.exit(1)

    content = generate_prelude(cpython_path)

    args.output.write_text(content)

    print(f"\n{'=' * 70}", file=sys.stderr)
    print(f"✓ Generated {args.output}", file=sys.stderr)
    print(f"{'=' * 70}\n", file=sys.stderr)


if __name__ == '__main__':
    main()

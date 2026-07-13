"""Import every agentmaker submodule to catch broken top-level imports.

Run with all extras installed (`uv sync --all-extras`) to catch package-level import
conflicts. Optional SDKs are imported lazily inside operations, so their behavior is
covered by the all-extras pytest job rather than this smoke check. Exits non-zero and
lists the offenders if any module fails to import.
"""

import importlib
import pkgutil

import agentmaker

errors: list[str] = []


def _record(name: str) -> None:
    errors.append(name)


for module in pkgutil.walk_packages(
    agentmaker.__path__, agentmaker.__name__ + ".", onerror=_record
):
    if module.name.rsplit(".", 1)[-1] == "__main__":  # importing a __main__ would run it
        continue
    try:
        importlib.import_module(module.name)
    except Exception as exc:
        errors.append(f"{module.name}: {exc!r}")

if errors:
    print("Import failures:")
    for item in errors:
        print(f"  - {item}")
    raise SystemExit(1)

print("All agentmaker modules imported OK.")

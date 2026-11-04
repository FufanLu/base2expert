"""Patch a module of verl as it is imported, from sitecustomize (cluster/mopd_site).

The parts of verl that P8 changes run in Ray processes (the TaskRunner, the actor worker), where
code in the job's main process cannot reach them; a hook installed at interpreter start-up in
every process can.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from collections.abc import Callable
from types import ModuleType


class _Hook(importlib.abc.MetaPathFinder):
    def __init__(self, target: str, patch: Callable[[ModuleType], None]) -> None:
        self.target = target
        self.patch = patch

    def find_spec(self, name, path, target=None):
        if name != self.target:
            return None
        sys.meta_path.remove(self)  # find the real spec, then wrap its loader
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        run = loader.exec_module
        patch = self.patch

        def exec_module(module: ModuleType) -> None:
            run(module)
            patch(module)

        loader.exec_module = exec_module  # type: ignore[method-assign]
        return spec


def on_import(target: str, patch: Callable[[ModuleType], None]) -> None:
    """Patch the module now if it is imported, else when it is."""
    if target in sys.modules:
        patch(sys.modules[target])
    elif not any(isinstance(f, _Hook) and f.target == target for f in sys.meta_path):
        sys.meta_path.insert(0, _Hook(target, patch))

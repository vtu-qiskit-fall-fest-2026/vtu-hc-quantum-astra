"""
Quantum Astra — Offline Grover Search Lab
VTU–Qiskit Fall Fest 2026 (Project ID: VTU-QFF-HC-0008)
"""

import sys
import os
import importlib.abc
import importlib.util

__version__ = "1.0.0"
__project__ = "Quantum Astra"
__project_id__ = "VTU-QFF-HC-0008"


class _DirectFileFinder(importlib.abc.MetaPathFinder):
    """
    Direct file finder that ensures seamless module resolution across
    different environments, container mounts, and local execution paths.
    """
    def find_spec(self, fullname, path, target=None):
        parts = fullname.split(".")
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(cur_dir)

        candidates = []
        if parts[0] == "src":
            sub = parts[1:]
            if sub:
                candidates.append(os.path.join(cur_dir, *sub) + ".py")
                candidates.append(os.path.join(cur_dir, *sub, "__init__.py"))
            else:
                candidates.append(os.path.join(cur_dir, "__init__.py"))
        else:
            candidates.append(os.path.join(cur_dir, *parts) + ".py")
            candidates.append(os.path.join(root_dir, *parts) + ".py")
            candidates.append(os.path.join(cur_dir, *parts, "__init__.py"))
            candidates.append(os.path.join(root_dir, *parts, "__init__.py"))

        for cand in candidates:
            if os.path.isfile(cand):
                return importlib.util.spec_from_file_location(fullname, cand)
        return None


# Install finder if not already present
if not any(isinstance(f, _DirectFileFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _DirectFileFinder())

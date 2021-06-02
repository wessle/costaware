import os
import importlib


def module_from_path(module_name, module_path):
    """Returns a module from an absolute path to the module."""

    spec = importlib.util.spec_from_file_location(module_name,
                                                  module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

"""Registry-based adapter framework for bridging HT_H3_BUNDLE to other packs.

An adapter subclasses AdapterBase and lives in an adapter_*.py next to this
file. It is discovered automatically; when its target package is missing,
available() is False, the bridge nodes hide that entry, and only an INFO line
is logged — never an error at import time. Conversion failures raise with the
target pack named.
"""

import importlib
import logging
import os

log = logging.getLogger("HT.compat")

_ADAPTERS = {}


class AdapterBase:
    """One external bundle format. Subclasses set name/target_package and
    implement the conversions they support."""

    name = ""
    target_package = ""
    external_type = ""

    @classmethod
    def available(cls):
        if not cls.target_package:
            return False
        try:
            importlib.import_module(cls.target_package)
            return True
        except ImportError:
            return False

    @classmethod
    def bundle_to_external(cls, ht_bundle):
        raise NotImplementedError(
            "adapter %s does not implement HT -> %s" % (cls.name, cls.external_type))

    @classmethod
    def bundle_from_external(cls, ext_bundle):
        raise NotImplementedError(
            "adapter %s does not implement %s -> HT" % (cls.name, cls.external_type))


def register_adapter(cls):
    if not (isinstance(cls, type) and issubclass(cls, AdapterBase)):
        raise ValueError("register_adapter expects an AdapterBase subclass")
    if not cls.name:
        raise ValueError("adapter %s must set name" % cls.__name__)
    _ADAPTERS[cls.name] = cls
    return cls


def _discover():
    here = os.path.dirname(__file__)
    for fn in sorted(os.listdir(here)):
        if fn.startswith("adapter_") and fn.endswith(".py"):
            mod = "." + fn[:-3]
            try:
                importlib.import_module(mod, package=__package__)
            except Exception:
                log.warning("HT compat: adapter module %s failed to load", fn)
    return dict(_ADAPTERS)


def adapters():
    return _ADAPTERS


def available_adapters():
    out = {}
    for name, adapter in _ADAPTERS.items():
        if adapter.available():
            out[name] = adapter
        else:
            log.info("HT compat: target package for adapter '%s' (%s) not installed, "
                     "bridge entry hidden", name, adapter.target_package or "?")
    return out


def get_adapter(name):
    adapter = _ADAPTERS.get(name)
    if adapter is None:
        raise ValueError("HT Compat: unknown target format %r (registered: %s)"
                         % (name, sorted(_ADAPTERS)))
    if not adapter.available():
        raise ValueError("HT Compat: format %r needs package %r, which is not installed"
                         % (name, adapter.target_package))
    return adapter


def to_external(name, ht_bundle):
    try:
        return get_adapter(name).bundle_to_external(ht_bundle)
    except Exception as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("HT Compat:"):
            raise
        raise RuntimeError("HT Compat: converting HT_H3_BUNDLE -> '%s' failed in "
                           "package %s: %s" % (name, get_adapter(name).target_package, exc))


def from_external(name, ext_bundle):
    try:
        return get_adapter(name).bundle_from_external(ext_bundle)
    except Exception as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("HT Compat:"):
            raise
        raise RuntimeError("HT Compat: converting '%s' -> HT_H3_BUNDLE failed in "
                           "package %s: %s" % (name, get_adapter(name).target_package, exc))

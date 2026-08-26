# Architecture

## Adding an adapter

1. Create `compat/adapter_<target>.py`.
2. Import `AdapterBase` and `register_adapter` from `compat.registry`.
3. Set a human-readable `name`, `target_package`, and `external_type`.
4. Make `available()` return false when the target package cannot be imported.
5. Implement `bundle_to_external` and `bundle_from_external`, mapping every field
   explicitly. Do not return a partially mapped object.
6. Decorate the class with `@register_adapter`.

The registry discovers adapter modules at package startup. The generic Compat
Out/In nodes obtain the currently available formats at runtime. Business nodes
only consume and produce `HT_H3_BUNDLE`; they must never import a third-party
node package or branch on an external bundle type.

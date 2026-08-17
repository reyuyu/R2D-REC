# -*- coding: utf-8 -*-
"""TRL 0.24.0 import fix: neutralize tuple-truthy availability flags for
NOT-installed optional dependencies (mergekit, llm_blender, weave, ...).
Fixes TRL bug where is_xxx_available() returns (False, None) which is truthy.
Only flags whose first element is False are set to False; installed packages
are untouched. Must be imported BEFORE trl.trainer.grpo_trainer."""
import trl.import_utils as _iu

_PATCHED = []
for _name in dir(_iu):
    if _name.startswith("_") and _name.endswith("_available"):
        _v = getattr(_iu, _name)
        if isinstance(_v, tuple) and len(_v) >= 1 and _v[0] is False:
            setattr(_iu, _name, False)
            _PATCHED.append(_name)

# also patch nested import_utils of trl.trainer if it re-exports flags
try:
    import trl.trainer.import_utils as _tiu  # may not exist
    for _name in dir(_tiu):
        if _name.startswith("_") and _name.endswith("_available"):
            _v = getattr(_tiu, _name)
            if isinstance(_v, tuple) and len(_v) >= 1 and _v[0] is False:
                setattr(_tiu, _name, False)
                _PATCHED.append(_name)
except Exception:
    pass

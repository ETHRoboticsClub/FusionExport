"""Fusion add-in entry point."""

import os
import sys
import traceback

_DIR = os.path.dirname(os.path.realpath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

_addin = None


def run(context):
    global _addin
    try:
        from fusion2usd.ui.addin import AddIn

        _addin = AddIn()
        _addin.start()
    except Exception:
        _report("Fusion2USD failed to start")


def stop(context):
    global _addin
    try:
        if _addin is not None:
            _addin.stop()
    except Exception:
        _report("Fusion2USD failed to stop cleanly")
    finally:
        _addin = None
        # Fusion keeps modules loaded between add-in restarts; drop ours so edits take effect.
        for name in [m for m in sys.modules if m == "fusion2usd" or m.startswith("fusion2usd.")]:
            del sys.modules[name]


def _report(title):
    import adsk.core

    adsk.core.Application.get().userInterface.messageBox(f"{title}:\n\n{traceback.format_exc()}")

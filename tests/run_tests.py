"""Minimal test runner (no pytest needed): python tests/run_tests.py"""
import importlib, sys, traceback, glob, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
fails = 0
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "test_*.py"))):
    mod = importlib.import_module(os.path.basename(f)[:-3])
    for name in dir(mod):
        if name.startswith("test_"):
            try:
                getattr(mod, name)(); print("PASS", mod.__name__, name)
            except Exception:
                fails += 1; print("FAIL", mod.__name__, name); traceback.print_exc()
sys.exit(1 if fails else 0)

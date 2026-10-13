"""The old desktop app's file name, kept so whatever still starts Fizgig by it - a pod image's start-up script,
an old shortcut - keeps working. The classic desktop app is gone: this runs fizgig.py, the web UI (--web serves
it; otherwise its own window)."""
import os
import runpy
import sys

if __name__ == "__main__":
    entry = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fizgig.py")
    sys.argv[0] = entry
    runpy.run_path(entry, run_name="__main__")

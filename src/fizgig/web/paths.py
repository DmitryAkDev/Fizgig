"""Where the web UI and the app window keep their own files: one user_data folder at Fizgig's root (git-ignored),
not a dozen files beside the launchers. The desktop app's own files (prefs.json, .last_used.json) stay where it
has always kept them."""
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
USER_DATA = os.path.join(BASE_DIR, "user_data")
os.makedirs(USER_DATA, exist_ok=True)


def user_file(name):
    return os.path.join(USER_DATA, name)

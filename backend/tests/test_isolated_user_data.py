"""No test can write the real user data directory.

conftest points ``CODEFYUI_USER_DATA_DIR`` at a temporary directory for the
whole session, before anything imports the app. Until it did, three test
files that drive the app's lifespan with ``with TestClient(app)`` rewrote the
real ``<user data>/codefyui/session.token``, and every CLI command talking to
a server running beside the test run was refused (403) until that server
restarted. The same variable roots the plugin lockfile, the pack control
files, the Codex login and the asset cache, so one redirect covers them all.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from platformdirs import user_data_dir

from app.config import settings
from app.core import auth, plugin_loader
from app.core.llm_proxy import codex_auth
from app.core.packs import paths as pack_paths
from tests import conftest


def test_every_user_data_path_a_test_can_write_is_outside_the_real_directory():
    real = Path(user_data_dir("codefyui", appauthor=False)).resolve()
    # Paths only: nothing here creates a directory, so a run without the
    # redirect fails without having touched the real one.
    paths = {
        "the session token (written by every lifespan)": auth.token_file_path(),
        "the plugin lockfile": plugin_loader.lockfile_path(),
        "settings.PLUGINS_USER_DIR (read once, at import)": settings.PLUGINS_USER_DIR,
        "the pack control files": pack_paths.control_dir(),
        "the Codex login": codex_auth.auth_file(),
    }
    real_ones = [f"{what}: {path}" for what, path in paths.items()
                 if path.resolve().is_relative_to(real)]
    assert not real_ones, (
        "these resolve inside the real user data directory, so a test run "
        "overwrites what a running server and the CLI use; conftest must set "
        "CODEFYUI_USER_DATA_DIR before the app is imported:\n  "
        + "\n  ".join(real_ones))


def test_the_session_uses_its_own_user_data_dir():
    """The variable names the directory conftest made for this session, not
    one the shell arrived with: a `cdui dev` shell exports it, pointing at a
    live `.codefyui_dev/`, which is outside the platformdirs directory the
    test above guards and is just as real."""
    value = os.environ["CODEFYUI_USER_DATA_DIR"]
    assert value == conftest._TEST_USER_DATA_DIR, value
    assert Path(value).name.startswith("codefyui-test-userdata-"), value
    assert value != conftest._REAL_USER_DATA_DIR, (
        "the session kept the CODEFYUI_USER_DATA_DIR it started with; conftest "
        "must replace it, not set it only when it is missing")


def test_a_value_the_shell_set_is_replaced(tmp_path):
    """The test above, in a second pytest started with the variable set.

    CI starts with it unset, and there a conftest that only filled it in
    when missing (``setdefault``) would pass the test above too.
    """
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CODEFYUI_USER_DATA_DIR"] = str(tmp_path / "set-by-the-shell")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"tests/{Path(__file__).name}::test_the_session_uses_its_own_user_data_dir"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=300,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, output
    assert "1 passed" in output, output

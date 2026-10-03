"""Shared fixtures: one small offline build per test process, cached."""
import atexit
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tdes  # noqa: E402,F401
from tdes.build import build  # noqa: E402
from tdes.config import small_config  # noqa: E402
from tdes.util import RunLog, rmtree_force  # noqa: E402

_CACHE = {}


def tmpdir():
    d = tempfile.mkdtemp(prefix="tdes_test_")
    atexit.register(rmtree_force, d)
    return d


def small_build():
    """Build corpus, tokenizer, shards, registry and schedule for the small config once."""
    if "art" not in _CACHE:
        art = os.path.join(tmpdir(), "art")
        cfg = small_config()
        build(art, cfg, RunLog(os.path.join(art, "run.log"), "test-build", echo=False))
        _CACHE["art"], _CACHE["cfg"] = art, cfg
    return _CACHE["art"], _CACHE["cfg"]


def env():
    from tdes.trainer import Env
    art, _ = small_build()
    if "env" not in _CACHE:
        _CACHE["env"] = Env(art)
    return _CACHE["env"]

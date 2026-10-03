"""Hashing, canonical JSON, hash-chained append-only ledgers and the run log."""
import datetime as _dt
import hashlib
import json
import os
import shutil
import stat
import sys
import time

import numpy as np


# ----------------------------------------------------------------------------- hashing
def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(*arrays) -> str:
    """Hash of dtype, shape and raw little-endian bytes of each array, in order."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype.str).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def seed_int(*parts) -> int:
    """Derive a 63-bit integer seed from any JSON-serialisable parts."""
    return int(sha256_json(list(parts))[:15], 16)


# ----------------------------------------------------------------------------- files
def write_json(path: str, obj, readonly: bool = False) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    if readonly:
        make_readonly(path)
    return path


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_readonly(path: str) -> None:
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def is_readonly(path: str) -> bool:
    return not (os.stat(path).st_mode & stat.S_IWRITE)


def rmtree_force(path: str) -> None:
    """Remove a tree that may contain read-only (sealed) files."""
    def _onerror(func, p, _exc):
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
        func(p)
    if os.path.exists(path):
        shutil.rmtree(path, onerror=_onerror)


def rel(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace("\\", "/")


# ----------------------------------------------------------------------------- ledger
class Ledger:
    """Append-only JSONL ledger. Every record carries its offset (`seq`), the hash of the
    previous record and its own hash, so any edit, deletion or reordering is detectable.
    Records are never rewritten; a crash rollback is itself an appended record."""

    GENESIS = "0" * 64

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.count = 0
        self.last_hash = self.GENESIS
        if os.path.exists(path):
            for rec in self.read(path):
                self.count = rec["seq"] + 1
                self.last_hash = rec["event_hash"]

    def append(self, event_type: str, payload: dict) -> dict:
        rec = {"seq": self.count, "type": event_type}
        rec.update(payload)
        rec["prev_hash"] = self.last_hash
        rec["event_hash"] = sha256_json(rec)
        line = canonical_json(rec) + "\n"
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        self.count += 1
        self.last_hash = rec["event_hash"]
        return rec

    @staticmethod
    def read(path: str) -> list:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def verify_chain(records: list):
        """Return (ok, first_bad_seq)."""
        prev = Ledger.GENESIS
        for i, rec in enumerate(records):
            body = {k: v for k, v in rec.items() if k != "event_hash"}
            if rec.get("seq") != i or rec.get("prev_hash") != prev or sha256_json(body) != rec.get("event_hash"):
                return False, i
            prev = rec["event_hash"]
        return True, None

    @staticmethod
    def effective(records: list) -> list:
        """Records minus everything a later `rollback` record superseded."""
        dropped = set()
        for rec in records:
            if rec["type"] == "rollback":
                dropped.update(rec["rolled_back_seqs"])
        return [r for r in records if r["seq"] not in dropped and r["type"] != "rollback"]


# ----------------------------------------------------------------------------- run log
class RunLog:
    """Human-readable execution log shared by every process of the demo (append mode)."""

    def __init__(self, path: str, component: str, echo: bool = True):
        self.path = path
        self.component = component
        self.echo = echo
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _write(self, level: str, msg: str) -> None:
        ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"{ts} {level:<6} [{self.component}] {msg}"
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
        if self.echo:
            try:
                sys.stdout.write(line + "\n")
            except UnicodeEncodeError:
                sys.stdout.write(line.encode("ascii", "replace").decode() + "\n")
            sys.stdout.flush()

    def info(self, msg: str) -> None:
        self._write("INFO", msg)

    def event(self, name: str, detail: str = "") -> None:
        """A pipeline-stage event such as `shards created`."""
        self._write("EVENT", f"{name}" + (f" | {detail}" if detail else ""))

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        tag = "[PASS]" if ok else "[FAIL]"
        self._write("CHECK", f"{tag} {name}" + (f" | {detail}" if detail else ""))
        return bool(ok)

    def section(self, title: str) -> None:
        self._write("INFO", "=" * 12 + f" {title} " + "=" * 12)


class Stopwatch:
    def __init__(self):
        self.totals = {}

    def add(self, key: str, seconds: float) -> None:
        self.totals[key] = self.totals.get(key, 0.0) + seconds

    class _Ctx:
        def __init__(self, sw, key):
            self.sw, self.key = sw, key

        def __enter__(self):
            self.t0 = time.perf_counter()

        def __exit__(self, *exc):
            self.sw.add(self.key, time.perf_counter() - self.t0)

    def time(self, key: str):
        return Stopwatch._Ctx(self, key)

import os
import shutil
import stat
import unittest

import numpy as np

from tests.helpers import env, small_build, tmpdir
from tdes.corpus import CLEANING_PIPELINE_HASH
from tdes.firewall import FirewallViolation, ngram_hashes
from tdes.packing import build_sequence, span_id
from tdes.shards import ShardIntegrityError, read_shard, validate_manifest
from tdes.util import Ledger, read_json


class ShardTests(unittest.TestCase):
    def test_every_manifest_validates(self):
        e = env()
        for sid, m in e.manifests.items():
            bad = [c for c, ok, _ in validate_manifest(e.art, m, e.tokenizer.hash, {CLEANING_PIPELINE_HASH}) if not ok]
            self.assertEqual(bad, [], sid)
            self.assertTrue(sid.endswith(m["content_hash"][:12]))

    def test_shard_files_are_sealed(self):
        e = env()
        m = e.manifests[e.catalog["admitted_train"][0]]
        for f in m["files"]:
            self.assertFalse(os.stat(os.path.join(e.art, m["path"], f)).st_mode & stat.S_IWRITE)

    def test_modified_shard_is_refused(self):
        e = env()
        m = dict(e.manifests[e.catalog["admitted_train"][0]])
        root = tmpdir()
        dst = os.path.join(root, m["path"])
        shutil.copytree(os.path.join(e.art, m["path"]), dst)
        p = os.path.join(dst, "tokens.npy")
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
        arr = np.load(p)
        arr[5] = (int(arr[5]) + 1) % 200
        np.save(p, arr)
        with self.assertRaises(ShardIntegrityError):
            read_shard(root, m)

    def test_wrong_tokenizer_hash_fails_validation(self):
        e = env()
        m = e.manifests[e.catalog["admitted_train"][0]]
        failed = [c for c, ok, _ in validate_manifest(e.art, m, "f" * 64, {CLEANING_PIPELINE_HASH}) if not ok]
        self.assertIn("tokenizer_hash", failed)

    def test_contaminated_shard_has_clean_derivative_with_lineage(self):
        e = env()
        cat = e.catalog["shards"]
        bad = [s for s, st in cat.items() if st == "quarantined_eval_overlap"]
        self.assertEqual(len(bad), 1)
        kids = [m for m in e.manifests.values() if bad[0] in m["parent_shard_ids"]]
        self.assertEqual(len(kids), 1)
        self.assertEqual(cat[kids[0]["shard_id"]], "admitted_train")
        self.assertEqual(kids[0]["n_docs"], e.manifests[bad[0]]["n_docs"] - 1)


class FirewallTests(unittest.TestCase):
    def test_ngram_fingerprint_detects_copy(self):
        a = np.arange(40) % 17 + 5
        b = np.concatenate([np.arange(30) + 300, a[10:30], np.arange(5) + 400])
        self.assertTrue(np.isin(ngram_hashes(b), ngram_hashes(a)).any())
        self.assertFalse(np.isin(ngram_hashes(np.arange(30) + 300), ngram_hashes(a)).any())

    def test_never_train_shards_refused_at_pool_entry(self):
        e = env()
        for sid in list(e.registry.test_shards) + list(e.registry.validation_shards) + list(e.registry.proxy_shards):
            ok, _ = e.firewall.request_training_admission(e.manifests[sid], actor="unittest")
            self.assertFalse(ok, sid)
        recs = Ledger.read(e.firewall.ledger.path)
        self.assertTrue(any(r["type"] == "shard_blocked" and r["actor"] == "unittest" for r in recs))

    def test_serve_time_check_blocks_eval_and_validation_tokens(self):
        e = env()
        L = e.cfg["train"]["seq_len"]
        for sid in (sorted(e.registry.test_shards)[0], sorted(e.registry.validation_shards)[0]):
            d = e.store.get(sid).docs[0]
            n = min(L, d["length"])
            seq = build_sequence([{"span_id": span_id(sid, 0, 0, n), "shard_id": sid, "doc_index": 0,
                                   "doc_id": d["doc_id"], "start": 0, "end": n, "epoch": 0}], e.store, L,
                                 "general_web", "concat_chop")
            with self.assertRaises(FirewallViolation):
                e.firewall.check_sequence(seq, e.store)

    def test_admitted_shards_carry_no_eval_overlap(self):
        e = env()
        for sid in e.catalog["admitted_train"]:
            self.assertEqual(e.firewall.scan_documents(e.store.get(sid), e.tokenizer), {}, sid)
            self.assertEqual(e.manifests[sid]["contamination_status"], "clean")

    def test_build_is_reproducible(self):
        from tdes.build import build
        from tdes.util import RunLog
        art, cfg = small_build()
        art2 = os.path.join(tmpdir(), "art")
        build(art2, cfg, RunLog(os.path.join(art2, "run.log"), "test", echo=False))
        a = read_json(os.path.join(art, "manifests", "catalog.json"))
        b = read_json(os.path.join(art2, "manifests", "catalog.json"))
        self.assertEqual(a["shards"], b["shards"], "same inputs -> same shard ids (content hashes)")
        self.assertEqual(read_json(os.path.join(art, "manifests", "tokenizer.lock.json"))["tokenizer_hash"],
                         read_json(os.path.join(art2, "manifests", "tokenizer.lock.json"))["tokenizer_hash"])


if __name__ == "__main__":
    unittest.main()

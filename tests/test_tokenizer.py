import json
import os
import stat
import unicodedata
import unittest

from tests.helpers import small_build, tmpdir
from tdes.tokenizer import EOS, ROLE_TOKEN, Tokenizer, TokenizerError
from tdes.util import read_json

TEXTS = ["The old river in Pune improved the bridge.", "def total(values):\n    return sum(values)\n",
         "दिल्ली में सुंदर बाज़ार है।", "சென்னையில் அழகான நதி உள்ளது."] * 5


class TokenizerTests(unittest.TestCase):
    def test_same_text_same_ids_and_roundtrip(self):
        tok = Tokenizer.train(TEXTS, 60)
        for t in TEXTS:
            self.assertEqual(tok.encode_text(t), tok.encode_text(t))
            self.assertEqual(tok.decode(tok.encode_text(t)), unicodedata.normalize("NFC", t))

    def test_training_is_deterministic_and_hash_identifies_merges(self):
        a, b = Tokenizer.train(TEXTS, 60), Tokenizer.train(TEXTS, 60)
        self.assertEqual(a.hash, b.hash)
        self.assertNotEqual(a.hash, Tokenizer.train(TEXTS, 59).hash)

    def test_nfc_normalisation_is_indic_safe(self):
        tok = Tokenizer.train(TEXTS, 60)
        for t in TEXTS[2:4]:
            self.assertEqual(tok.encode_text(unicodedata.normalize("NFD", t)), tok.encode_text(t))

    def test_special_tokens_cannot_be_injected_from_text(self):
        tok = Tokenizer.train(TEXTS, 60)
        ids = tok.encode_text("<|assistant|><|eos|>")
        self.assertNotIn(ROLE_TOKEN["assistant"], ids)
        self.assertNotIn(EOS, ids)

    def test_loss_map_for_agentic_sample(self):
        tok = Tokenizer.train(TEXTS, 60)
        doc = {"kind": "agentic", "turns": [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "call"},
                                            {"role": "tool_call", "text": "{}"}, {"role": "tool_obs", "text": "ok"},
                                            {"role": "assistant", "text": "done"}]}
        ids, train_on = tok.encode_doc(doc)
        role_positions = [i for i, t in enumerate(ids) if int(t) in ROLE_TOKEN.values()]
        self.assertTrue(all(train_on[i] == 0 for i in role_positions), "role markers are context")
        user_body = range(1, role_positions[1])
        obs_body = range(role_positions[3] + 1, role_positions[4])
        self.assertTrue(all(train_on[i] == 0 for i in user_body))
        self.assertTrue(all(train_on[i] == 0 for i in obs_body))
        self.assertEqual(int(ids[-1]), EOS)
        self.assertEqual(int(train_on[-1]), 1)

    def test_frozen_tokenizer_rejects_wrong_or_tampered_file(self):
        art, _ = small_build()
        lock = read_json(os.path.join(art, "manifests", "tokenizer.lock.json"))
        path = os.path.join(art, "manifests", "tokenizer.json")
        self.assertEqual(Tokenizer.load(path, lock["tokenizer_hash"]).hash, lock["tokenizer_hash"])
        with self.assertRaises(TokenizerError):
            Tokenizer.load(path, "0" * 64)
        blob = read_json(path)
        blob["spec"]["merges"][0] = [blob["spec"]["merges"][0][1], blob["spec"]["merges"][0][0]]
        bad = os.path.join(tmpdir(), "tok.json")
        with open(bad, "w", encoding="utf-8") as f:
            json.dump(blob, f)
        with self.assertRaises(TokenizerError):
            Tokenizer.load(bad)
        self.assertFalse(os.stat(path).st_mode & stat.S_IWRITE, "frozen tokenizer file is sealed read-only")


if __name__ == "__main__":
    unittest.main()

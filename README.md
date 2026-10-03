# V5 Training Data Execution System (ERA V5, Session 6)

This is a small but complete training data execution system. It takes documents all the way to an
audited training run and proves, from artifacts on disk, what the model consumed, why it consumed
it, what it learned from it and how the run can be reconstructed.

```
documents -> tokenized shards -> manifests -> mixture schedule -> packing -> batches -> training
-> consumption ledger -> learning ledger -> checkpoint -> crash -> resume -> replay -> fork -> audit
```

## Run it

```bash
python run_demo.py
```

It needs Python 3.10+, numpy and PyTorch (CPU is enough, no GPU):

```bash
pip install -r requirements.txt
```

The demo takes about 2 minutes on a laptop CPU, about half of which is the test suite. The
command deletes and regenerates `submission_artifacts/`, runs the 39 invariant tests, and exits 0
only if every requirement in the evidence bundle passes.

To train on an NVIDIA GPU instead, install a CUDA build of torch (for example
`pip install torch --index-url https://download.pytorch.org/whl/cu126`) and run:

```bash
python run_demo.py --device cuda
```

CPU stays the default because the demo must run on any machine. On the development laptop
(GTX 1660 Ti) the GPU run also passes every requirement, including bit-exact resume and replay,
but it is slower (about 4.0k vs 5.5k raw tokens/s). This model is so small that kernel-launch
overhead outweighs the GPU's compute, and most of the wall time is OPUS scoring and ledger I/O.
Weight hashes are reproducible within a device, not across devices.

To run only the tests:

```bash
python -m unittest discover -s tests -t . -v
```

## What the demo does

| # | Phase | Where |
|---|---|---|
| 1 | Generates a deterministic multi-lane corpus with provenance and licences, cleans it (NFC, control characters) and drops exact duplicates | `tdes/corpus.py` |
| 2 | Trains a byte-level BPE tokenizer, freezes it (read-only file, hash lock) and reloads it to verify the hash | `tdes/tokenizer.py` |
| 3 | Writes immutable shards (tokens, loss map, index; sealed read-only, content-hash ids) and their manifests | `tdes/shards.py` |
| 4 | Registers test/validation/proxy shards, quarantines a web shard that leaked a benchmark item, blocks an unknown-licence source, and runs a firewall drill | `tdes/firewall.py`, `tdes/build.py` |
| 5 | Compiles curriculum stages into per-step lane quotas with protected floors and an anneal fence | `tdes/mixture.py` |
| 6 | Runs an uninterrupted **reference** run, which defines the expected stream | `tdes/trainer.py` |
| 7 | Runs **main**: OPUS selects candidates, batches are packed, and the model trains with checkpoints every 4 steps. The process is then **killed** (`os._exit(86)`) inside step 21 | `tdes/loader.py`, `tdes/opus.py`, `tdes/model.py` |
| 8 | A new process **resumes** from the step-20 checkpoint and proves that its first batch is the expected one | `trainer.mode_resume` |
| 9 | **Replays** steps 9–16 from the ledger, starting at the step-8 checkpoint, and proves ids, spans, hashes and weights match | `trainer.mode_replay` |
| 10 | **Forks** a new data branch from the step-12 checkpoint with a changed mixture and OPUS threshold | `trainer.mode_fork` |
| 11 | **Audits** everything from disk, measures performance, runs the tests and writes the evidence bundle | `tdes/audit.py`, `tdes/performance.py`, `tdes/evidence.py` |

## Assignment checklist

Each item the assignment asks the system to demonstrate, where it is implemented, and which
generated artifact proves it.

| Requirement | Implementation | Proof in `submission_artifacts/` |
|---|---|---|
| Immutable tokenized shards with manifests | `shards.py`: content-hash ids, read-only files, hash re-verified on every load | `manifests/shards/*.json`, `reports/manifest_validation.json` |
| Frozen tokenizer and content hashes | `tokenizer.py`: canonical-spec hash, sealed file, lock file | `manifests/tokenizer.lock.json`, `[PASS] tokenizer_hash_verified` |
| Packing policies for different data types | `packing.py`: `concat_chop`, `best_fit_chunked`, `structure_preserving` | `reports/packing_policy_lab.json` |
| Correct loss masks, attention masks and position ids | `packing.build_sequence` / `verify_sequence`, used by the model | `reports/packed_batch_report.json`, `audit.mask_and_position_invariants` |
| Curriculum stages, lane weights and protected floors | `mixture.py`: per-step quotas, floors reserved first, anneal fence | `manifests/mixture_schedule.json`, `audit_report.json#mixture` |
| Evaluation and validation firewalls | `firewall.py`: registry, n-gram fingerprints, canaries, checks at admission, pool entry and serve time | `ledgers/firewall.jsonl`, `[PASS] eval_shard_blocked` |
| OPUS acceptance, rejection, deferral, protected-floor override | `opus.py`, `loader.py` | `ledgers/main/opus.jsonl`, `audit.all_decision_kinds_exercised` |
| Training consumption and learning ledgers | `trainer.py`, hash-chained `util.Ledger` | `ledgers/main/consumption.jsonl`, `learning.jsonl` |
| Token-level or sample-level loss tracking | Per-token CE and per-sample loss before/after each update | `ledgers/main/token_trace/`, `reports/learning_report.json` |
| Checkpoints tied to ledger offsets | `state.json` stores offset + last hash of every ledger | `checkpoints/*/step_*/state.json`, `audit.checkpoints_bound_to_ledger_offsets` |
| Crash recovery without skipped or repeated batches | Real `os._exit` mid-step, then resume with an explicit rollback record | `reports/resume_report.json`, `[PASS] resume_next_batch_matched` |
| Replay of the same historical data stream | `mode_replay` rebuilds batches from ledger span refs | `reports/replay_report.json`, `[PASS] replay_hash_matched` |
| Forking from an earlier checkpoint | `mode_fork` writes a `branch_forked` record and its own schedule | `reports/fork_report.json`, `ledgers/fork-*/` |
| Packing utilization and useful loss-bearing tokens/s | `performance.py`, reconciled against the audit's recount | `performance.json` |

The submission items are covered as well: one command (`python run_demo.py`), automated tests
(`tests/`, 39 tests), the execution log (`run.log`, with all 13 required events and the 5
required `[PASS]` lines), the evidence bundle (`evidence.json`, `evidence.md`), and the
generated manifests, ledgers, checkpoints and performance report.

## Architecture

```
               corpus.py            tokenizer.py              shards.py + firewall.py
 documents ──► clean + dedup ──► frozen BPE (hash) ──► sealed shards + manifests ──► admission / quarantine
                                                                  │                         │
                                       mixture.py                 ▼                         ▼
 curriculum stages ──► per-step quotas + floors ──► loader.py: lane streams ──► OPUS (opus.py) ──► Batch
                                                       (packing.py)               │   accept / reject /
                                                                                  │   defer / override
                                                                                  ▼
                       trainer.py: firewall re-check ─► packing invariants ─► microbatches x ranks ─► AdamW
                                       │                                                          │
                     ledgers/<branch>/consumption.jsonl   opus.jsonl   learning.jsonl   token_trace/
                                       │                                                          │
                     checkpoints/<branch>/step_N  = weights + optimizer + loader state + ledger offsets
                                       │
                     crash ─► resume (rollback record) ─► replay (from ledger) ─► fork (new branch id)
                                       │
                          audit.py ─► performance.py ─► evidence.py
```

### The model

The model is a one-block causal transformer in PyTorch (`tdes/model.py`), trained with
`torch.optim.AdamW`, global-norm clipping and a warmup-cosine schedule. Gradients come from
autograd, and the tests check them against finite differences. Checkpoints are `model.pt` and
`optimizer.pt` written with `torch.save`. The model deliberately consumes everything the data
system produces:

- token ids
- position ids, used to index learned position embeddings
- segment ids, used to build the block-causal attention mask
- the loss mask

It returns the cross-entropy of every token, which feeds the learning ledger. Torch runs with
`torch.use_deterministic_algorithms(True)`, one CPU thread, TF32 off and a fixed cuBLAS
workspace, so float reductions are deterministic on either device and resume and replay can be checked **bit for bit** through weight hashes. OPUS
scores each candidate with `torch.autograd.grad` and the cosine similarity to the proxy gradient.

## Design decisions

**Lanes follow the Session 5 (Ex05) V5 spec.** The lanes are `general_web`, `code`,
`math_science`, `indic` (Hindi and Tamil, as in the Ex03 tokenizer), `reasoning`, `agentic` and
`anneal_reserve`. The protected floors are Indic 12.5%, agentic 6.25% and reasoning 6.25%. There
are three stages, `foundation`, `reasoning_midtrain` and `anneal`. General web is cut to 0% in the
anneal stage, and the anneal reserve can only be drawn there.

**Frozen tokenizer.**
- The tokenizer hash is the sha256 of its canonical spec: normalisation, pretokenizer regex,
  special tokens, loss roles and merges.
- Every manifest and shard index records this hash, and the loader refuses any shard tokenized
  under a different hash.
- Pretokenization keeps Devanagari and Tamil combining marks inside words.
- Special tokens can only be produced structurally, so text can never inject `<|assistant|>`.

**Immutable shards.**
- A shard's id embeds its content hash, which covers the tokens, the loss map and the index. Its
  files are sealed read-only.
- Every load re-verifies the hash, so a modified shard can never be served.
- A contaminated shard is never edited. Instead, a new derived shard is written with
  `parent_shard_ids` lineage.

**One mask convention for every policy.** Masks can therefore be rebuilt from span references alone:

| Field | Rule |
|---|---|
| `segment_ids` | 1..k for each packed span, 0 for padding |
| `position_ids` | Restart at 0 in every segment |
| Attention | Token i may attend to token j iff they share a segment and j ≤ i |
| `labels[i]` | `tokens[i+1]` if that token is in the same segment, otherwise -1 |
| `loss_mask[i]` | 1 iff the target token i+1 is loss-bearing in the shard's loss map |

**Packing per data type** (see `reports/packing_policy_lab.json` for pad-only, greedy and best-fit comparisons):

| Lane | Policy | Why |
|---|---|---|
| web, math, Indic | `concat_chop` | Plain pretraining tolerates cuts. EOS marks document ends, and attention and positions reset at every boundary |
| code | `best_fit_chunked` | Files are split only at line boundaries, then packed best-fit-decreasing |
| reasoning, agentic, anneal | `structure_preserving` | Whole samples are never split. Context turns (user, tool_obs) carry no loss. Oversize samples are dropped and counted, never truncated |

**Mixture compiler.**
- Floors are reserved first. Remaining slots go to the lane with the most accumulated credit
  (weight × B per step).
- The cumulative realised share therefore stays within about one slot of the plan. The audit
  checks this, and also checks that consumption equals the compiled quotas exactly.
- Supply is checked against packable tokens, and lanes that need repetition are flagged (`scarcity`).

**OPUS.**
- Each candidate's gradient is compared, by cosine similarity, with the gradient of a trusted proxy
  set at the current weights. The proxy is English, code and maths only, which makes it
  deliberately biased.
- τ is the 30th percentile of the step's candidate scores.
- Per lane, candidates are **accepted** (above τ, within quota), **deferred** (above τ, quota full,
  re-offered next step), or **rejected**. The rejection reasons are `low_proxy_utility`,
  `quota_pressure`, `duplicate` and `stage_mismatch`.
- A protected lane below its floor takes its best rejected candidates as a
  **protected-floor override**. In the demo this mostly rescues Indic, which is the proxy-bias
  story from the lecture.
- Each decision is a ledger record holding the candidate id, shard ids, lane, stage, scoring
  checkpoint, model hash, proxy version, score, τ, status, reason, override flag and an
  effective-token estimate.

**Ledgers.**
- Ledgers are append-only JSONL. Each record holds its offset, `prev_hash` and `event_hash`, so any
  edit, deletion or reordering is detectable (a test tampers with one record and watches the audit
  fail).
- Each microbatch record holds the rank, microbatch id, sample ids, full span references
  (shard, doc, token range, epoch), the loss-mask hash, the attention and position policy, lane,
  stage, tokenizer version, dataloader version and OPUS decision id.
- The learning ledger holds loss before and after each update per sample and per span, perplexity,
  the most surprising tokens, grad norm, model phase, repeated-pass number and the checkpoint
  before the update.
- `token_trace/step_N.npz` holds the cross-entropy of every loss-bearing token.

**Checkpoints bind model state to data state.** A checkpoint holds the weights, the AdamW moments,
the scheduler step, the loader state (stream cursors, deferred queue, duplicate window, candidate
counter), and the offset and last hash of every ledger. The `checkpoint_saved` record is appended
before the files are atomically renamed into place, so a checkpoint only exists once its ledger
record does.

**Crash, resume, replay and fork.**

- **Crash.** A real `os._exit` inside step 21 leaves an orphaned `batch_planned` record and 3
  `microbatch_consumed` records on disk.
- **Resume.** A new process loads the latest checkpoint and verifies the ledger offset and hash. It
  then appends a `rollback` record that supersedes, without deleting, everything written after the
  offset. Step 21's batch is checked against two sources: the plan the dead process wrote, and the
  uninterrupted reference run. The 3 microbatches that were consumed before the crash are checked
  to be re-served bit-identically. The audit then proves steps 1..32 were committed exactly once
  each, and that the final weights equal the reference run's.
- **Replay** never samples. It rebuilds every packed sample from the span references in the ledger,
  using the hash-verified shards, and compares sample ids, spans, sequence hashes, loss-mask
  hashes, batch hashes and post-step weights.
- **Fork** starts a new branch id whose first ledger record is `branch_forked`. That record holds
  the parent checkpoint, the parent ledger offset, the divergence step and the config diff. The
  forked mixture schedule is saved next to the original.

**Evidence is derived, never asserted.** `evidence.json` only aggregates checks that the build, the
trainer and the independent audit computed and wrote to `reports/`. The audit re-reads everything
from disk. For example, it re-scans every consumed token against the eval fingerprints, rebuilds
every consumed sample from its span references, and recounts the tokens that
`performance.json` reports.

## Generated artifacts (`submission_artifacts/`)

| Path | Contents |
|---|---|
| `run.log` | Full event sequence with `[PASS]` / `[FAIL]` lines (all processes append to it) |
| `evidence.json`, `evidence.md` | Per-requirement result, underlying checks, evidence pointers and key values |
| `performance.json` | Raw, useful (loss-bearing) and accepted tokens/s, packing utilization against a pad-only baseline, loader wait, OPUS rejection by lane, cache hit rate, resume and replay latency, and a reconciliation against the audit's recount |
| `manifests/` | `tokenizer.json` and its `.lock`, `shards/*.json`, `catalog.json`, `eval_registry.json` with fingerprints, `mixture_schedule*.json`, `cleaning_report.json` |
| `shards/` | Immutable tokenized shards |
| `ledgers/` | `firewall.jsonl`, plus `<branch>/{consumption,opus,learning}.jsonl` and `token_trace/` for `main`, `reference`, `replay-main-s00008` and `fork-*` |
| `checkpoints/` | `<branch>/step_N/{model.pt, optimizer.pt, state.json, COMPLETE}` |
| `reports/` | Build, admission, manifest validation, packing lab, packed-batch report, resume, replay, fork, audit, learning report cards, tests |
| `perf/` | Raw per-process counters and timings that `performance.json` aggregates |
| `corpus/` | The raw generated documents |

## Where each grading area is evidenced

| Area | Evidence |
|---|---|
| End-to-end execution | `run.log`: 13 required events in order, exit code 0 |
| Shards, manifests, tokenizer | `manifests/`, `reports/manifest_validation.json`, `tokenizer.lock.json` |
| Packing, masks, batches | `reports/packed_batch_report.json`, `audit_report.json#packing` (every consumed sample rebuilt and checked) |
| Mixture, floors, OPUS | `mixture_schedule.json`, `audit_report.json#mixture`, `ledgers/main/opus.jsonl` |
| Consumption and learning ledgers | `ledgers/main/*.jsonl`, `token_trace/`, `reports/learning_report.json` |
| Checkpoint, crash, resume, replay, fork | `reports/{resume,replay,fork}_report.json`, `audit_report.json#checkpoints` |
| Eval and validation firewall | `ledgers/firewall.jsonl`, `manifests/eval_registry.json`, `audit_report.json#firewall` |
| Throughput and packing efficiency | `performance.json` (reconciled against the ledger) |
| Tests and documentation | `tests/`, `reports/tests.log`, this README |

## Limitations and honest notes

- The corpus is synthetic and template-generated, so the model is tiny and runs on CPU. Loss falls
  from 6.47 to about 5.4 in 32 steps. The point is the data system, not model quality.
- Throughput numbers are wall-clock measurements on whatever machine runs the demo, so they vary
  between runs. The counts behind them do not vary and are reconciled exactly against the ledger.
  "Loader wait" is the single-process analogue of GPU idle time.
- The ledgers, manifests, shards and checkpoints are byte-reproducible for a given seed on a given
  machine. Wall-clock fields only appear in `run.log` and `perf/`.
- Git does not keep the read-only bit, so in a fresh clone the committed shards are writable.
  Their content and file hashes still verify. `python run_demo.py` regenerates and re-seals them.
- Agentic samples longer than the 128-token window are dropped under `structure_preserving`, never
  cut. They are counted in `run.log` (`every_active_lane_has_packable_supply`).

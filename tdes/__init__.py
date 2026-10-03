"""TDES: a small but complete Training Data Execution System (ERA V5, Session 6).

documents -> tokenized shards -> manifests -> mixture schedule -> packing -> batches
-> training -> consumption ledger -> learning ledger -> checkpoint -> crash -> resume
-> replay -> fork -> audit
"""
import os

# Bit-exact resume/replay needs deterministic BLAS reductions: pin every BLAS to one
# thread before numpy is imported anywhere in the package.
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
# cuBLAS needs a fixed workspace to be deterministic (only matters with --device cuda).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "1.0.0"

"""
ColabFold MSA Server + per-position conservation scores.
Writes _cons.npy in msa_dir and deletes temporary .a3m after compute.
"""
import io
import os
import tarfile
import time

import numpy as np
import requests

AA_STD = set("ACDEFGHIKLMNPQRSTVWY")


def _load_valid_cached_conservation(cons_path: str, expected_len: int):
    if not os.path.exists(cons_path):
        return None
    try:
        cached = np.load(cons_path)
        if (
            cached.ndim != 1
            or len(cached) != expected_len
            or not np.isfinite(cached).all()
        ):
            raise ValueError("invalid conservation cache")
        return cached.astype(np.float32, copy=False)
    except Exception:
        try:
            os.remove(cons_path)
        except OSError:
            pass
        return None


def fetch_msa_colabfold(
    sequence: str,
    protein_name: str,
    msa_dir: str,
    poll_interval: int = 15,
    max_wait: int = 7200,
) -> str:
    """
    Submit to ColabFold public API, download .a3m, return path.
    Skips if already cached.
    """
    os.makedirs(msa_dir, exist_ok=True)
    a3m_path = os.path.join(msa_dir, f"{protein_name}.a3m")
    if os.path.exists(a3m_path):
        return a3m_path

    base = "https://api.colabfold.com"
    response = requests.post(
        f"{base}/ticket/msa",
        data={"q": f">query\n{sequence}\n", "mode": "all"},
        timeout=60,
    )
    response.raise_for_status()
    job_id = response.json()["id"]
    print(f"  [ColabFold] {protein_name}: submitted job {job_id}")

    elapsed = 0
    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval
        try:
            status = requests.get(f"{base}/ticket/{job_id}", timeout=30).json()["status"]
        except Exception:
            continue

        if status == "COMPLETE":
            break
        if status == "ERROR":
            raise RuntimeError(f"ColabFold job failed: {protein_name}")
    else:
        raise TimeoutError(f"ColabFold timed out after {max_wait}s: {protein_name}")

    download = requests.get(f"{base}/result/download/{job_id}", timeout=120)
    download.raise_for_status()
    archive = tarfile.open(fileobj=io.BytesIO(download.content))

    for member in archive.getmembers():
        if member.name.endswith(".a3m"):
            extracted = archive.extractfile(member)
            if extracted is None:
                break
            with open(a3m_path, "wb") as output:
                output.write(extracted.read())
            return a3m_path

    raise ValueError(f"No .a3m in ColabFold result: {protein_name}")


def parse_a3m(a3m_path: str) -> list:
    """
    Parse a3m → list of strings, each length L (query length).
    Strips lower-case insertion columns.
    """
    sequences = []
    buffer = []

    with open(a3m_path) as handle:
        for line in handle:
            line = line.rstrip()
            if line.startswith(">"):
                if buffer:
                    sequences.append("".join(buffer))
                    buffer = []
            else:
                buffer.append("".join(c for c in line if c == "-" or c.isupper()))

    if buffer:
        sequences.append("".join(buffer))

    if not sequences:
        raise ValueError(f"Empty a3m: {a3m_path}")

    length = len(sequences[0])
    return [sequence for sequence in sequences if len(sequence) == length]


def compute_conservation(seqs: list) -> np.ndarray:
    """
    conservation[i] = 1 - H_norm(col i), H_norm = Shannon H / log2(20)
    Gaps count in denominator → gapped columns are penalised.
    Returns float32 array shape (L,) in [0, 1].
    """
    length = len(seqs[0])
    count = len(seqs)
    conservation = np.zeros(length, dtype=np.float32)

    for col in range(length):
        aa_counts = {}
        for sequence in seqs:
            aa = sequence[col]
            if aa in AA_STD:
                aa_counts[aa] = aa_counts.get(aa, 0) + 1

        freqs = np.array([value / count for value in aa_counts.values()], dtype=np.float64)
        entropy = float(-np.sum(freqs * np.log2(freqs + 1e-12)))
        conservation[col] = np.float32(max(0.0, 1.0 - entropy / np.log2(20)))

    return conservation


def get_conservation(
    protein_name: str,
    sequence: str,
    msa_dir: str,
    temp_msa_dir: str = "temp",
    force_recompute: bool = False,
    post_delay: float = 6.0,
) -> np.ndarray:
    """
    Returns (L,) float32 conservation.
    Final cache is written to msa_dir as <protein>_cons.npy.
    Temporary .a3m is downloaded into temp_msa_dir and removed after use.
    """
    os.makedirs(msa_dir, exist_ok=True)
    os.makedirs(temp_msa_dir, exist_ok=True)

    cons_path = os.path.join(msa_dir, f"{protein_name}_cons.npy")
    if not force_recompute:
        cached = _load_valid_cached_conservation(cons_path, len(sequence))
        if cached is not None:
            return cached

    a3m_path = os.path.join(temp_msa_dir, f"{protein_name}.a3m")
    if not os.path.exists(a3m_path) or force_recompute:
        fetch_msa_colabfold(sequence, protein_name, temp_msa_dir)
        time.sleep(post_delay)

    temp_cons_path = f"{cons_path}.tmp"
    try:
        seqs = parse_a3m(a3m_path)
        cons = compute_conservation(seqs)

        if len(cons) != len(sequence):
            raise ValueError(
                f"{protein_name}: cons len {len(cons)} != seq len {len(sequence)}"
            )

        with open(temp_cons_path, "wb") as handle:
            np.save(handle, cons)
        os.replace(temp_cons_path, cons_path)

        if os.path.exists(a3m_path):
            os.remove(a3m_path)

        return cons
    except Exception:
        if os.path.exists(temp_cons_path):
            try:
                os.remove(temp_cons_path)
            except OSError:
                pass
        if os.path.exists(a3m_path):
            try:
                os.remove(a3m_path)
            except OSError:
                pass
        raise
"""
Masked Marginal Scoring for ProSST — ProteinGym Benchmark

Protocol:
  For each mutated position i, mask that position in input_ids and run a
  forward pass. Score = log P(mt_i | masked context) - log P(wt_i | masked context).

  Three modes:
    --mode direct         → original single-pass scoring (baseline)
    --mode masked_seq     → mask sequence token only at position i
    --mode masked_both    → mask sequence AND structure token at position i

  Batching: all unique mutated positions are batched into one forward pass.
  For proteins with very long sequences, use --chunk_size to limit per-pass batch size.
"""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from Bio import SeqIO
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_gpu_ids(gpus_arg: str) -> list[int]:
    if not torch.cuda.is_available():
        return []

    n_available = torch.cuda.device_count()
    if gpus_arg.lower() == "all":
        return list(range(n_available))

    parsed = []
    for token in gpus_arg.split(","):
        token = token.strip()
        if not token:
            continue
        gpu_id = int(token)
        if 0 <= gpu_id < n_available:
            parsed.append(gpu_id)

    seen = set()
    unique = []
    for gpu_id in parsed:
        if gpu_id not in seen:
            unique.append(gpu_id)
            seen.add(gpu_id)
    return unique


def read_seq(fasta: str) -> str:
    for record in SeqIO.parse(fasta, "fasta"):
        return str(record.seq)
    raise ValueError(f"No sequence found in FASTA file: {fasta}")


def tokenize_structure_sequence(structure_sequence: list[int]) -> torch.Tensor:
    shifted = [i + 3 for i in structure_sequence]
    shifted = [1, *shifted, 2]
    return torch.tensor([shifted], dtype=torch.long)


def read_names(fasta_dir: str) -> list[str]:
    return [p.stem for p in Path(fasta_dir).glob("*.fasta")]


def parse_mutants(mutant_df: pd.DataFrame):
    parsed = []
    for mutant in mutant_df["mutant"].tolist():
        substitutions = []
        for sub in mutant.split(":"):
            wt_aa, pos_str, mt_aa = sub[0], sub[1:-1], sub[-1]
            substitutions.append((wt_aa, int(pos_str) - 1, mt_aa))
        parsed.append(substitutions)
    return parsed


@torch.no_grad()
def score_direct(model, input_ids, ss_input_ids, attention_mask, mutants_parsed, vocab):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ss_input_ids=ss_input_ids,
    )
    logits = torch.log_softmax(outputs.logits[:, 1:-1, :], dim=-1)[0]

    scores = []
    for substitutions in mutants_parsed:
        score = sum(
            logits[idx, vocab[mt]].item() - logits[idx, vocab[wt]].item()
            for wt, idx, mt in substitutions
        )
        scores.append(score)
    return scores


@torch.no_grad()
def score_masked_marginal(
    model,
    input_ids,
    ss_input_ids,
    attention_mask,
    mutants_parsed,
    vocab,
    mask_seq_token_id: int,
    mask_ss_token_id: int,
    mask_structure: bool = False,
    chunk_size: int = 256,
):
    unique_positions = sorted({idx for substitutions in mutants_parsed for _, idx, _ in substitutions})
    if not unique_positions:
        return [0.0] * len(mutants_parsed)

    n_positions = len(unique_positions)
    pos_to_batch_idx = {pos: i for i, pos in enumerate(unique_positions)}

    batched_seq = input_ids.repeat(n_positions, 1)
    batched_ss = ss_input_ids.repeat(n_positions, 1)
    batched_attn = attention_mask.repeat(n_positions, 1)

    for batch_idx, pos in enumerate(unique_positions):
        token_pos = pos + 1
        batched_seq[batch_idx, token_pos] = mask_seq_token_id
        if mask_structure:
            batched_ss[batch_idx, token_pos] = mask_ss_token_id

    all_logits = []
    start = 0
    active_chunk_size = max(1, chunk_size)
    while start < n_positions:
        end = min(start + active_chunk_size, n_positions)
        chunk_slice = slice(start, end)
        try:
            out = model(
                input_ids=batched_seq[chunk_slice],
                attention_mask=batched_attn[chunk_slice],
                ss_input_ids=batched_ss[chunk_slice],
            )
            token_positions = torch.tensor(
                [pos + 1 for pos in unique_positions[start:end]],
                device=out.logits.device,
                dtype=torch.long,
            )
            batch_indices = torch.arange(end - start, device=out.logits.device)
            masked_logits = out.logits[batch_indices, token_positions, :]
            masked_logits = torch.log_softmax(masked_logits, dim=-1).cpu()
            all_logits.append(masked_logits)
            start = end
        except torch.cuda.OutOfMemoryError:
            if not torch.cuda.is_available() or active_chunk_size == 1:
                raise
            torch.cuda.empty_cache()
            active_chunk_size = max(1, active_chunk_size // 2)
            print(
                f"OOM in masked marginal at chunk [{start}:{end}] - retrying with chunk_size={active_chunk_size}"
            )

    all_logits = torch.cat(all_logits, dim=0)

    scores = []
    for substitutions in mutants_parsed:
        score = 0.0
        for wt, idx, mt in substitutions:
            batch_idx = pos_to_batch_idx[idx]
            logits_row = all_logits[batch_idx, :]
            score += logits_row[vocab[mt]].item() - logits_row[vocab[wt]].item()
        scores.append(score)
    return scores


@torch.no_grad()
def score_protein(
    model,
    tokenizer,
    vocab,
    residue_sequence_dir,
    structure_sequence_dir,
    mutant_dir,
    output_mutant_dir,
    name,
    model_name,
    mode: str = "masked_seq",
    chunk_size: int = 256,
    ss_mask_token_id: int = 0,
):
    residue_fasta = Path(residue_sequence_dir) / f"{name}.fasta"
    structure_fasta = Path(structure_sequence_dir) / f"{name}.fasta"
    mutant_file = Path(mutant_dir) / f"{name}.csv"
    output_file = Path(output_mutant_dir) / f"{name}.csv"

    sequence = read_seq(str(residue_fasta))
    structure_sequence = [int(i) for i in read_seq(str(structure_fasta)).split(",")]

    ss_input_ids = tokenize_structure_sequence(structure_sequence).to(device)
    tok = tokenizer([sequence], return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    df = pd.read_csv(mutant_file)
    mutants_parsed = parse_mutants(df)

    if mode == "direct":
        scores = score_direct(
            model,
            input_ids,
            ss_input_ids,
            attention_mask,
            mutants_parsed,
            vocab,
        )
    elif mode == "masked_seq":
        scores = score_masked_marginal(
            model,
            input_ids,
            ss_input_ids,
            attention_mask,
            mutants_parsed,
            vocab,
            mask_seq_token_id=tokenizer.mask_token_id,
            mask_ss_token_id=ss_mask_token_id,
            mask_structure=False,
            chunk_size=chunk_size,
        )
    elif mode == "masked_both":
        scores = score_masked_marginal(
            model,
            input_ids,
            ss_input_ids,
            attention_mask,
            mutants_parsed,
            vocab,
            mask_seq_token_id=tokenizer.mask_token_id,
            mask_ss_token_id=ss_mask_token_id,
            mask_structure=True,
            chunk_size=chunk_size,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from direct, masked_seq, masked_both.")

    col = f"{model_name}_{mode}"
    df[col] = scores
    df.to_csv(output_file, index=False)
    corr = spearmanr(df["DMS_score"], df[col]).correlation
    print(f"[{mode}] {name}: {corr:.4f}")
    return corr


def main():
    global device

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--residue_dir", type=str, default="example_data/residue_sequence")
    parser.add_argument("--structure_dir", type=str, default="example_data/structure_sequence/2048")
    parser.add_argument("--mutant_dir", type=str, default="example_data/substitutions")
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="Comma-separated GPU ids for DataParallel (default: 0,1,2,3). Use 'all' for all visible GPUs.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="masked_seq",
        choices=["direct", "masked_seq", "masked_both"],
        help=(
            "direct: original single-pass scoring; "
            "masked_seq: mask sequence token; "
            "masked_both: mask sequence and structure tokens"
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="Max number of masked inputs per forward pass",
    )
    parser.add_argument(
        "--ss_mask_token_id",
        type=int,
        default=0,
        help="Structure token ID used for masking in masked_both mode (default: 0)",
    )
    args = parser.parse_args()

    output_dir = Path(args.mutant_dir).parent / f"substitutions_{args.mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output CSVs → {output_dir}")

    print("Loading model...")
    model = AutoModelForMaskedLM.from_pretrained(args.model_path, trust_remote_code=True)
    model.cls.predictions.decoder.weight = model.prosst.embeddings.word_embeddings.weight

    gpu_ids = parse_gpu_ids(args.gpus)
    if gpu_ids:
        device = torch.device(f"cuda:{gpu_ids[0]}")
        model = model.to(device)
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        device = torch.device("cpu")
        model = model.to(device)
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    vocab = tokenizer.get_vocab()
    model_name = Path(args.model_path).name

    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    print(f"Using device: {device}")
    print(f"Using GPU IDs: {gpu_ids if gpu_ids else 'CPU'}")
    print(f"tokenizer.mask_token_id: {tokenizer.mask_token_id}")
    print(f"ss_embeddings shape: {tuple(base_model.prosst.embeddings.ss_embeddings.weight.shape)}")
    print(f"ss_mask_token_id (active): {args.ss_mask_token_id}")

    protein_names = read_names(args.residue_dir)
    print(f"Scoring {len(protein_names)} proteins in mode={args.mode}")

    results = {}
    for name in tqdm(protein_names):
        corr = score_protein(
            model,
            tokenizer,
            vocab,
            residue_sequence_dir=args.residue_dir,
            structure_sequence_dir=args.structure_dir,
            mutant_dir=args.mutant_dir,
            output_mutant_dir=str(output_dir),
            name=name,
            model_name=model_name,
            mode=args.mode,
            chunk_size=args.chunk_size,
            ss_mask_token_id=args.ss_mask_token_id,
        )
        results[name] = corr

    mean_spearman = sum(results.values()) / len(results) if results else float("nan")
    print("\n" + "=" * 50)
    print(f"Mean Spearman ({args.mode}): {mean_spearman:.4f}")
    print("=" * 50)

    summary = pd.DataFrame.from_dict(results, orient="index", columns=["spearman"])
    summary.to_csv(output_dir / "summary.csv")
    print(f"Per-protein summary saved to {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

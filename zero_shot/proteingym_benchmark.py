from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from scipy.stats import spearmanr
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from argparse import ArgumentParser
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_seq(fasta):
    for record in SeqIO.parse(fasta, "fasta"):
        return str(record.seq)


def tokenize_structure_sequence(structure_sequence):
    shift_structure_sequence = [i + 3 for i in structure_sequence]
    shift_structure_sequence = [1, *shift_structure_sequence, 2]
    return torch.tensor(
        [
            shift_structure_sequence,
        ],
        dtype=torch.long,
    )


def read_names(fasta_dir):
    files = Path(fasta_dir).glob("*.fasta")
    names = [file.stem for file in files]
    return names


@torch.no_grad()
def score_protein(
    model,
    tokenizer,
    residue_sequence_dir: str,
    structure_sequence_dir: str,
    mutant_dir: str,
    output_mutant_dir: str,
    name: str,
    model_name: str,
):
    print(f"Scoring {name}...")
    residue_fasta = Path(residue_sequence_dir) / f"{name}.fasta"
    structure_fasta = Path(structure_sequence_dir) / f"{name}.fasta"
    mutant_file = Path(mutant_dir) / f"{name}.csv"
    output_mutant_file = Path(output_mutant_dir) / f"{name}.csv"
    sequence = read_seq(residue_fasta)
    structure_sequence = read_seq(structure_fasta)

    structure_sequence = [int(i) for i in structure_sequence.split(",")]
    ss_input_ids = tokenize_structure_sequence(structure_sequence).to(device)
    tokenized_results = tokenizer([sequence], return_tensors="pt")
    input_ids = tokenized_results["input_ids"].to(device)
    attention_mask = tokenized_results["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ss_input_ids=ss_input_ids,
        labels=input_ids,
    )

    logits = outputs.logits
    logits = torch.log_softmax(logits[:, 1:-1, :], dim=-1)

    df = pd.read_csv(mutant_file)
    mutants = df["mutant"].tolist()
    scores = []
    vocab = tokenizer.get_vocab()
    for mutant in mutants:
        pred_score = 0
        for sub_mutant in mutant.split(":"):
            wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
            score = logits[0, idx, vocab[mt]] - logits[0, idx, vocab[wt]]
            pred_score += score.item()
        scores.append(pred_score)

    df[model_name] = scores
    df.to_csv(output_mutant_file, index=False)
    corr = spearmanr(df["DMS_score"], df[model_name]).correlation
    print(f"{name}: {corr}")
    return corr


def conservation_gate(
    conservation: torch.Tensor,
    center: float = 0.5,
    sharpness: float = 8.0,
) -> torch.Tensor:
    return torch.sigmoid(sharpness * (conservation - center))


@torch.no_grad()
def score_protein_conservation(
    model,
    tokenizer,
    residue_sequence_dir: str,
    structure_sequence_dir: str,
    mutant_dir: str,
    output_mutant_dir: str,
    name: str,
    model_name: str,
    msa_dir: str,
    cons_center: float = 0.5,
    cons_sharpness: float = 8.0,
    zero_init: bool = False,
):
    try:
        from zero_shot.conservation_utils import get_conservation
    except ModuleNotFoundError:
        from conservation_utils import get_conservation

    print(f"Scoring {name} [conservation gate, zero_init={zero_init}]...")
    residue_fasta = Path(residue_sequence_dir) / f"{name}.fasta"
    structure_fasta = Path(structure_sequence_dir) / f"{name}.fasta"
    mutant_file = Path(mutant_dir) / f"{name}.csv"
    output_mutant_file = Path(output_mutant_dir) / f"{name}.csv"
    sequence = read_seq(residue_fasta)
    structure_sequence = read_seq(structure_fasta)
    structure_sequence = [int(i) for i in structure_sequence.split(",")]

    length = len(sequence)
    if zero_init:
        gate = torch.ones(length, dtype=torch.float32, device=device)
    else:
        cons_np = get_conservation(
            protein_name=name,
            sequence=sequence,
            msa_dir=msa_dir,
        )
        if len(cons_np) != length:
            print(
                f"  [warn] {name}: cons len {len(cons_np)} != seq len {length}. Using ones."
            )
            gate = torch.ones(length, dtype=torch.float32, device=device)
        else:
            cons_tensor = torch.tensor(cons_np, dtype=torch.float32, device=device)
            gate = conservation_gate(
                cons_tensor,
                center=cons_center,
                sharpness=cons_sharpness,
            )

    ss_input_ids = tokenize_structure_sequence(structure_sequence).to(device)
    tokenized_results = tokenizer([sequence], return_tensors="pt")
    input_ids = tokenized_results["input_ids"].to(device)
    attention_mask = tokenized_results["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ss_input_ids=ss_input_ids,
        labels=input_ids,
    )

    logits = outputs.logits
    logits = torch.log_softmax(logits[:, 1:-1, :], dim=-1)

    df = pd.read_csv(mutant_file)
    mutants = df["mutant"].tolist()
    scores = []
    vocab = tokenizer.get_vocab()

    for mutant in mutants:
        pred_score = 0.0
        for sub_mutant in mutant.split(":"):
            wt = sub_mutant[0]
            idx = int(sub_mutant[1:-1]) - 1
            mt = sub_mutant[-1]
            raw = logits[0, idx, vocab[mt]] - logits[0, idx, vocab[wt]]
            gated = raw * gate[idx]
            pred_score += gated.item()
        scores.append(pred_score)

    df[model_name] = scores
    df.to_csv(output_mutant_file, index=False)
    corr = spearmanr(df["DMS_score"], df[model_name]).correlation
    print(f"  {name}: Spearman ρ = {corr:.4f}")
    return corr


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--residue_dir",
        type=str,
        required=False,
        default="example_data/residue_sequence",
        help="Directory containing FASTA files of residue sequences",
    )
    parser.add_argument(
        "--structure_dir",
        type=str,
        required=False,
        default="example_data/structure_sequence/2048",
        help="Directory containing FASTA files of structure sequences",
    )
    parser.add_argument(
        "--mutant_dir",
        type=str,
        required=False,
        default="example_data/substitutions",
        help="Directory containing CSV files with mutants",
    )
    parser.add_argument(
        "--msa_dir",
        type=str,
        default=None,
        help="Dir with precomputed <protein>_cons.npy files. If set, conservation gating is used.",
    )
    parser.add_argument(
        "--zero_init",
        action="store_true",
        help="Use gate=ones (no gating). Sanity check: output must match baseline.",
    )
    parser.add_argument("--cons_center", type=float, default=0.5)
    parser.add_argument("--cons_sharpness", type=float, default=8.0)
    args = parser.parse_args()
    output_mutant_dir = Path(args.mutant_dir).parent / "substitutions_copy"
    output_mutant_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scored CSV outputs will be written to: {output_mutant_dir}")

    print("Loading model...")
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    model.cls.predictions.decoder.weight = (
        model.prosst.embeddings.word_embeddings.weight
    )
    assert model.cls.predictions.decoder.weight.data_ptr() == model.prosst.embeddings.word_embeddings.weight.data_ptr()
    print("Manually tied ✓")
    print("Decoder weight shape:", model.cls.predictions.decoder.weight.shape)
    print("Embedding weight shape:", model.prosst.embeddings.word_embeddings.weight.shape)

    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model_name = args.model_path.split("/")[-1]
    protein_names = read_names(args.residue_dir)
    print(f"Found {len(protein_names)} proteins.")
    print(protein_names)

    all_corrs = []
    for protein_name in protein_names:
        if args.msa_dir is not None or args.zero_init:
            corr = score_protein_conservation(
                model,
                tokenizer=tokenizer,
                residue_sequence_dir=args.residue_dir,
                structure_sequence_dir=args.structure_dir,
                mutant_dir=args.mutant_dir,
                output_mutant_dir=str(output_mutant_dir),
                model_name=model_name,
                name=protein_name,
                msa_dir=args.msa_dir,
                cons_center=args.cons_center,
                cons_sharpness=args.cons_sharpness,
                zero_init=args.zero_init,
            )
        else:
            corr = score_protein(
                model,
                tokenizer=tokenizer,
                residue_sequence_dir=args.residue_dir,
                structure_sequence_dir=args.structure_dir,
                mutant_dir=args.mutant_dir,
                output_mutant_dir=str(output_mutant_dir),
                model_name=model_name,
                name=protein_name,
            )
        if corr is not None:
            all_corrs.append(corr)

    if all_corrs:
        print(
            f"\nMean Spearman ρ across {len(all_corrs)} proteins: {np.mean(all_corrs):.4f}"
        )


if __name__ == "__main__":
    main()

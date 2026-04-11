from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from Bio import PDB
from argparse import ArgumentParser

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


def load_plddt_from_pdb(pdb_path: str, default: float = 80.0) -> torch.Tensor:
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("p", pdb_path)
    scores = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] != " ":
                    continue
                try:
                    scores.append(residue["CA"].get_bfactor())
                except KeyError:
                    scores.append(default)
    return torch.tensor(scores, dtype=torch.float32)


def plddt_gate(
    plddt: torch.Tensor,
    center: float = 70.0,
    sharpness: float = 0.15,
) -> torch.Tensor:
    return torch.sigmoid(sharpness * (plddt - center))


def _debug_embedding_output(model, input_ids, ss_input_ids, attention_mask):
    captured = {}

    def _probe(module, args, output):
        captured["output"] = output

    h = model.prosst.embeddings.register_forward_hook(_probe)
    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ss_input_ids=ss_input_ids,
            labels=input_ids,
        )
    h.remove()
    out = captured["output"]
    if isinstance(out, tuple):
        print(f"Embedding output is a {len(out)}-tuple")
        for i, t in enumerate(out):
            print(f"  [{i}] shape={t.shape}, dtype={t.dtype}")
    else:
        print(f"Embedding output is a single tensor: shape={out.shape}")


@torch.no_grad()
def score_protein_plddt(
    model,
    tokenizer,
    residue_sequence_dir: str,
    structure_sequence_dir: str,
    mutant_dir: str,
    output_mutant_dir: str,
    name: str,
    model_name: str,
    pdb_dir: str = None,
    plddt_center: float = 70.0,
    plddt_sharpness: float = 0.15,
    crystal_default_gate: float = 0.8,
    debug_embedding_hook: bool = False,
):
    print(f"Scoring {name} [pLDDT-gated]...")
    residue_fasta = Path(residue_sequence_dir) / f"{name}.fasta"
    structure_fasta = Path(structure_sequence_dir) / f"{name}.fasta"
    mutant_file = Path(mutant_dir) / f"{name}.csv"
    output_mutant_file = Path(output_mutant_dir) / f"{name}.csv"
    sequence = read_seq(residue_fasta)
    structure_sequence = read_seq(structure_fasta)

    structure_sequence = [int(i) for i in structure_sequence.split(",")]
    model_device = next(model.parameters()).device
    ss_input_ids = tokenize_structure_sequence(structure_sequence).to(model_device)
    tokenized_results = tokenizer([sequence], return_tensors="pt")
    input_ids = tokenized_results["input_ids"].to(model_device)
    attention_mask = tokenized_results["attention_mask"].to(model_device)

    L = input_ids.shape[1] - 2
    if pdb_dir is not None:
        pdb_path = Path(pdb_dir) / f"{name}.pdb"
        if pdb_path.exists():
            plddt = load_plddt_from_pdb(str(pdb_path)).to(model_device)
            if plddt.shape[0] != L:
                print(
                    f"  [warn] pLDDT length {plddt.shape[0]} != sequence length {L}; using default gate"
                )
                gate = torch.full((L,), crystal_default_gate, device=model_device)
            else:
                gate = plddt_gate(
                    plddt,
                    center=plddt_center,
                    sharpness=plddt_sharpness,
                )
        else:
            print(
                f"  [warn] PDB not found at {pdb_path}; using default gate {crystal_default_gate}"
            )
            gate = torch.full((L,), crystal_default_gate, device=model_device)
    else:
        gate = torch.full((L,), crystal_default_gate, device=model_device)

    gate_padded = F.pad(gate, (1, 1), value=1.0).unsqueeze(0).unsqueeze(-1)

    if debug_embedding_hook:
        _debug_embedding_output(model, input_ids, ss_input_ids, attention_mask)

    emb_module = model.prosst.embeddings

    def _plddt_gate_hook(module, args, output):
        if not isinstance(output, tuple) or len(output) < 2:
            raise RuntimeError(
                "Embedding output is not a 2-tuple; cannot apply pLDDT gate hook safely."
            )
        combined, ss_contrib = output[0], output[1]
        gated_ss = gate_padded * ss_contrib
        gated_combined = combined - ss_contrib + gated_ss
        return (gated_combined, gated_ss)

    hook = emb_module.register_forward_hook(_plddt_gate_hook)
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ss_input_ids=ss_input_ids,
            labels=input_ids,
        )
    finally:
        hook.remove()

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
    print(f"  {name}: {corr:.4f}")
    return corr


def read_names(fasta_dir):
    files = Path(fasta_dir).glob("*.fasta")
    names = [file.stem for file in files]
    return names


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
        "--pdb_dir",
        type=str,
        required=False,
        default=None,
        help="Directory containing PDB files named {protein}.pdb for pLDDT gating",
    )
    parser.add_argument(
        "--plddt_center",
        type=float,
        required=False,
        default=70.0,
        help="Sigmoid gate center for pLDDT gating",
    )
    parser.add_argument(
        "--plddt_sharpness",
        type=float,
        required=False,
        default=0.15,
        help="Sigmoid gate sharpness for pLDDT gating",
    )
    parser.add_argument(
        "--crystal_default_gate",
        type=float,
        required=False,
        default=0.8,
        help="Fallback gate value when pLDDT is unavailable",
    )
    parser.add_argument(
        "--debug_embedding_hook",
        action="store_true",
        help="Print embedding output shape info once per protein before gated forward",
    )
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

    print("Scoring proteins...")
    model_name = args.model_path.split("/")[-1]
    protein_names = read_names(args.residue_dir)
    print(protein_names)
    for protein_name in protein_names:
        score_protein_plddt(
            model,
            tokenizer=tokenizer,
            residue_sequence_dir=args.residue_dir,
            structure_sequence_dir=args.structure_dir,
            mutant_dir=args.mutant_dir,
            output_mutant_dir=str(output_mutant_dir),
            model_name=model_name,
            name=protein_name,
            pdb_dir=args.pdb_dir,
            plddt_center=args.plddt_center,
            plddt_sharpness=args.plddt_sharpness,
            crystal_default_gate=args.crystal_default_gate,
            debug_embedding_hook=args.debug_embedding_hook,
        )


if __name__ == "__main__":
    main()

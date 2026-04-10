import os
from argparse import ArgumentParser
from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn.functional as F
from Bio import SeqIO
from scipy.stats import spearmanr
from transformers import AutoModelForMaskedLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


def _get_model_device(model):
    return next(model.parameters()).device


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


def _resolve_attr_path(obj, path):
    current = obj
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _get_embedding_module(model):
    candidate_paths = [
        "esm.embeddings",
        "prosst.embeddings",
        "base_model.embeddings",
        "model.embeddings",
    ]
    for path in candidate_paths:
        module = _resolve_attr_path(model, path)
        if module is not None:
            return module
    raise AttributeError("Could not locate model embedding module for soft-token injection")


def _get_structure_embeddings(module):
    for attr in ["structure_embeddings", "ss_embeddings", "ss_embedding"]:
        if hasattr(module, attr):
            return getattr(module, attr)
    raise AttributeError("Could not locate structure embedding table on embedding module")


def _get_or_fit_projection(model, soft_embedding_dim: int, structure_vocab_size: int):
    hidden_dim = model.config.hidden_size
    key = f"{soft_embedding_dim}->{hidden_dim}:vocab{structure_vocab_size}"
    model_device = _get_model_device(model)
    if not hasattr(score_protein_soft, "_proj"):
        score_protein_soft._proj = {}
    if key in score_protein_soft._proj:
        return score_protein_soft._proj[key].to(model_device)

    static_dir = Path(__file__).resolve().parent.parent / "prosst" / "structure" / "static"
    cluster_model_path = static_dir / f"{structure_vocab_size}.joblib"
    if not cluster_model_path.exists():
        raise FileNotFoundError(
            f"Missing centroid file for vocab size {structure_vocab_size}: {cluster_model_path}"
        )

    centroids = torch.tensor(
        joblib.load(cluster_model_path).cluster_centers_,
        dtype=torch.float32,
        device=model_device,
    )
    if centroids.ndim != 2 or centroids.shape[1] != soft_embedding_dim:
        raise ValueError(
            f"Centroid shape mismatch: expected (*, {soft_embedding_dim}), got {tuple(centroids.shape)}"
        )

    emb_module = _get_embedding_module(model)
    structure_embeddings = _get_structure_embeddings(emb_module)
    struct_weight = structure_embeddings.weight.detach().to(model_device)

    if struct_weight.shape[0] == structure_vocab_size + 3:
        target = struct_weight[3 : 3 + structure_vocab_size]
    elif struct_weight.shape[0] >= structure_vocab_size:
        target = struct_weight[:structure_vocab_size]
    else:
        raise ValueError(
            f"Structure embedding table too small: {struct_weight.shape[0]} rows for vocab {structure_vocab_size}"
        )

    projection = torch.linalg.lstsq(centroids, target).solution
    score_protein_soft._proj[key] = projection.detach().cpu()
    return projection


def _load_centroids(structure_vocab_size: int, soft_embedding_dim: int, model_device):
    key = f"{structure_vocab_size}:{soft_embedding_dim}"
    if not hasattr(score_protein_soft, "_centroids"):
        score_protein_soft._centroids = {}
    if key not in score_protein_soft._centroids:
        static_dir = Path(__file__).resolve().parent.parent / "prosst" / "structure" / "static"
        cluster_model_path = static_dir / f"{structure_vocab_size}.joblib"
        if not cluster_model_path.exists():
            raise FileNotFoundError(
                f"Missing centroid file for vocab size {structure_vocab_size}: {cluster_model_path}"
            )
        centroids = torch.tensor(
            joblib.load(cluster_model_path).cluster_centers_,
            dtype=torch.float32,
        )
        if centroids.ndim != 2 or centroids.shape[1] != soft_embedding_dim:
            raise ValueError(
                f"Centroid shape mismatch: expected (*, {soft_embedding_dim}), got {tuple(centroids.shape)}"
            )
        score_protein_soft._centroids[key] = centroids
    return score_protein_soft._centroids[key].to(model_device)


def _get_structure_token_embedding_table(model, structure_vocab_size: int):
    emb_module = _get_embedding_module(model)
    structure_embeddings = _get_structure_embeddings(emb_module)
    struct_weight = structure_embeddings.weight.detach()
    if struct_weight.shape[0] == structure_vocab_size + 3:
        return struct_weight[3 : 3 + structure_vocab_size], emb_module, structure_embeddings
    if struct_weight.shape[0] >= structure_vocab_size:
        return struct_weight[:structure_vocab_size], emb_module, structure_embeddings
    raise ValueError(
        f"Structure embedding table too small: {struct_weight.shape[0]} rows for vocab {structure_vocab_size}"
    )


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
    model_device = _get_model_device(model)
    sequence = read_seq(residue_fasta)
    structure_sequence = read_seq(structure_fasta)

    structure_sequence = [int(i) for i in structure_sequence.split(",")]
    ss_input_ids = tokenize_structure_sequence(structure_sequence).to(model_device)
    tokenized_results = tokenizer([sequence], return_tensors="pt")
    input_ids = tokenized_results["input_ids"].to(model_device)
    attention_mask = tokenized_results["attention_mask"].to(model_device)

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


@torch.no_grad()
def score_protein_soft(
    model,
    tokenizer,
    residue_sequence_dir: str,
    soft_embeddings_dir: str,
    mutant_dir: str,
    output_mutant_dir: str,
    name: str,
    model_name: str,
    soft_embedding_dim: int = 256,
    structure_vocab_size: int = 2048,
    soft_temperature: float = None,
):
    print(f"Scoring {name} [soft tokens]...")
    residue_fasta = Path(residue_sequence_dir) / f"{name}.fasta"
    soft_emb_file = Path(soft_embeddings_dir) / f"{name}.pt"
    mutant_file = Path(mutant_dir) / f"{name}.csv"
    output_mutant_file = Path(output_mutant_dir) / f"{name}.csv"

    sequence = read_seq(residue_fasta)
    model_device = _get_model_device(model)
    soft_struct_emb = torch.load(soft_emb_file, map_location=model_device)

    tokenized_results = tokenizer([sequence], return_tensors="pt")
    input_ids = tokenized_results["input_ids"].to(model_device)
    attention_mask = tokenized_results["attention_mask"].to(model_device)

    hidden_dim = model.config.hidden_size
    token_embedding_table, emb_module, structure_embeddings = _get_structure_token_embedding_table(
        model=model,
        structure_vocab_size=structure_vocab_size,
    )

    if soft_struct_emb.ndim != 2 or soft_struct_emb.shape[1] != soft_embedding_dim:
        raise ValueError(
            f"Expected soft embedding shape (L, {soft_embedding_dim}), got {tuple(soft_struct_emb.shape)}"
        )

    residue_len = input_ids.shape[1] - 2
    if soft_struct_emb.shape[0] != residue_len:
        raise ValueError(
            f"Length mismatch for {name}: sequence has {residue_len} residues but soft embeddings have {soft_struct_emb.shape[0]} rows"
        )

    soft_struct_emb = soft_struct_emb.to(model_device)
    token_embedding_table = token_embedding_table.to(model_device)

    if soft_temperature is not None:
        if soft_temperature < 0:
            raise ValueError("soft_temperature must be >= 0")
        centroids = _load_centroids(
            structure_vocab_size=structure_vocab_size,
            soft_embedding_dim=soft_embedding_dim,
            model_device=model_device,
        )
        soft_norm = F.normalize(soft_struct_emb, p=2, dim=-1)
        centroid_norm = F.normalize(centroids, p=2, dim=-1)
        cosine_sim = torch.mm(soft_norm, centroid_norm.T)
        if soft_temperature == 0:
            hard_idx = torch.argmax(cosine_sim, dim=-1)
            projected = token_embedding_table[hard_idx]
        else:
            soft_weights = torch.softmax(cosine_sim / soft_temperature, dim=-1)
            projected = torch.mm(soft_weights, token_embedding_table)
    else:
        projection = _get_or_fit_projection(
            model=model,
            soft_embedding_dim=soft_embedding_dim,
            structure_vocab_size=structure_vocab_size,
        )
        projected = soft_struct_emb @ projection

    pad = torch.zeros(1, hidden_dim, device=model_device)
    struct_hidden = torch.cat([pad, projected, pad], dim=0).unsqueeze(0)

    activation_store = {}

    def pre_hook(module, args, kwargs):
        if kwargs is None:
            return None
        ss_ids = kwargs.get("ss_input_ids", None)
        if ss_ids is not None:
            activation_store["ss_emb"] = structure_embeddings(ss_ids)
        return None

    def embedding_hook(module, args, output):
        ss_emb_added = activation_store.get("ss_emb", None)
        if ss_emb_added is None:
            return output
        if isinstance(output, tuple):
            corrected = output[0] - ss_emb_added + struct_hidden
            return (corrected, *output[1:])
        return output - ss_emb_added + struct_hidden

    hook_pre = emb_module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    hook_post = emb_module.register_forward_hook(embedding_hook)
    try:
        dummy_ss = tokenize_structure_sequence([0] * residue_len).to(model_device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ss_input_ids=dummy_ss,
            labels=input_ids,
        )
    finally:
        hook_pre.remove()
        hook_post.remove()

    logits = outputs.logits
    logits = torch.log_softmax(logits[:, 1:-1, :], dim=-1)

    df = pd.read_csv(mutant_file)
    mutants = df["mutant"].tolist()
    vocab = tokenizer.get_vocab()
    scores = []
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
    print(f"{name}: {corr:.4f}")
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
        "--soft_embeddings_dir",
        type=str,
        required=False,
        default=None,
        help="Directory containing per-protein soft embeddings (*.pt)",
    )
    parser.add_argument(
        "--soft_embedding_dim",
        type=int,
        required=False,
        default=256,
        help="Dimension of each soft residue embedding",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=False,
        default=2048,
        help="Structure vocabulary size used to generate soft embeddings",
    )
    parser.add_argument(
        "--soft_temperature",
        type=float,
        required=False,
        default=None,
        help="Temperature used to build soft embeddings; enables token-embedding mixing path",
    )
    args = parser.parse_args()

    soft_embeddings_dir = args.soft_embeddings_dir or os.getenv("PROSST_SOFT_EMBEDDINGS_DIR")
    use_soft = soft_embeddings_dir is not None
    soft_temperature = args.soft_temperature
    if soft_temperature is None:
        env_temp = os.getenv("PROSST_SOFT_TEMPERATURE")
        if env_temp is not None:
            soft_temperature = float(env_temp)

    output_mutant_dir = Path(args.mutant_dir).parent / "substitutions_copy"
    output_mutant_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scored CSV outputs will be written to: {output_mutant_dir}")
    if use_soft:
        print(f"Using soft structure embeddings from: {soft_embeddings_dir}")
    else:
        print(f"Using hard structure sequence FASTA from: {args.structure_dir}")

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
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("Scoring proteins...")
    model_name = args.model_path.split("/")[-1]
    protein_names = read_names(args.residue_dir)
    print(protein_names)
    for protein_name in protein_names:
        if use_soft:
            score_protein_soft(
                model,
                tokenizer=tokenizer,
                residue_sequence_dir=args.residue_dir,
                soft_embeddings_dir=soft_embeddings_dir,
                mutant_dir=args.mutant_dir,
                output_mutant_dir=str(output_mutant_dir),
                model_name=model_name,
                name=protein_name,
                soft_embedding_dim=args.soft_embedding_dim,
                structure_vocab_size=args.vocab_size,
                soft_temperature=soft_temperature,
            )
        else:
            score_protein(
                model,
                tokenizer=tokenizer,
                residue_sequence_dir=args.residue_dir,
                structure_sequence_dir=args.structure_dir,
                mutant_dir=args.mutant_dir,
                output_mutant_dir=str(output_mutant_dir),
                model_name=model_name,
                name=protein_name,
            )


if __name__ == "__main__":
    main()

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from scipy.stats import spearmanr
from transformers import AutoModelForMaskedLM, AutoTokenizer

from prosst.structure.get_sst_seq import SSTPredictor, process_pdb_file
from prosst.structure.utils.data_utils import convert_graph
from zero_shot.proteingym_benchmark import (
    read_names,
    read_seq,
    score_protein_soft,
    tokenize_structure_sequence,
)


def _load_or_build_subgraphs(
    predictor: SSTPredictor,
    protein_name: str,
    pdb_file: Path,
    cache_subgraph_dir: Path,
    require_cached: bool = True,
):
    cache_subgraph_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_subgraph_dir / f"{protein_name}.pt"

    if not cache_file.exists():
        if require_cached:
            raise FileNotFoundError(
                f"Missing cached subgraph: {cache_file}. "
                "Set --allow_build_missing_cache to build it."
            )
        process_pdb_file(
            pdb_file=str(pdb_file),
            subgraph_depth=predictor.subgraph_depth,
            max_distance=predictor.max_distance,
            num_threads=predictor.num_threads,
            cache_subgraph_dir=str(cache_subgraph_dir),
        )

    subgraph_dict = torch.load(cache_file)
    graphs = [convert_graph(g) for g in subgraph_dict.values()]
    return graphs


def _compute_soft_embedding_for_protein(
    predictor: SSTPredictor,
    cluster_model,
    centroids: torch.Tensor,
    centroids_norm: torch.Tensor,
    graphs,
    temperature: float,
):
    batch_graphs = Batch.from_data_list(graphs)
    batch_graphs.node_s = torch.zeros_like(batch_graphs.node_s)
    batch_graphs = batch_graphs.to(predictor.device)

    with torch.no_grad():
        h_V = (batch_graphs.node_s, batch_graphs.node_v)
        h_E = (batch_graphs.edge_s, batch_graphs.edge_v)
        node_embeddings = predictor.model.get_embedding(h_V, batch_graphs.edge_index, h_E)
        graph_embeddings = scatter_mean(node_embeddings, batch_graphs.batch, dim=0)

        graph_embeddings_norm = F.normalize(graph_embeddings, p=2, dim=-1)
        cosine_sim = torch.mm(graph_embeddings_norm, centroids_norm.T)

        if temperature == 0:
            hard_idx_np = cluster_model.predict(graph_embeddings_norm.detach().cpu().numpy())
            hard_idx = torch.as_tensor(hard_idx_np, dtype=torch.long, device=predictor.device)
            return {
                "type": "hard_ids",
                "data": hard_idx.cpu(),
                "temperature": float(temperature),
            }

        soft_weights = torch.softmax(cosine_sim / temperature, dim=-1)
        return {
            "type": "token_weights",
            "data": soft_weights.cpu(),
            "temperature": float(temperature),
        }


def _compute_hard_ids_for_protein(
    predictor: SSTPredictor,
    centroids_norm: torch.Tensor,
    graphs,
):
    batch_graphs = Batch.from_data_list(graphs)
    batch_graphs.node_s = torch.zeros_like(batch_graphs.node_s)
    batch_graphs = batch_graphs.to(predictor.device)

    with torch.no_grad():
        h_V = (batch_graphs.node_s, batch_graphs.node_v)
        h_E = (batch_graphs.edge_s, batch_graphs.edge_v)
        node_embeddings = predictor.model.get_embedding(h_V, batch_graphs.edge_index, h_E)
        graph_embeddings = scatter_mean(node_embeddings, batch_graphs.batch, dim=0)
        graph_embeddings_norm = F.normalize(graph_embeddings, p=2, dim=-1)
        cosine_sim = torch.mm(graph_embeddings_norm, centroids_norm.T)
        hard_idx = torch.argmax(cosine_sim, dim=-1)

    return hard_idx.cpu()


@torch.no_grad()
def _score_protein_hard_ids(
    model,
    tokenizer,
    residue_sequence_dir: Path,
    mutant_dir: Path,
    output_mutant_dir: Path,
    name: str,
    model_name: str,
    structure_token_ids: torch.Tensor,
):
    print(f"Scoring {name} [hard tokens @ T=0]...")
    residue_fasta = residue_sequence_dir / f"{name}.fasta"
    mutant_file = mutant_dir / f"{name}.csv"
    output_mutant_file = output_mutant_dir / f"{name}.csv"

    sequence = read_seq(residue_fasta)
    ss_input_ids = tokenize_structure_sequence(structure_token_ids.tolist()).to(next(model.parameters()).device)

    tokenized_results = tokenizer([sequence], return_tensors="pt")
    input_ids = tokenized_results["input_ids"].to(next(model.parameters()).device)
    attention_mask = tokenized_results["attention_mask"].to(next(model.parameters()).device)

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


def run_temperature(
    temperature: float,
    model,
    tokenizer,
    predictor: SSTPredictor,
    cluster_model,
    centroids: torch.Tensor,
    centroids_norm: torch.Tensor,
    residue_dir: Path,
    pdb_dir: Path,
    mutant_dir: Path,
    output_mutant_dir: Path,
    temp_soft_dir: Path,
    cache_subgraph_dir: Path,
    model_name: str,
    avg_script_path: Path,
    avg_log_path: Path,
    require_cached: bool,
):
    if output_mutant_dir.exists():
        shutil.rmtree(output_mutant_dir)
    output_mutant_dir.mkdir(parents=True, exist_ok=True)

    if temp_soft_dir.exists():
        shutil.rmtree(temp_soft_dir)
    temp_soft_dir.mkdir(parents=True, exist_ok=True)

    protein_names = read_names(str(residue_dir))

    print(f"\n=== Temperature {temperature} ===")
    for i, protein_name in enumerate(protein_names, start=1):
        print(f"\n[{i}/{len(protein_names)}] {protein_name} :: start", flush=True)
        pdb_file = pdb_dir / f"{protein_name}.pdb"
        if not pdb_file.exists():
            raise FileNotFoundError(f"Missing PDB file: {pdb_file}")

        print(f"[{i}/{len(protein_names)}] loading cached subgraphs", flush=True)
        graphs = _load_or_build_subgraphs(
            predictor=predictor,
            protein_name=protein_name,
            pdb_file=pdb_file,
            cache_subgraph_dir=cache_subgraph_dir,
            require_cached=require_cached,
        )
        print(f"[{i}/{len(protein_names)}] loaded {len(graphs)} subgraphs", flush=True)

        if temperature == 0:
            print(f"[{i}/{len(protein_names)}] running GVP + hard tokenization", flush=True)
            hard_ids = _compute_hard_ids_for_protein(
                predictor=predictor,
                centroids_norm=centroids_norm,
                graphs=graphs,
            )
            print(
                f"[{i}/{len(protein_names)}] hard token length {hard_ids.shape[0]}",
                flush=True,
            )

            print(f"[{i}/{len(protein_names)}] scoring", flush=True)
            _score_protein_hard_ids(
                model=model,
                tokenizer=tokenizer,
                residue_sequence_dir=residue_dir,
                mutant_dir=mutant_dir,
                output_mutant_dir=output_mutant_dir,
                name=protein_name,
                model_name=model_name,
                structure_token_ids=hard_ids,
            )
            del hard_ids
        else:
            print(f"[{i}/{len(protein_names)}] running GVP + soft encoding", flush=True)
            soft_payload = _compute_soft_embedding_for_protein(
                predictor=predictor,
                cluster_model=cluster_model,
                centroids=centroids,
                centroids_norm=centroids_norm,
                graphs=graphs,
                temperature=temperature,
            )
            payload_data = soft_payload["data"]
            print(
                f"[{i}/{len(protein_names)}] soft payload {soft_payload['type']} shape {tuple(payload_data.shape)}",
                flush=True,
            )

            temp_soft_file = temp_soft_dir / f"{protein_name}.pt"
            torch.save(soft_payload, temp_soft_file)
            print(f"[{i}/{len(protein_names)}] wrote temp embedding {temp_soft_file.name}", flush=True)

            print(f"[{i}/{len(protein_names)}] scoring", flush=True)
            score_protein_soft(
                model=model,
                tokenizer=tokenizer,
                residue_sequence_dir=str(residue_dir),
                soft_embeddings_dir=str(temp_soft_dir),
                mutant_dir=str(mutant_dir),
                output_mutant_dir=str(output_mutant_dir),
                name=protein_name,
                model_name=model_name,
                structure_vocab_size=centroids.shape[0],
                soft_temperature=temperature,
            )

            temp_soft_file.unlink(missing_ok=True)
            print(f"[{i}/{len(protein_names)}] deleted temp embedding", flush=True)
        print(f"[{i}/{len(protein_names)}] done {protein_name}", flush=True)

        if torch.cuda.is_available() and str(predictor.device).startswith("cuda"):
            torch.cuda.empty_cache()

    with avg_log_path.open("w") as handle:
        subprocess.run(
            [
                "python3",
                str(avg_script_path),
                "--data_dir",
                str(output_mutant_dir),
            ],
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )

    shutil.rmtree(temp_soft_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="AI4Protein/ProSST-2048")
    parser.add_argument("--residue_dir", type=str, default="zero_shot/example_data/residue_sequence")
    parser.add_argument("--mutant_dir", type=str, default="zero_shot/example_data/substitutions")
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument("--cache_subgraph_dir", type=str, default="zero_shot/cache_subgraphs")
    parser.add_argument("--temps", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1, 0.3, 1.0])
    parser.add_argument("--vocab_size", type=int, default=2048)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--output_root", type=str, default="zero_shot/stream_results")
    parser.add_argument("--allow_build_missing_cache", action="store_true")
    args = parser.parse_args()

    residue_dir = Path(args.residue_dir)
    mutant_dir = Path(args.mutant_dir)
    pdb_dir = Path(args.pdb_dir)
    cache_subgraph_dir = Path(args.cache_subgraph_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    avg_script_path = Path(__file__).resolve().parent / "average_spearman.py"

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script. Use --device cuda on a CUDA-enabled node.")

    print("Loading language model on CUDA...")
    model = AutoModelForMaskedLM.from_pretrained(args.model_path, trust_remote_code=True)
    model.cls.predictions.decoder.weight = model.prosst.embeddings.word_embeddings.weight
    assert model.cls.predictions.decoder.weight.data_ptr() == model.prosst.embeddings.word_embeddings.weight.data_ptr()
    print("Decoder/embedding weights manually tied ✓")
    model = model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model_name = args.model_path.split("/")[-1]

    print("Loading structure encoder on CUDA...")
    predictor = SSTPredictor(
        structure_vocab_size=args.vocab_size,
        device=args.device,
        max_batch_nodes=10000,
        num_processes=args.num_processes,
        num_threads=args.num_threads,
    )

    cluster_model_path = Path(predictor.cluster_dir) / f"{args.vocab_size}.joblib"
    cluster_model = joblib.load(cluster_model_path)
    centroids = torch.tensor(
        cluster_model.cluster_centers_,
        dtype=torch.float32,
        device=args.device,
    )
    centroids_norm = F.normalize(centroids, p=2, dim=-1)

    for temperature in args.temps:
        temp_tag = str(temperature).replace(".", "p")
        output_mutant_dir = output_root / f"substitutions_T_{temp_tag}"
        temp_soft_dir = output_root / f"tmp_soft_T_{temp_tag}"
        avg_log_path = output_root / f"T{temperature}.log"

        run_temperature(
            temperature=temperature,
            model=model,
            tokenizer=tokenizer,
            predictor=predictor,
            cluster_model=cluster_model,
            centroids=centroids,
            centroids_norm=centroids_norm,
            residue_dir=residue_dir,
            pdb_dir=pdb_dir,
            mutant_dir=mutant_dir,
            output_mutant_dir=output_mutant_dir,
            temp_soft_dir=temp_soft_dir,
            cache_subgraph_dir=cache_subgraph_dir,
            model_name=model_name,
            avg_script_path=avg_script_path,
            avg_log_path=avg_log_path,
            require_cached=not args.allow_build_missing_cache,
        )

    print(f"Done. Logs and scored CSVs are in: {output_root}")


if __name__ == "__main__":
    main()

import argparse
import os
from pathlib import Path

import joblib
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
from torch_scatter import scatter_mean

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except Exception:
    FSDP = None

from prosst.structure.get_sst_seq import SSTPredictor, pdb_conventer, subgraph_conventer


def _model_get_embedding(model, h_V, edge_index, h_E):
    if hasattr(model, "get_embedding"):
        return model.get_embedding(h_V, edge_index, h_E)
    if hasattr(model, "module") and hasattr(model.module, "get_embedding"):
        return model.module.get_embedding(h_V, edge_index, h_E)
    raise AttributeError("Model does not expose get_embedding")


def _init_distributed(distributed: bool):
    if not distributed:
        return {"enabled": False, "rank": 0, "world_size": 1, "local_rank": 0, "initialized_here": False}

    initialized_here = False
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        initialized_here = True

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return {
        "enabled": True,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "initialized_here": initialized_here,
    }


def _finalize_distributed(dist_ctx):
    if dist_ctx.get("enabled") and dist_ctx.get("initialized_here") and dist.is_initialized():
        dist.destroy_process_group()


def _prepare_data_loader(
    predictor: SSTPredictor,
    pdb_dir: str,
    cache_subgraph_dir: str = None,
    pdb_files_override=None,
):
    if pdb_files_override is not None:
        pdb_files = [str(p) for p in pdb_files_override]
    else:
        pdb_files = sorted(Path(pdb_dir).glob("*.pdb"))
        pdb_files = [str(p) for p in pdb_files]

    if not pdb_files:
        raise ValueError(f"No .pdb files found in {pdb_dir}")

    cache_path = Path(cache_subgraph_dir) if cache_subgraph_dir else None
    use_cached_subgraphs = False
    if cache_path is not None and cache_path.exists():
        cached_files = list(cache_path.glob("*.pt"))
        use_cached_subgraphs = len(cached_files) == len(pdb_files)

    if use_cached_subgraphs:
        print(f"Loading cached subgraphs from {cache_path}...")
        data_loader, results = subgraph_conventer(
            subgraph_dir=str(cache_path),
            pdb_dir=pdb_dir,
            max_batch_nodes=predictor.max_batch_nodes,
            num_processes=predictor.num_processes,
        )
        return data_loader, results

    print(f"Building subgraphs for {len(pdb_files)} PDBs...")
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)

    data_loader, results = pdb_conventer(
        pdb_files,
        subgraph_depth=predictor.subgraph_depth,
        max_distance=predictor.max_distance,
        max_batch_nodes=predictor.max_batch_nodes,
        error_file=None,
        num_processes=predictor.num_processes,
        num_threads=predictor.num_threads,
        cache_subgraph_dir=str(cache_path) if cache_path is not None else None,
    )

    return data_loader, results


def _collect_graph_embeddings(model, dataloader, device):
    all_graph_embeddings = []
    print("Running GVP encoder once to collect per-residue graph embeddings...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch.to(device)
            h_V = (batch.node_s, batch.node_v)
            h_E = (batch.edge_s, batch.edge_v)
            node_embeddings = _model_get_embedding(model, h_V, batch.edge_index, h_E)
            graph_embeddings = scatter_mean(node_embeddings, batch.batch, dim=0)
            all_graph_embeddings.append(graph_embeddings.cpu())
    return torch.cat(all_graph_embeddings, dim=0)


def _soft_from_node_embeddings(
    node_embeddings_cpu: torch.Tensor,
    centroids: torch.Tensor,
    cluster_model,
    temperature: float,
    device: str,
    chunk_size: int = 4096,
):
    if temperature < 0:
        raise ValueError("temperature must be >= 0")

    centroids_norm = F.normalize(centroids, p=2, dim=-1)
    total = node_embeddings_cpu.shape[0]
    chunks = []

    with torch.no_grad():
        for start in tqdm(
            range(0, total, chunk_size),
            desc="Soft assignment chunks",
            total=(total + chunk_size - 1) // chunk_size,
        ):
            end = min(start + chunk_size, total)
            node_chunk = node_embeddings_cpu[start:end].to(device)
            if temperature == 0:
                node_chunk_norm = F.normalize(node_chunk, p=2, dim=-1)
                hard_idx_np = cluster_model.predict(node_chunk_norm.detach().cpu().numpy())
                hard_idx = torch.as_tensor(hard_idx_np, dtype=torch.long, device=device)
                soft_embeddings = centroids[hard_idx]
            else:
                node_chunk_norm = F.normalize(node_chunk, p=2, dim=-1)
                cosine_sim = torch.mm(node_chunk_norm, centroids_norm.T)
                soft_weights = torch.softmax(cosine_sim / temperature, dim=-1)
                soft_embeddings = torch.mm(soft_weights, centroids)
            chunks.append(soft_embeddings.cpu())

    return torch.cat(chunks, dim=0)


def _save_soft_embeddings(all_soft_embeddings, results, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    cursor = 0
    total_proteins = len(results)
    for index, result in enumerate(
        tqdm(results, desc="Saving protein embeddings", total=total_proteins), start=1
    ):
        name = Path(result["name"]).stem
        length = len(result["aa_seq"])
        protein_emb = all_soft_embeddings[cursor : cursor + length]
        cursor += length
        out_path = Path(output_dir) / f"{name}.pt"
        torch.save(protein_emb, out_path)
        print(
            f"[{index}/{total_proteins}] Saved {name}: shape {tuple(protein_emb.shape)}",
            flush=True,
        )


def precompute_soft_embeddings(
    pdb_dir: str,
    output_dir: str,
    temperature: float = 0.1,
    structure_vocab_size: int = 2048,
    device: str = "cuda",
    num_processes: int = 12,
    num_threads: int = 16,
    max_batch_nodes: int = 10000,
    cache_subgraph_dir: str = None,
    chunk_size: int = 4096,
):
    if temperature < 0:
        raise ValueError("temperature must be >= 0")

    precompute_soft_embeddings_for_temperatures(
        pdb_dir=pdb_dir,
        output_root=str(Path(output_dir).parent),
        temperatures=[temperature],
        structure_vocab_size=structure_vocab_size,
        device=device,
        num_processes=num_processes,
        num_threads=num_threads,
        max_batch_nodes=max_batch_nodes,
        cache_subgraph_dir=cache_subgraph_dir,
        chunk_size=chunk_size,
        output_name_override={temperature: Path(output_dir).name},
    )


def precompute_soft_embeddings_for_temperatures(
    pdb_dir: str,
    output_root: str,
    temperatures,
    structure_vocab_size: int = 2048,
    device: str = "cuda",
    num_processes: int = 12,
    num_threads: int = 16,
    max_batch_nodes: int = 10000,
    cache_subgraph_dir: str = None,
    chunk_size: int = 4096,
    output_name_override=None,
    distributed: bool = False,
    use_fsdp: bool = False,
    shard_by_rank: bool = False,
):
    output_name_override = output_name_override or {}

    dist_ctx = _init_distributed(distributed)
    rank = dist_ctx["rank"]
    world_size = dist_ctx["world_size"]

    if torch.cuda.is_available() and device.startswith("cuda"):
        if distributed:
            device = f"cuda:{dist_ctx['local_rank']}"
            torch.cuda.set_device(dist_ctx["local_rank"])

    all_pdb_files = sorted(Path(pdb_dir).glob("*.pdb"))
    if shard_by_rank and world_size > 1:
        local_pdb_files = [p for i, p in enumerate(all_pdb_files) if i % world_size == rank]
    else:
        local_pdb_files = all_pdb_files

    if rank == 0:
        print(f"Distributed enabled={distributed}, world_size={world_size}, shard_by_rank={shard_by_rank}")
    print(f"Rank {rank}: assigned {len(local_pdb_files)} PDB files")

    predictor = SSTPredictor(
        structure_vocab_size=structure_vocab_size,
        device=device,
        max_batch_nodes=max_batch_nodes,
        num_processes=num_processes,
        num_threads=num_threads,
    )

    if use_fsdp:
        if FSDP is None:
            raise RuntimeError("FSDP is not available in this PyTorch installation")
        fsdp_group = None
        if distributed and shard_by_rank:
            fsdp_group = dist.new_group([rank])
        predictor.model = FSDP(
            predictor.model,
            process_group=fsdp_group,
            device_id=torch.device(device) if str(device).startswith("cuda") else None,
            use_orig_params=True,
        )

    cluster_model_path = str(Path(predictor.cluster_dir) / f"{structure_vocab_size}.joblib")
    cluster_model = joblib.load(cluster_model_path)
    centroids = torch.tensor(cluster_model.cluster_centers_, dtype=torch.float32, device=device)

    data_loader, results = _prepare_data_loader(
        predictor=predictor,
        pdb_dir=pdb_dir,
        cache_subgraph_dir=cache_subgraph_dir,
        pdb_files_override=local_pdb_files,
    )

    if len(results) == 0:
        print(f"Rank {rank}: no proteins assigned, skipping.")
        _finalize_distributed(dist_ctx)
        return

    graph_embeddings = _collect_graph_embeddings(
        model=predictor.model,
        dataloader=data_loader,
        device=device,
    )
    print(f"Collected graph embeddings: {tuple(graph_embeddings.shape)}")

    for temperature in temperatures:
        print(f"Computing soft embeddings for T={temperature}...")
        all_soft_embeddings = _soft_from_node_embeddings(
            node_embeddings_cpu=graph_embeddings,
            centroids=centroids,
            cluster_model=cluster_model,
            temperature=temperature,
            device=device,
            chunk_size=chunk_size,
        )

        output_name = output_name_override.get(
            temperature, f"T_{str(temperature).replace('.', 'p')}"
        )
        output_dir = str(Path(output_root) / output_name)
        _save_soft_embeddings(
            all_soft_embeddings=all_soft_embeddings,
            results=results,
            output_dir=output_dir,
        )
        print(f"Done. Soft embeddings saved to {output_dir}")

    if dist_ctx["enabled"]:
        dist.barrier()
    _finalize_distributed(dist_ctx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--output_dir", required=False, default=None)
    parser.add_argument("--output_root", required=False, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--temperatures", type=float, nargs="+", default=None)
    parser.add_argument("--vocab_size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_processes", type=int, default=12)
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--max_batch_nodes", type=int, default=10000)
    parser.add_argument("--cache_subgraph_dir", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--use_fsdp", action="store_true")
    parser.add_argument("--shard_by_rank", action="store_true")
    args = parser.parse_args()

    if args.temperatures is not None:
        output_root = args.output_root
        if output_root is None:
            if args.output_dir is None:
                raise ValueError("Provide --output_root (or --output_dir) when using --temperatures")
            output_root = str(Path(args.output_dir).parent)

        precompute_soft_embeddings_for_temperatures(
            pdb_dir=args.pdb_dir,
            output_root=output_root,
            temperatures=args.temperatures,
            structure_vocab_size=args.vocab_size,
            device=args.device,
            num_processes=args.num_processes,
            num_threads=args.num_threads,
            max_batch_nodes=args.max_batch_nodes,
            cache_subgraph_dir=args.cache_subgraph_dir,
            chunk_size=args.chunk_size,
            distributed=args.distributed,
            use_fsdp=args.use_fsdp,
            shard_by_rank=args.shard_by_rank,
        )
    else:
        if args.output_dir is None:
            raise ValueError("--output_dir is required when --temperatures is not provided")

        precompute_soft_embeddings_for_temperatures(
            pdb_dir=args.pdb_dir,
            output_root=str(Path(args.output_dir).parent),
            temperatures=[args.temperature],
            structure_vocab_size=args.vocab_size,
            device=args.device,
            num_processes=args.num_processes,
            num_threads=args.num_threads,
            max_batch_nodes=args.max_batch_nodes,
            cache_subgraph_dir=args.cache_subgraph_dir,
            chunk_size=args.chunk_size,
            output_name_override={args.temperature: Path(args.output_dir).name},
            distributed=args.distributed,
            use_fsdp=args.use_fsdp,
            shard_by_rank=args.shard_by_rank,
        )

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import torch

from zero_shot.precompute_soft_embeddings import precompute_soft_embeddings_for_temperatures


def format_temp_name(temp: float) -> str:
    return str(temp).replace(".", "p")


def run_score_script(repo_root: Path, log_file: Path, soft_embeddings_dir: Path, temperature: float):
    env = os.environ.copy()
    env["PROSST_SOFT_EMBEDDINGS_DIR"] = str(soft_embeddings_dir)
    env["PROSST_SOFT_TEMPERATURE"] = str(temperature)

    subprocess.run(
        ["bash", "score.sh", str(log_file)],
        cwd=str(repo_root),
        env=env,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdb_dir",
        type=str,
        required=True,
        help="Directory containing ProteinGym PDB files",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.3, 1.0],
    )
    parser.add_argument(
        "--soft_output_root",
        type=str,
        default="zero_shot/soft_embeddings",
        help="Root directory for temperature-specific soft embeddings",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="zero_shot/temp_logs",
        help="Directory for per-temperature score logs",
    )
    parser.add_argument("--vocab_size", type=int, default=2048)
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--use_fsdp", action="store_true")
    parser.add_argument("--distributed_precompute", action="store_true")
    parser.add_argument("--num_processes", type=int, default=12)
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--max_batch_nodes", type=int, default=10000)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument(
        "--cache_subgraph_dir",
        type=str,
        default="zero_shot/cache_subgraphs",
        help="Cache path for generated subgraphs to accelerate reruns",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    soft_output_root = repo_root / args.soft_output_root
    logs_dir = repo_root / args.logs_dir
    substitutions_copy_dir = repo_root / "zero_shot" / "example_data" / "substitutions_copy"
    cache_subgraph_dir = repo_root / args.cache_subgraph_dir if args.cache_subgraph_dir else None

    soft_output_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    for temp in args.temperatures:
        temp_tag = format_temp_name(temp)
        soft_dir = soft_output_root / f"T_{temp_tag}"
        if soft_dir.exists():
            shutil.rmtree(soft_dir)
            print(f"Deleted previous soft embedding dir: {soft_dir}")

    if args.distributed_precompute and args.nproc_per_node > 1:
        cmd = [
            "torchrun",
            "--nproc_per_node",
            str(args.nproc_per_node),
            "-m",
            "zero_shot.precompute_soft_embeddings",
            "--pdb_dir",
            args.pdb_dir,
            "--output_root",
            str(soft_output_root),
            "--temperatures",
            *[str(t) for t in args.temperatures],
            "--vocab_size",
            str(args.vocab_size),
            "--device",
            args.device,
            "--num_processes",
            str(args.num_processes),
            "--num_threads",
            str(args.num_threads),
            "--max_batch_nodes",
            str(args.max_batch_nodes),
            "--chunk_size",
            str(args.chunk_size),
            "--distributed",
            "--shard_by_rank",
        ]
        if cache_subgraph_dir:
            cmd.extend(["--cache_subgraph_dir", str(cache_subgraph_dir)])
        if args.use_fsdp:
            cmd.append("--use_fsdp")

        print("Running distributed precompute:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(repo_root), check=True)
    else:
        output_name_override = {
            temp: f"T_{format_temp_name(temp)}" for temp in args.temperatures
        }
        precompute_soft_embeddings_for_temperatures(
            pdb_dir=args.pdb_dir,
            output_root=str(soft_output_root),
            temperatures=args.temperatures,
            structure_vocab_size=args.vocab_size,
            device=args.device,
            num_processes=args.num_processes,
            num_threads=args.num_threads,
            max_batch_nodes=args.max_batch_nodes,
            cache_subgraph_dir=str(cache_subgraph_dir) if cache_subgraph_dir else None,
            chunk_size=args.chunk_size,
            output_name_override=output_name_override,
        )

    for temp in args.temperatures:
        temp_tag = format_temp_name(temp)
        soft_dir = soft_output_root / f"T_{temp_tag}"
        log_file = logs_dir / f"temp_{temp_tag}.log"

        if substitutions_copy_dir.exists():
            shutil.rmtree(substitutions_copy_dir)
            print(f"Deleted previous substitutions_copy: {substitutions_copy_dir}")

        print(f"Running score.sh for temperature={temp} -> {log_file}")
        run_score_script(
            repo_root=repo_root,
            log_file=log_file,
            soft_embeddings_dir=soft_dir,
            temperature=temp,
        )

    print("Temperature sweep completed.")


if __name__ == "__main__":
    main()

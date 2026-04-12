"""
Fetch MSAs from ColabFold and cache _cons.npy for all proteins.

Usage:
    python precompute_conservation.py \
        --residue_dir zero_shot/example_data/residue_sequence \
        --msa_dir     zero_shot/msa_files \
        --temp_dir    temp \
    --delay       45
"""
import argparse
import os
import sys
from pathlib import Path

from Bio import SeqIO
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zero_shot.conservation_utils import get_conservation


def read_seq(path):
    for record in SeqIO.parse(path, "fasta"):
        return str(record.seq)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residue_dir", default="zero_shot/example_data/residue_sequence")
    parser.add_argument("--msa_dir", default="zero_shot/msa_files")
    parser.add_argument("--temp_dir", default="temp")
    parser.add_argument("--delay", type=float, default=45.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.msa_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    fastas = sorted(Path(args.residue_dir).glob("*.fasta"))
    failed = []

    for fasta in tqdm(fastas, desc="MSA + conservation"):
        name = fasta.stem
        try:
            sequence = read_seq(str(fasta))
            conservation = get_conservation(
                name,
                sequence,
                args.msa_dir,
                temp_msa_dir=args.temp_dir,
                force_recompute=args.force,
                post_delay=args.delay,
            )
            tqdm.write(f"  {name}: L={len(conservation)}, mean={conservation.mean():.3f}")
        except Exception as error:
            tqdm.write(f"[FAIL] {name}: {error}")
            failed.append(name)

    if failed:
        output = os.path.join(args.msa_dir, "failed.txt")
        Path(output).write_text("\n".join(failed))
        print(f"\n{len(failed)} failed → {output}")
    else:
        print("\nAll done.")


if __name__ == "__main__":
    main()
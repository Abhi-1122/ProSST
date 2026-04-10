from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

IGNORED_SCORE_COLUMNS = {"mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"}


def discover_score_column(df, explicit_column=None):
    if explicit_column:
        if explicit_column not in df.columns:
            raise ValueError(f"Requested score column '{explicit_column}' missing")
        return explicit_column
    candidates = [c for c in df.columns if c not in IGNORED_SCORE_COLUMNS]
    if not candidates:
        raise ValueError("No score column found")
    return candidates[-1]


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="zero_shot/example_data/substitutions_copy")
    parser.add_argument("--reference_file", type=str, default="zero_shot/example_data/DMS_substitutions.csv",
                        help="Path to DMS substitutions reference CSV")
    parser.add_argument("--score_column", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ref = pd.read_csv(args.reference_file)
    # ref must have: DMS_id, UniProt_ID, DMS_filename, coarse_selection_type

    records = []
    skipped = 0

    for _, row in ref.iterrows():
        dms_id    = row["DMS_id"]
        uniprot   = row["UniProt_ID"]
        sel_type  = row["coarse_selection_type"]
        csv_path  = data_dir / dms_id / f"{dms_id}.csv"   # adjust glob if needed

        # Try to find the file — your CSVs may just be named by DMS_id
        if not csv_path.exists():
            # fallback: search for any csv containing dms_id
            matches = list(data_dir.glob(f"**/{dms_id}.csv"))
            if not matches:
                print(f"Skipping {dms_id}: file not found")
                skipped += 1
                continue
            csv_path = matches[0]

        try:
            df = pd.read_csv(csv_path)
            col = discover_score_column(df, args.score_column)
            rho = spearmanr(df["DMS_score"], df[col], nan_policy="omit").correlation
            if pd.isna(rho):
                raise ValueError("NaN Spearman")
            records.append({"DMS_id": dms_id, "UniProt_ID": uniprot,
                             "coarse_selection_type": sel_type, "spearman": rho})
            print(f"{dms_id}: {rho:.4f}")
        except Exception as e:
            print(f"Skipping {dms_id}: {e}")
            skipped += 1

    if not records:
        raise RuntimeError("No valid correlations computed")

    df_results = pd.DataFrame(records)

    # ── Level 1: already have per-DMS spearman ──────────────────────────────

    # ── Level 2: average per UniProt_ID ─────────────────────────────────────
    uniprot_avg = (df_results
                   .groupby(["UniProt_ID", "coarse_selection_type"])["spearman"]
                   .mean()
                   .reset_index())

    # ── Level 3: average per functional category, then average across cats ──
    cat_avg = (uniprot_avg
               .groupby("coarse_selection_type")["spearman"]
               .mean())

    final_score = cat_avg.mean()

    print(f"\nFiles processed: {len(records)}")
    print(f"Files skipped:   {skipped}")
    print(f"\nPer-category averages:")
    for cat, val in cat_avg.items():
        print(f"  {cat:25s}: {val:.4f}")
    print(f"\nFinal Spearman (official weighting): {final_score:.6f}")
    print(f"Simple mean (your old method):       "
          f"{df_results['spearman'].mean():.6f}")


if __name__ == "__main__":
    main()
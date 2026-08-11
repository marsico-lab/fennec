import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .helm_converter import create_helm_notation
from .preprocessing import prepare_meta_features_and_one_hot


DEFAULT_CHUNKSIZE = 2000
REQUIRED_SEQUENCE_COLUMNS = ["siRNA_list", "siRNA_passenger_list"]


def default_output_path(input_path: Path) -> Path:
    suffix = "".join(input_path.suffixes)
    stem = input_path.name[:-len(suffix)] if suffix else input_path.name
    return input_path.with_name(f"{stem}_helm_validated.csv.gz")


def default_summary_path(output_path: Path) -> Path:
    if output_path.suffix == ".json":
        return output_path
    suffix = "".join(output_path.suffixes)
    stem = output_path.name[:-len(suffix)] if suffix else output_path.name
    return output_path.with_name(f"{stem}_summary.json")


def _ensure_required_columns(df: pd.DataFrame):
    missing = [column for column in REQUIRED_SEQUENCE_COLUMNS if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")


def _ensure_target_column(df: pd.DataFrame) -> pd.DataFrame:
    if "mRNA_remaining_pct" in df.columns:
        return df

    df = df.copy()
    df["mRNA_remaining_pct"] = 0.0
    return df


def _build_helm_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    helm_df = df.drop(columns=["siRNA_list", "siRNA_passenger_list"]).copy()
    return helm_df


def validate_helm_one_hot_csv(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    summary_path: Optional[str | Path] = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
    write_output: bool = True,
) -> Dict[str, Any]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    if output_path is None:
        output_path = default_output_path(input_path)
    output_path = Path(output_path)

    if summary_path is None:
        summary_path = default_summary_path(output_path)
    summary_path = Path(summary_path)

    if write_output and output_path.exists():
        output_path.unlink()

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path) if write_output else None,
        "rows_processed": 0,
        "chunks_processed": 0,
        "helm_creation_errors": 0,
        "matrix_mismatch_rows": 0,
        "matrix_mismatch_positions": 0,
        "first_mismatch_rows": [],
        "all_equal": True,
    }

    for chunk_idx, chunk in enumerate(pd.read_csv(input_path, compression="infer", chunksize=chunksize)):
        _ensure_required_columns(chunk)
        chunk = _ensure_target_column(chunk)

        helms = []
        for row_idx, guide, passenger in zip(
            chunk.index.tolist(),
            chunk["siRNA_list"],
            chunk["siRNA_passenger_list"],
        ):
            try:
                helms.append(create_helm_notation(str(guide), str(passenger)))
            except Exception:
                helms.append(None)
                summary["helm_creation_errors"] += 1
                if len(summary["first_mismatch_rows"]) < 10:
                    summary["first_mismatch_rows"].append(int(row_idx))
                summary["all_equal"] = False

        chunk = chunk.copy()
        chunk["HELM"] = helms

        if write_output:
            chunk.to_csv(
                output_path,
                index=False,
                compression="gzip",
                mode="w" if chunk_idx == 0 else "a",
                header=(chunk_idx == 0),
            )

        valid_chunk = chunk[chunk["HELM"].notna()].copy()
        if not valid_chunk.empty:
            orig_df = valid_chunk.copy()
            helm_df = _build_helm_dataframe(valid_chunk)

            X_orig, _, _ = prepare_meta_features_and_one_hot(orig_df)
            X_helm, _, _ = prepare_meta_features_and_one_hot(helm_df)

            neq = X_orig != X_helm
            diff_positions = int(neq.sum().item())
            if diff_positions:
                row_mask = neq.any(dim=2).any(dim=1)
                mismatch_rows = valid_chunk.index[row_mask.cpu().numpy()].tolist()
                summary["matrix_mismatch_rows"] += len(mismatch_rows)
                summary["matrix_mismatch_positions"] += diff_positions
                if len(summary["first_mismatch_rows"]) < 10:
                    needed = 10 - len(summary["first_mismatch_rows"])
                    summary["first_mismatch_rows"].extend(int(x) for x in mismatch_rows[:needed])
                summary["all_equal"] = False

        summary["rows_processed"] += len(chunk)
        summary["chunks_processed"] += 1
        print(
            f"chunk={chunk_idx} rows={summary['rows_processed']} "
            f"all_equal={summary['all_equal']} mismatches={summary['matrix_mismatch_rows']}"
        )

    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"summary_path {summary_path}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create HELM strings from siRNA columns, write a HELM-augmented CSV, and validate one-hot equivalence.",
    )
    parser.add_argument("input", help="Input CSV path. Compression is inferred from the file extension.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV path for the HELM-augmented copy. Defaults to <input>_helm_validated.csv.gz",
    )
    parser.add_argument(
        "--summary",
        help="Output JSON path for the validation summary. Defaults to <output>_summary.json",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNKSIZE,
        help=f"Rows to process per chunk (default: {DEFAULT_CHUNKSIZE})",
    )
    parser.add_argument(
        "--no-write-output",
        action="store_true",
        help="Run the validation without writing the HELM-augmented CSV copy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = validate_helm_one_hot_csv(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary,
        chunksize=args.chunksize,
        write_output=not args.no_write_output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
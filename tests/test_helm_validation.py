import json

import pandas as pd

from fennec.helm_validation import validate_helm_one_hot_csv


def test_validate_helm_one_hot_csv_round_trip(tmp_path):
    input_path = tmp_path / "mini.csv.gz"
    output_path = tmp_path / "mini_helm.csv.gz"
    summary_path = tmp_path / "mini_summary.json"

    df = pd.DataFrame(
        {
            "gene_target_species": ["Homo sapiens", "Homo sapiens"],
            "cell_line_donor": ["HeLa", "HeLa"],
            "siRNA_concentration": [10.0, 10.0],
            "Transfection_method": ["RNAiMAX", "RNAiMAX"],
            "Duration_after_transfection_h": [24.0, 24.0],
            "mRNA_remaining_pct": [50.0, 60.0],
            "siRNA_list": [
                "VP U a a c dA c (C2p) u c u u a c Uf g Af u u a c s c s c",
                "u A G U G U U c A G A G G A A C U U C dT s dT",
            ],
            "siRNA_passenger_list": [
                "a s u s u u g g a g a Af Uf Gf a Cf u a a a s u s g",
                "dT s dT A u c A C A A G u c c U u G A A G",
            ],
        }
    )
    df.to_csv(input_path, index=False, compression="gzip")

    summary = validate_helm_one_hot_csv(
        input_path=input_path,
        output_path=output_path,
        summary_path=summary_path,
        chunksize=1,
    )

    assert summary["all_equal"] is True
    assert summary["helm_creation_errors"] == 0
    assert summary["matrix_mismatch_rows"] == 0
    assert summary["matrix_mismatch_positions"] == 0

    output_df = pd.read_csv(output_path, compression="gzip")
    assert "HELM" in output_df.columns
    assert output_df.loc[0, "HELM"].startswith("RNA1{[VP].R(U)P")
    assert output_df.loc[0, "HELM"].endswith("$$$$V2.0")

    written_summary = json.loads(summary_path.read_text())
    assert written_summary["all_equal"] is True
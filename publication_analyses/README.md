# publication_analyses/ — scripts for the paper's figures and tables

This directory holds standalone plotting scripts that produce the paper's figures and tables. Each script uses argparse for input data paths and depends only on standard scientific Python libraries (pandas, matplotlib, seaborn, numpy, scipy, sklearn).

## Subdir → figure/table map

| Subdir | Figures/Tables produced | Script |
|---|---|---|
| `patent_stats/` | Fig 2 (dataset overview); SFig 1-3 (embedding correlations, MI, PCA) | `plot_dataset_overview.py`, `plot_embedding_correlations.py` |
| `investigate_chemical_diversity_for_design/` | Fig 2 chemical-diversity panel | `plot_chemical_diversity.py` |
| `ablation/` | Fig 3 model performance + ablation | `plot_ablation_fig3.py` |
| `ACVR1C_val/` | Fig 4 (ACVR1C data-efficiency) + SFig 9 | `plot_data_efficiency_fig4.py`, `plot_acvr1c_sfig9.py` |
| `roche_val/` | Fig 4 (JAK1 data-efficiency) + zero-shot JAK1 table | `plot_data_efficiency_fig4.py`, `table_zeroshot_jak1.py` |
| `deep_feature_importance/` | Fig 5 IG attribution maps | `compute_integrated_gradients.py` |
| `chem_mod_walk_exp/` | Fig 6 greedy beam modification scan | `plot_mod_enhancer_final_graph.py`, `plot_mod_enhancer_final_graph_helm.py` |
| `OligoFormer/` | SFig 5-7 benchmark panels | `plot_oligoformer_efficacy.py`, `plot_benchmark_comparison.py`, `plot_inter_dataset_validation.py` |
| `APP_Val/` | Zero-shot APP table | `table_zeroshot_app.py` |

## Known gaps

1. **Patent OCR pipeline** (Fig 2 upstream data) is not shipped.
2. **AHSA1 raw wetlab data** (Fig 7) is released separately by the wetlab collaborators — see the paper.
3. **Model weights** are on Zenodo: [10.5281/zenodo.21824464](https://zenodo.org/records/21824464).

## Running these scripts

1. Install the fennec environment (see repo root `README.md`).
2. Download the dataset and weights from Zenodo.
3. Each script accepts `--help` for usage; pass your local data paths via CLI arguments.

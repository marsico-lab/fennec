![FENNEC Logo](documentation/FENNEC.png)

# FENNEC

**Fine-Tuned Ensemble of Neural Networks for siRNA Efficiency Characterization** — public code release accompanying the paper *"FENNEC: Fine-Tuned Ensemble Neural Networks Accelerate Chemically Modified siRNA Screening and Design"* (submitted to *Nature Computational Science*; link added on acceptance).

This repository provides the **inference, generative design, feature-engineering, and publication-analysis** code for FENNEC. Training code is not included in this release — the released checkpoints (see [Model weights](#model-weights)) are intended for inference and design.

---

## What FENNEC does

- Predicts siRNA knockdown efficacy from sequence + chemical modification pattern.
- Fuses RNA foundation-model embeddings (RNA-FM, RiNALMo, Orthrus 4-track) with engineered thermodynamic and chemistry features.
- Supports iterative *in-silico* modification search (greedy beam / open-world / closed-world) via `fennec.seq_designer`.
- Ships HELM ⇄ Fennec-notation bidirectional conversion for interoperability with cheminformatics pipelines (see `README_HELM_CONVERTER.md`).

## Install

```bash
conda env create -f environment.yml   # this already does `pip install -e .`
conda activate fennec

# Two GitHub-only packages:
pip install git+https://github.com/Zinthrow/BioInUtils.git
git clone https://github.com/bowang-lab/Orthrus.git
pip install -e Orthrus/ --no-deps

# GPU extras (requires CUDA 12.1 toolkit):
export TORCH_CUDA_ARCH_LIST=8.9   # your GPU's compute capability
pip install --no-build-isolation causal_conv1d==1.2.0.post2 mamba-ssm==1.2.0.post1

# Verify:
python tests/test_env.py
```

## Repository layout

```
src/fennec/                    # main package
  featurizer.py                #   sequence + modification tokenization
  preprocessing.py             #   feature-vector assembly
  helm_converter.py            #   HELM ⇄ Fennec bidirectional
  seq_designer/                #   siRNA design pipeline
  models/                      #   TCN + FM model definitions
  fm_utils.py                  #   FM loaders (RNA-FM, RiNALMo, Orthrus)
  plotter.py                   #   CV loss / density plotting + calc_key_stats
  support_files/               #   published seed-region toxicity lookup (Alnylam)

helper_scripts/                # data-prep utilities (mRNA fetch, in-vitro CSV preproc)
tests/                         # env + HELM converter + HELM validation tests
figures/                       # model_architecture.py (reproduces SFig 4)
publication_analyses/          # scripts producing the paper's Figs 2–7 and SFigs 1–11
                               # See publication_analyses/README.md for figure→code map
```

## Quick start: inference

The model takes **three** input tracks: the one-hot sequence/chemistry tensor, the meta-feature
matrix, and the RNA foundation-model embedding of the target transcript. `preprocess_input_tracks`
builds all three from a dataframe and caches the FM embeddings under `emb_cache_dir`.

```python
import torch
from fennec.models import MODELS
from fennec.preprocessing import preprocess_input_tracks

device = "cuda"

# Build all three input tracks (runs the FM once and caches to emb_cache_dir)
X, y, X_meta, X_emb, mrna_map_index = preprocess_input_tracks(
    dataframe,                  # needs siRNA / passenger / mRNA + assay metadata columns
    "emb_cache_dir",            # where the FM embeddings are written
    device=device,
    fm_type="orthrus_4track",
    slice_length=100,
    with_passenger=True,
    with_thermo=True,
)
X, X_meta, X_emb = X.float(), X_meta.float(), X_emb.float()
X_emb = X_emb[mrna_map_index]   # expand per-transcript embeddings to per-siRNA rows

# Instantiate with dimensions taken from the data, then load a Zenodo checkpoint
n, total_channels, seq_len = X.shape
_, fm_seq_len, fm_embed_dim = X_emb.shape

model = MODELS["TCN_resnet_lstm_ablate_LSTM"](   # == "TCN_FM_multi_head"
    input_channels=total_channels,
    meta_input_dim=X_meta.shape[-1],
    fm_seq_len=fm_seq_len,
    fm_embed_dim=fm_embed_dim,
    num_classes=1,
    regression=True,
).to(device)
model.load_state_dict(torch.load("path/to/checkpoint.pt", weights_only=False))
model.eval()

with torch.no_grad():
    preds = model(X.to(device), X_meta.to(device), X_emb.to(device)).squeeze(1).cpu().numpy()
```

For the multi-model ensemble pattern used in the paper, see `src/fennec/seq_designer/siRNA_design_pipeline.py` — it loads a set of checkpoints and averages predictions.

## Model weights and data availability

The curated training dataset **and** the pre-trained weights are archived together in a single
Zenodo deposit:

> **DOI:** [10.5281/zenodo.21824464](https://zenodo.org/records/21824464)
>
> https://zenodo.org/records/21824464

The weights are the 10-seed triple-Orthrus ensemble (`TCN_FM_multi_head`, also registered as
`TCN_resnet_lstm_ablate_LSTM`, with `orthrus_4track_triple` embeddings). The dataset comprises
patent-derived siRNA activity measurements curated from 42 patents. See the paper for full details
on data provenance, quality-control filters, and train/test splitting.

## Publication figure map

Each figure maps to a script in this repo. See `publication_analyses/README.md` for details.

| Figure / Table | What it shows | Where the code lives |
|---|---|---|
| **Fig 1** | Architecture cartoon | `documentation/FENNEC.png` + `figures/model_architecture.py` |
| **Fig 2** | Dataset overview (per-gene N, KD/mod distributions) | `publication_analyses/patent_stats/plot_dataset_overview.py`, `investigate_chemical_diversity_for_design/plot_chemical_diversity.py` |
| **Fig 3** | Model performance + ablation + ensembling gain | `publication_analyses/ablation/plot_ablation_fig3.py` |
| **Fig 4** | Data-efficiency curves (ACVR1C, JAK1) | `publication_analyses/ACVR1C_val/plot_data_efficiency_fig4.py`, `roche_val/plot_data_efficiency_fig4.py` |
| **Fig 5** | Integrated-Gradients attribution maps | `publication_analyses/deep_feature_importance/compute_integrated_gradients.py` |
| **Fig 6** | Greedy beam *in-silico* modification scan | `publication_analyses/chem_mod_walk_exp/plot_mod_enhancer_final_graph_helm.py` |
| **Fig 7** | AHSA1 experimental validation | see paper; design pipeline in `src/fennec/seq_designer/` |
| **SFig 1** | Assay/meta-context composition (transfection, concentration, cell line, duration, modification frequencies) | `publication_analyses/patent_stats/plot_dataset_overview.py` (`categorical_summaries.png`, `composition.png`) |
| **SFig 2-4** | Orthrus embedding correlations, MI, PCA | `publication_analyses/patent_stats/plot_embedding_correlations.py` |
| **SFig 5** | Detailed architecture | `figures/model_architecture.py` |
| **SFig 6-8** | AHSA1 benchmark and concentration-correlation panels | `publication_analyses/OligoFormer/plot_benchmark_comparison.py`, `plot_oligoformer_efficacy.py`, `plot_inter_dataset_validation.py` |
| **SFig 9** | Greedy beam search schematic | schematic only — no generating script |
| **SFig 10** | ACVR1C at 0.1/1/10 nM | `publication_analyses/ACVR1C_val/plot_acvr1c_sfig9.py` (filename retains an earlier figure number) |
| **SFig 11** | CV10 condition-grouped Spearman | not shipped |
| **Zero-shot APP / JAK1 tables** | Cross-experiment eval | `publication_analyses/APP_Val/table_zeroshot_app.py`, `roche_val/table_zeroshot_jak1.py` |

## Testing

```bash
pytest tests/                       # HELM converter round-trip + env sanity
python tests/test_env.py            # verify torch/FM stack loads
```

## Citation

Citation info will be added on paper acceptance.

## License

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). See `LICENSE` for the full text.

## Contact

Alexander Larsen — (contact via GitHub Issues on this repo, or via the paper's corresponding author).

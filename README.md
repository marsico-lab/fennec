![FENNEC Logo](documentation/FENNEC.png)

# FENNEC

**Fine-tuned Ensemble of Neural Networks for siRNA Efficacy Classification** — public code release accompanying the paper *"[paper title placeholder — link once accepted]"* (submitted to *Nucleic Acids Research*).

This repository provides the **inference, generative design, feature-engineering, and publication-analysis** code for FENNEC.

---

## What FENNEC does

- Predicts siRNA knockdown efficacy from sequence + chemical modification pattern.
- Fuses RNA foundation-model embeddings (RNA-FM, RiNALMo, Orthrus 4-track) with engineered thermodynamic and chemistry features.
- Supports iterative *in-silico* modification search (greedy beam / open-world / closed-world) via `fennec.seq_designer`.
- Ships HELM ⇄ Fennec-notation bidirectional conversion for interoperability with cheminformatics pipelines (see `README_HELM_CONVERTER.md`).

## Install

```bash
conda env create -f environment.yml
conda activate fennec
pip install -e .

# Two GitHub-only packages:
pip install --no-deps git+https://github.com/Zinthrow/BioInUtils.git
git clone https://github.com/bowang-lab/Orthrus.git
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
cp -r Orthrus/orthrus $SITE/

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
publication_analyses/          # scripts producing the paper's Figs 2–7 and SFigs 1–10
                               # See publication_analyses/README.md for figure→code map
```

## Quick start: inference

```python
from fennec.models import MODELS
from fennec.preprocessing import preprocess_input_tracks
from fennec.featurizer import helm_one_hot_encode
import torch

# Load one of the ensemble members (weights from Zenodo)
model = MODELS["TCN_FM_multi_head"](
    input_channels=54, meta_input_dim=61,
    fm_embed_dim=1536, attn_num_heads=4, attn_dim=191,
)
model.load_state_dict(torch.load("path/to/checkpoint.pt", map_location="cpu"), strict=True)
model.eval()

# Prepare inputs
feats = preprocess_input_tracks(dataframe)   # engineered thermo/chemistry features
onehot = helm_one_hot_encode(sequences, chemistry_annotations)

with torch.no_grad():
    preds = model(onehot, feats)
```

For the multi-model ensemble pattern used in the paper, see `src/fennec/seq_designer/siRNA_design_pipeline.py` — it loads a set of checkpoints and averages predictions.

## Model weights

Pre-trained weights for the triple Orthrus model (10-seed ensemble, `TCN_FM_multi_head` + `orthrus_4track_triple`) are available on Zenodo:

> **DOI:** [10.5281/zenodo.21824464](https://zenodo.org/records/21824464)
>
> https://zenodo.org/records/21824464

## Data availability

The training dataset and pre-trained model weights are archived on Zenodo:

> **DOI:** [10.5281/zenodo.21824464](https://zenodo.org/records/21824464)

The dataset comprises patent-derived siRNA activity measurements curated from 43 patent families. See the paper for full details on data provenance, quality-control filters, and train/test splitting.

## Publication figure map

Each figure maps to a script in this repo. See `publication_analyses/README.md` for details.

| Figure / Table | What it shows | Where the code lives |
|---|---|---|
| **Fig 1** | Architecture cartoon | `documentation/FENNEC.png` + `figures/model_architecture.py` |
| **Fig 2** | Dataset overview (per-gene N, KD/mod distributions) | `publication_analyses/patent_stats/plot_dataset_overview.py`, `investigate_chemical_diversity_for_design/plot_chemical_diversity.py` |
| **Fig 3** | Model performance + ablation + ensembling gain | `publication_analyses/ablation/plot_ablation_fig3.py` |
| **Fig 4** | Data-efficiency curves (ACVR1C, JAK1) | `publication_analyses/ACVR1C_val/plot_data_efficiency_fig4.py`, `roche_val/plot_data_efficiency_fig4.py` |
| **Fig 5** | Integrated-Gradients attribution maps | `publication_analyses/deep_feature_importance/compute_integrated_gradients.py` |
| **Fig 6** | Greedy beam *in-silico* modification scan | `publication_analyses/chem_mod_walk_exp/plot_mod_enhancer_final_graph.py` |
| **Fig 7** | AHSA1 experimental validation | see paper; design pipeline in `src/fennec/seq_designer/` |
| **SFig 1-3** | Orthrus embedding correlations, MI, PCA | `publication_analyses/patent_stats/plot_embedding_correlations.py` |
| **SFig 4** | Detailed architecture | `figures/model_architecture.py` |
| **SFig 5-7** | AHSA1 benchmark panels | `publication_analyses/OligoFormer/plot_benchmark_comparison.py`, `plot_oligoformer_efficacy.py` |
| **SFig 9** | ACVR1C at 0.1/1/10 nM | `publication_analyses/ACVR1C_val/plot_acvr1c_sfig9.py` |
| **Zero-shot APP / JAK1 tables** | Cross-experiment eval | `publication_analyses/APP_Val/table_zeroshot_app.py`, `roche_val/table_zeroshot_jak1.py` |

## Testing

```bash
pytest tests/                       # HELM converter round-trip + env sanity
python tests/test_env.py            # verify torch/FM stack loads
```

## Citation

Citation info will be added on paper acceptance.

## License

This work is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). See `LICENSE` for the full text.

## Contact

Alexander Larsen — (contact via GitHub Issues on this repo, or via the paper's corresponding author).

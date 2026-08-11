#!/usr/bin/env bash
# =============================================================================
# fennec setup script
# Installs all dependencies including GitHub-only packages that cannot be
# resolved through conda/pip alone.
#
# Usage:
#   bash setup.sh              # auto-detects conda or pip
#   bash setup.sh --pip-only   # skip conda, use pip/uv directly
#   bash setup.sh --skip-gpu   # skip mamba-ssm / causal_conv1d (no GPU)
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${BOLD}[fennec]${RESET} $*"; }
success() { echo -e "${GREEN}  ✓${RESET} $*"; }
warn()    { echo -e "${YELLOW}  ⚠${RESET} $*"; }
error()   { echo -e "${RED}  ✗${RESET} $*"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
PIP_ONLY=false
SKIP_GPU=false
for arg in "$@"; do
  case $arg in
    --pip-only)  PIP_ONLY=true ;;
    --skip-gpu)  SKIP_GPU=true ;;
    -h|--help)
      echo "Usage: bash setup.sh [--pip-only] [--skip-gpu]"
      echo "  --pip-only   Use pip/uv instead of conda"
      echo "  --skip-gpu   Skip mamba-ssm and causal_conv1d (CPU-only mode)"
      exit 0 ;;
    *) warn "Unknown argument: $arg" ;;
  esac
done

# ── Determine pip executable ──────────────────────────────────────────────────
# Prefer the venv pip, then uv pip, then plain pip3/pip
find_pip() {
  if [[ -f ".venv/bin/pip" ]]; then
    echo ".venv/bin/pip"
  elif [[ -f ".venv/bin/python3" ]]; then
    echo ".venv/bin/python3 -m pip"
  elif command -v uv &>/dev/null; then
    # uv pip needs a target; use the active venv or system
    echo "uv pip install --system"
  elif command -v pip3 &>/dev/null; then
    echo "pip3"
  elif command -v pip &>/dev/null; then
    echo "pip"
  else
    error "No pip found. Install pip or activate a virtual environment first."
    exit 1
  fi
}

uv_pip_install() {
  # Wrapper that always installs into the project venv when uv is available,
  # otherwise falls back to whatever pip we found.
  if command -v uv &>/dev/null; then
    local python_flag=""
    if [[ -f ".venv/bin/python3" ]]; then
      python_flag="--python .venv/bin/python3"
    fi
    uv pip install $python_flag "$@"
  else
    $PIP "$@"
  fi
}

# ── Step 0: ViennaRNA (system package) ───────────────────────────────────────
install_viennarna() {
  info "Checking ViennaRNA (RNAfold)..."
  if command -v RNAfold &>/dev/null; then
    success "RNAfold $(RNAfold --version 2>&1 | head -1) already installed"
    return
  fi

  if [[ "$CONDA_DEFAULT_ENV" != "" ]] && command -v conda &>/dev/null; then
    info "Installing ViennaRNA via conda..."
    conda install -y -c bioconda -c conda-forge viennarna
  elif command -v apt-get &>/dev/null; then
    info "Installing ViennaRNA via apt..."
    sudo apt-get install -y vienna-rna
  elif command -v brew &>/dev/null; then
    info "Installing ViennaRNA via brew..."
    brew install viennarna
  else
    warn "Cannot auto-install ViennaRNA. Please install it manually:"
    warn "  conda: conda install -c bioconda viennarna"
    warn "  apt:   sudo apt-get install vienna-rna"
    warn "  brew:  brew install viennarna"
    warn "Thermo features (--with-thermo) will not work without RNAfold."
  fi
}

# ── Step 1: Conda environment (optional) ─────────────────────────────────────
setup_conda() {
  if $PIP_ONLY; then return; fi
  if ! command -v conda &>/dev/null; then
    warn "conda not found — skipping conda env creation, using pip directly."
    PIP_ONLY=true
    return
  fi

  info "Creating conda environment from environment.yml..."
  if conda env list | grep -q "^fennec "; then
    warn "Conda env 'fennec' already exists — skipping creation."
    warn "To recreate: conda env remove -n fennec && bash setup.sh"
  else
    conda env create -f environment.yml
    success "Conda env 'fennec' created"
  fi
  warn "Run 'conda activate fennec' before continuing if not already active."
}

# ── Step 2: Core pip packages ─────────────────────────────────────────────────
install_core_pip() {
  info "Installing core pip dependencies..."
  uv_pip_install \
    numpy pandas scipy scikit-learn \
    matplotlib seaborn tqdm click \
    biopython edlib xmltodict sh \
    rna-fm transformers
  success "Core packages installed"
}

# ── Step 3: primer3-py (version depends on Python) ───────────────────────────
install_primer3() {
  info "Installing primer3-py..."
  PYTHON_BIN=""
  if [[ -f ".venv/bin/python3" ]]; then
    PYTHON_BIN=".venv/bin/python3"
  elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
  fi

  PY_MINOR=$($PYTHON_BIN -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "99")
  PY_MAJOR=$($PYTHON_BIN -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "3")

  if [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -le 12 ]]; then
    # Python 3.12 and below: use the version BioInUtils expects
    uv_pip_install "primer3-py==2.0.3" || {
      warn "primer3-py==2.0.3 failed to build, falling back to latest"
      uv_pip_install "primer3-py"
    }
  else
    # Python 3.13+: 2.0.3 uses deprecated Cython syntax that won't compile
    warn "Python 3.13+ detected — installing primer3-py>=2.3.0 (API-compatible)"
    uv_pip_install "primer3-py"
  fi
  success "primer3-py installed"
}

# ── Step 4: BioInUtils ────────────────────────────────────────────────────────
install_bioinutils() {
  info "Installing BioInUtils (Zinthrow/BioInUtils)..."

  PYTHON_BIN=""
  if [[ -f ".venv/bin/python3" ]]; then
    PYTHON_BIN=".venv/bin/python3"
  elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
  fi

  if $PYTHON_BIN -c "import bioinutils" &>/dev/null 2>&1; then
    success "bioinutils already installed"
    return
  fi

  TMPDIR=$(mktemp -d)
  trap "rm -rf $TMPDIR" EXIT

  git clone --depth 1 https://github.com/Zinthrow/BioInUtils.git "$TMPDIR/BioInUtils"

  # Relax the primer3-py pin so it doesn't conflict with Python 3.13+
  sed -i 's/primer3-py == 2\.0\.3/primer3-py/' "$TMPDIR/BioInUtils/requirements.txt"

  # Install without enforcing the (now-relaxed) deps to avoid solver conflicts
  uv_pip_install --no-deps "$TMPDIR/BioInUtils/"

  if $PYTHON_BIN -c "import bioinutils" &>/dev/null 2>&1; then
    success "bioinutils installed"
  else
    error "bioinutils installation failed. Check the output above."
    exit 1
  fi
}

# ── Step 5: Orthrus ───────────────────────────────────────────────────────────
install_orthrus() {
  info "Installing Orthrus (bowang-lab/Orthrus)..."

  PYTHON_BIN=""
  if [[ -f ".venv/bin/python3" ]]; then
    PYTHON_BIN=".venv/bin/python3"
  elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
  fi

  if $PYTHON_BIN -c "from orthrus.gk_utils import seq_to_oh" &>/dev/null 2>&1; then
    success "orthrus already installed"
    return
  fi

  TMPDIR_ORTH=$(mktemp -d)
  trap "rm -rf $TMPDIR_ORTH" EXIT

  git clone --depth 1 https://github.com/bowang-lab/Orthrus.git "$TMPDIR_ORTH/Orthrus"

  # The Orthrus setup.py builds a Cython extension (_sequence) and requires
  # GenomeKit — neither is needed for fennec, which only uses gk_utils.py.
  # Fastest fix: copy the pure-Python package directly into site-packages.
  SITE=$($PYTHON_BIN -c "import site; print(site.getsitepackages()[0])")
  cp -r "$TMPDIR_ORTH/Orthrus/orthrus" "$SITE/"

  if $PYTHON_BIN -c "from orthrus.gk_utils import seq_to_oh" &>/dev/null 2>&1; then
    success "orthrus installed (gk_utils)"
  else
    error "orthrus installation failed. Check the output above."
    exit 1
  fi
}

# ── Step 6: GPU extras (mamba-ssm / causal_conv1d) ───────────────────────────
install_gpu_extras() {
  if $SKIP_GPU; then
    warn "Skipping GPU extras (--skip-gpu). Orthrus 4-track FM will not be available."
    return
  fi

  info "Installing Orthrus GPU backbone (causal_conv1d + mamba-ssm)..."
  warn "These require CUDA 12.1 toolkit headers. Skip with --skip-gpu if on CPU."

  uv_pip_install "causal_conv1d==1.2.0.post2" && \
  uv_pip_install "mamba-ssm==1.2.0.post1" && \
  success "GPU extras installed" || {
    warn "GPU extras failed (CUDA toolkit may be missing)."
    warn "Re-run with --skip-gpu to suppress this, or install manually:"
    warn "  pip install causal_conv1d==1.2.0.post2 mamba-ssm==1.2.0.post1"
  }
}

# ── Step 7: fennec itself ─────────────────────────────────────────────────────
install_fennec() {
  info "Installing fennec (editable)..."
  uv_pip_install --no-deps -e .
  success "fennec installed"
}

# ── Step 8: Verify ────────────────────────────────────────────────────────────
run_tests() {
  info "Running environment smoke-test..."
  PYTHON_BIN=""
  if [[ -f ".venv/bin/python3" ]]; then
    PYTHON_BIN=".venv/bin/python3"
  elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
  fi

  if $PYTHON_BIN tests/test_env.py; then
    echo ""
    success "All checks passed — fennec is ready to use!"
  else
    echo ""
    error "Some checks failed. See output above."
    exit 1
  fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}============================================${RESET}"
echo -e "${BOLD}  fennec environment setup${RESET}"
echo -e "${BOLD}============================================${RESET}"
echo ""

setup_conda
install_viennarna
install_core_pip
install_primer3
install_bioinutils
install_orthrus
install_gpu_extras
install_fennec
run_tests

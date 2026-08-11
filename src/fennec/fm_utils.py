import re
import glob
import torch
import os
import pandas as pd
from Bio.Seq import reverse_complement
from tqdm import tqdm
import edlib
import logging

try:
    import fm
except ImportError:
    pass

try:
    from rinalmo.pretrained import get_pretrained_model
except ImportError:
    pass

try:
    from transformers import AutoModel
    from orthrus.gk_utils import seq_to_oh
except ImportError:
    pass


logger = logging.getLogger()


# Load sequence data
# embed_map = pd.read_csv('RNA-FM_outputs/index_siRNA_mRNA.csv', index_col=0)

# Use GPU

def get_rinalmo_model(device="cuda"):

    # Load pretrained model
    model, alphabet = get_pretrained_model(model_name="giga-v1")
    model = model.to(device=device)
    model.eval()

    return model, alphabet

def get_orthrus_4track_model(device="cuda"):
    model_dir_path = "antichronology/orthrus-4-track"
    try:
        from transformers import AutoModel
        orthrus_4 = AutoModel.from_pretrained(
            model_dir_path,
            trust_remote_code=True
        ).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load Orthrus model. Ensure that the 'orthrus' package is installed correctly. Original error: {e}" +
                           f" The model can be downloaded from hugging face and saved as: {model_dir_path}")
    return orthrus_4


def get_rna_fm_model():
    # Load mRNA-FM model
    model2, alphabet2 = fm.pretrained.rna_fm_t12()
    batch_converter2 = alphabet2.get_batch_converter()
    model2.eval()  # disables dropout for deterministic results

    return model2, batch_converter2


def map_format_sirna_mrna(data: dict, slice_length: int = 100, slice_from: str = 'center'):

    valid_slice_from = {'center', 'five_prime', 'three_prime', 'five_prime_end', 'three_prime_start'}
    if slice_from not in valid_slice_from:
        raise ValueError(
            f"Invalid slice_from '{slice_from}'. Choose one of {sorted(valid_slice_from)}."
        )

    missing_tot = 0
    map_seqs = {}

    for si in data:
        mr = data[si].replace('U', 'T')
        if slice_from == 'five_prime':
            mslice = mr[:slice_length]
        elif slice_from == 'three_prime':
            mslice = mr[-slice_length:]
        elif slice_from == 'five_prime_end':
            # Last slice_length nt of the 5' UTR — the region immediately upstream of the CDS.
            mslice = mr[-slice_length:]
        elif slice_from == 'three_prime_start':
            # First slice_length nt of the 3' UTR — the region immediately downstream of the stop codon.
            mslice = mr[:slice_length]
        else:
            # Center mode slices around the aligned siRNA target neighborhood, not the full mRNA midpoint.
            flank = (slice_length - len(si)) // 2
            sit = si.replace("U", "T")
            sit = reverse_complement(sit)
            out = edlib.align(sit, str(mr), mode='HW', task="locations")
            start, end = out['locations'][0]
            editdistance = out['editDistance']
            if editdistance <= 7:
                if start - flank < 0:
                    start, end = 0, slice_length
                else:
                    start, end = start - flank, end + flank
                mslice = mr[start:end]
            else:
                mslice = ("N" * flank) + sit 
                logger.debug(
                         f"Reverse complement of siRNA not found in mRNA for index {si}. Edit distance: {editdistance}" +
                         " exceeds 7 tolerated mismatchs. Potential siRNA/mRNA flip. Assigning mRNA sequence to be siRNA."
                     )
                missing_tot += 1
        if len(mslice) < slice_length:
            mslice = mslice + "N" * (slice_length - len(mslice))
        map_seqs[si] = mslice[:slice_length]

    # TODO logging of this
    if missing_tot > 0:
        logger.warning(f'Missing a total of {missing_tot} matches out of {len(data)} unique mRNA.')

    return map_seqs


def process_mrnas_through_rna_fm(
    mrnas: list, 
    model2, 
    batch_converter2, 
    outdir, 
    batch_size,
    device='cuda'
):
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for i in tqdm(range(0, len(mrnas), batch_size)):

        _, _, batch_tokens = batch_converter2(mrnas[i: i + batch_size])

        # Extract embeddings (on CPU)
        with torch.no_grad():
            results = model2(batch_tokens.to(device), repr_layers=[12])
        token_embeddings = results["representations"][12].to('cpu')

        torch.save(
            token_embeddings,
            outdir
            + f"RNA-FM_embeddings_mrna__{i}:{i+batch_size}__flanking_N.pth",
        )

def process_mrnas_through_rinalmo(mrnas: list, 
    model, 
    alphabet, 
    outdir, 
    batch_size,
    only_cls_eos=False,
    device='cuda'):


    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for i in tqdm(range(batch_size, len(mrnas)+batch_size, batch_size)):
        fpath =  outdir + f'rinalmo_embeddings_mrna__{i-batch_size}:{i}.pth'
        #if os.path.exists(fpath):
        #   c#ontinue
        # Split sequences into batches
        seqs = mrnas[i-batch_size:i]


        # Process first batch
        tokens = torch.tensor(alphabet.batch_tokenize(seqs), dtype=torch.int64, device=device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model(tokens)
        tokens = tokens.to('cpu')
        representation = outputs["representation"].to('cpu')
        if only_cls_eos:
            representation = representation[:, [0, -1], :]

        # Save first batch outputs
        torch.save(representation, fpath)

        del outputs, tokens


def sequence_to_orthrus_representation(mrnas: list, model: torch.nn.Module, device: str):
    """
    Converts a list of mRNA sequences into their orthRus model representations.

    Args:
        mrnas (list of str): List of mRNA sequences to be converted.
        model (torch.nn.Module): The orthRus model used to generate representations.
        device (torch.device or str): The device to perform computations on (e.g., 'cpu' or 'cuda').

    Returns:
        torch.Tensor: Tensor containing the representations of the input mRNA sequences,
                      with shape (batch_size, 1, representation_dim), detached and moved to CPU.
    """
    one_hot = [seq_to_oh(v) for v in mrnas]
    torch_one_hot = torch.tensor(one_hot, dtype=torch.float32)
    torch_one_hot = torch_one_hot.permute(0, 2, 1)
    torch_one_hot = torch_one_hot.to(device=device)
    lengths = torch.tensor([torch_one_hot.shape[2]]).to(device=device)
    representation = model.representation(torch_one_hot, lengths)
    representation = representation.unsqueeze(1).detach().to('cpu')

    del torch_one_hot

    return representation

def process_mrnas_through_orthrus_4track(mrnas: list, 
    model,  
    outdir,
    batch_size=1000,
    device='cuda',
    disable_tqdm: bool = True):

    # tests have just confirmed that batch processing is required
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for i in tqdm(range(0, len(mrnas), batch_size), disable=disable_tqdm):
        fpath =  outdir + f'orthrus_4track_embeddings_mrna__{i}:{min(i+batch_size, len(mrnas))}.pth'
        representation = sequence_to_orthrus_representation(mrnas[i:min(i+batch_size, len(mrnas))], model, device)
        # dim should be N, 1, final rep dim

        torch.save(representation, fpath)
        del representation
    

def reload_fm_outputs(outdir):
    fils = {}

    for fil in glob.glob(f'{outdir}*'):
        r = re.search(r'(?<=mrna__)\d+(?=:)', fil)
        if not r:
            continue
        r = r.group()
        r = int(r)
        fils[r] = fil

    all_tensors = []

    for i in sorted(fils):
        fil = fils[i]
        all_tensors.append(torch.load(fil, map_location=torch.device('cpu')))
        os.remove(fil)
    fm_embeddings = torch.concat(all_tensors)
    fm_embeddings.shape

    return fm_embeddings


# Legacy function to reload RNA-FM outputs
# This function is kept for backward compatibility
def reload_rna_fm_outputs(outdir):
    """
    Reload RNA-FM outputs from the specified directory.
    """
    return reload_fm_outputs(outdir)


def run_mrna_through_one_hot(data: dict, 
                             outdir: str="One_Hot_Outputs/", 
                             device: str='cpu', 
                             slice_length: int = 100,
                             slice_from: str = 'center'):

    map_seqs = map_format_sirna_mrna(data, slice_length=slice_length, slice_from=slice_from)

    mrnas = list(map_seqs.items())
    ttdf = pd.DataFrame(mrnas, columns=["siRNA", "mRNA"])
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    index_csv_path = outdir + "index_siRNA_mRNA.csv"
    ttdf.to_csv(index_csv_path)

    nuc_dict = {"A": [1, 0, 0, 0], "T": [0, 1, 0, 0], "G": [0, 0, 1, 0], "C": [0, 0, 0, 1], "N": [0, 0, 0, 0], "R": [1, 0, 1, 0]}
    invalid_chars = {}

    def make_one_hot(seq):
        one_arr = []
        for nuc in str(seq).upper():
            if nuc not in nuc_dict:
                invalid_chars[nuc] = invalid_chars.get(nuc, 0) + 1
                nuc = "N"
            one_arr.append(nuc_dict[nuc])

        return one_arr
    
    one_hot_list = [make_one_hot(map_seqs[si]) for si in map_seqs]

    if invalid_chars:
        invalid_summary = ", ".join([f"{k}:{v}" for k, v in sorted(invalid_chars.items(), key=lambda x: x[0])])
        logger.warning(f"Encountered non-nucleotide mRNA characters during one-hot encoding; mapped to 'N' -> {invalid_summary}")

    fm_embeddings = torch.Tensor(one_hot_list)
    batch_size = len(fm_embeddings)
    
    fpath =  outdir + f'one_hot_embeddings_mrna__{0}:{batch_size}.pth'

    torch.save(fm_embeddings, fpath)


def run_mrna_through_gene_one_hot(data: dict,
                                  gene_dict: dict,
                                  outdir: str = "Gene_One_Hot_Outputs/",
                                  device: str = 'cpu',
                                  slice_length: int = 100,
                                  slice_from: str = 'center'):
    """
    Create a one-hot encoding over unique gene identities.

    The mRNA is still sliced via ``map_format_sirna_mrna`` so that
    ``index_siRNA_mRNA.csv`` is produced (required by the thermo
    feature path), but the embedding tensor itself is a simple
    one-hot vector whose length equals the number of unique genes
    in the dataset.

    Parameters
    ----------
    data : dict
        Mapping of siRNA -> mRNA (same as other ``run_mrna_through_*`` fns).
    gene_dict : dict
        Mapping of siRNA -> gene_target_symbol_name.
    outdir : str
        Directory to write outputs.
    device : str
        Unused (kept for API consistency).
    slice_length : int
        Passed to ``map_format_sirna_mrna`` for thermo slice.
    slice_from : str
        Passed to ``map_format_sirna_mrna`` for thermo slice.
    """
    # Still slice mRNAs so embed_map (index_siRNA_mRNA.csv) is available
    # for downstream thermo feature calculation.
    map_seqs = map_format_sirna_mrna(data, slice_length=slice_length, slice_from=slice_from)

    mrnas = list(map_seqs.items())
    ttdf = pd.DataFrame(mrnas, columns=["siRNA", "mRNA"])
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    index_csv_path = outdir + "index_siRNA_mRNA.csv"
    ttdf.to_csv(index_csv_path)

    # Build stable sorted list of all genes
    all_genes = sorted(set(gene_dict.values()))
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    n_genes = len(all_genes)

    logger.info(f"Gene one-hot encoding: {n_genes} unique genes")

    # Build one-hot tensor aligned to the embed_map order (unique siRNAs)
    one_hot_list = []
    for si, _ in mrnas:
        vec = [0.0] * n_genes
        gene = gene_dict.get(si, "Unknown")
        idx = gene_to_idx.get(gene)
        if idx is not None:
            vec[idx] = 1.0
        one_hot_list.append(vec)

    # Shape: (N_unique_sirna, 1, n_genes) — single "token" per sample
    fm_embeddings = torch.tensor(one_hot_list, dtype=torch.float32).unsqueeze(1)
    batch_size = len(fm_embeddings)
    fpath = outdir + f'gene_one_hot_embeddings_mrna__{0}:{batch_size}.pth'
    torch.save(fm_embeddings, fpath)


def run_mrna_through_rna_fm(data: dict, 
                            outdir="RNA-FM_outputs/", 
                            device='cuda', 
                            batch_size=500,
                            slice_length:int = 100,
                            slice_from: str = 'center',
                            model=None):
    if model is None:
        model2, batch_converter2 = get_rna_fm_model()
        model2.to(device)
    else:
        model2, batch_converter2 = model

    map_seqs = map_format_sirna_mrna(data, slice_length=slice_length, slice_from=slice_from)
    mrnas = list(map_seqs.items())
    ttdf = pd.DataFrame(mrnas, columns=["siRNA", "mRNA"])
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    index_csv_path = outdir + "index_siRNA_mRNA.csv"
    ttdf.to_csv(index_csv_path)

    process_mrnas_through_rna_fm(mrnas, model2, batch_converter2, outdir, batch_size, device=device)

def run_mrna_through_rinalmo(data: dict, 
                             outdir:str='RiNALMo_outputs/', 
                             device: str='cuda', 
                             only_cls_eos: bool = False,
                             batch_size:int=1000,
                             slice_length: int = 100,
                             slice_from: str = 'center',
                             model=None):
    
    if model is None:
        model, alphabet = get_rinalmo_model(device=device)
    else:
        model, alphabet = model
    map_seqs = map_format_sirna_mrna(data, slice_length=slice_length, slice_from=slice_from)
    mrnas = list(map_seqs.items())
    ttdf = pd.DataFrame(mrnas, columns=["siRNA", "mRNA"])
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    index_csv_path = outdir + "index_siRNA_mRNA.csv"
    ttdf.to_csv(index_csv_path)

    mrna_seqs = [item[1] for item in mrnas]

    process_mrnas_through_rinalmo(mrna_seqs, model, alphabet, outdir=outdir, batch_size=batch_size, device=device, only_cls_eos=only_cls_eos)

def run_mrna_through_orthrus_4track(data: dict, 
                                   outdir:str='Orthrus_4track_outputs/', 
                                   device: str='cuda', 
                                   slice_length: int = 100,
                                   slice_from: str = 'center',
                                   model=None):
    if model is None:
        model = get_orthrus_4track_model(device=device)
    map_seqs = map_format_sirna_mrna(data, slice_length=slice_length, slice_from=slice_from)
    mrnas = list(map_seqs.items())
    ttdf = pd.DataFrame(mrnas, columns=["siRNA", "mRNA"])
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    index_csv_path = outdir + "index_siRNA_mRNA.csv"
    ttdf.to_csv(index_csv_path)

    mrna_seqs = [item[1] for item in mrnas]

    process_mrnas_through_orthrus_4track(mrna_seqs, model, outdir=outdir, device=device)


# ---------------------------------------------------------------------------
# Triple-slice helpers: embed center + five_prime_end + three_prime_start and
# concatenate along the sequence-length dimension so the attention-pooling
# layer sees context from all three regions.
# ---------------------------------------------------------------------------

def _embed_slice_rna_fm(data: dict, model2, batch_converter2,
                        slice_str: str, slice_length: int, device: str,
                        batch_size: int = 500):
    """Return (N, L+2, 640) embeddings for one named slice."""
    map_seqs = map_format_sirna_mrna(data, slice_length=slice_length, slice_from=slice_str)
    mrnas = list(map_seqs.items())
    all_embs = []
    for i in range(0, len(mrnas), batch_size):
        _, _, batch_tokens = batch_converter2(mrnas[i:i + batch_size])
        with torch.no_grad():
            results = model2(batch_tokens.to(device), repr_layers=[12])
        all_embs.append(results["representations"][12].to('cpu'))
    return torch.cat(all_embs, dim=0)  # (N, L+2, 640)


def run_mrna_through_rna_fm_triple(
        data_center: dict,
        data_five_end: dict,
        data_three_start: dict,
        outdir: str = 'RNA-FM_triple_outputs/',
        device: str = 'cuda',
        slice_length: int = 50,
        batch_size: int = 500,
        model=None):
    """Embed center + five_prime + three_prime_start slices with RNA-FM,
    concatenate along the sequence-length dimension, and save.

    Output shape per siRNA: (3*(slice_length+2), 640).
    The index_siRNA_mRNA.csv maps using the center-slice mRNA for thermo.
    """
    if model is None:
        model2, batch_converter2 = get_rna_fm_model()
        model2.to(device)
    else:
        model2, batch_converter2 = model

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Write index CSV using center slice for downstream thermo compatibility.
    map_seqs_center = map_format_sirna_mrna(data_center, slice_length=slice_length, slice_from='center')
    mrnas_center = list(map_seqs_center.items())
    pd.DataFrame(mrnas_center, columns=["siRNA", "mRNA"]).to_csv(outdir + "index_siRNA_mRNA.csv")

    emb_center = _embed_slice_rna_fm(data_center, model2, batch_converter2,
                                     'center', slice_length, device, batch_size)
    emb_five   = _embed_slice_rna_fm(data_five_end, model2, batch_converter2,
                                     'five_prime', slice_length, device, batch_size)
    emb_three  = _embed_slice_rna_fm(data_three_start, model2, batch_converter2,
                                     'three_prime_start', slice_length, device, batch_size)

    # (N, 3*(L+2), 640)  — preserved in order: 5'end·center·3'start
    combined = torch.cat([emb_five, emb_center, emb_three], dim=1)
    torch.save(combined, outdir + f'rnafm_triple_embeddings_mrna__0:{len(combined)}.pth')


def run_mrna_through_orthrus_4track_triple(
        data_center: dict,
        data_five_end: dict,
        data_three_start: dict,
        outdir: str = 'Orthrus_triple_outputs/',
        device: str = 'cuda',
        slice_length: int = 150,
        model=None):
    """Embed center + five_prime + three_prime_start slices with Orthrus,
    concatenate along dim-1 (the 'sequence' dimension of length 1 per slice),
    and save.

    Output shape per siRNA: (3, orthrus_dim) — three pooled context vectors.
    """
    if model is None:
        model = get_orthrus_4track_model(device=device)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    def _embed_slice_orthrus(data: dict, slice_str: str):
        map_seqs = map_format_sirna_mrna(data, slice_length=slice_length, slice_from=slice_str)
        seqs = [v for _, v in map_seqs.items()]
        repr_list = []
        batch_size = 500
        for i in range(0, len(seqs), batch_size):
            rep = sequence_to_orthrus_representation(seqs[i:i + batch_size], model, device)
            repr_list.append(rep)  # each (batch, 1, D)
        return torch.cat(repr_list, dim=0)  # (N, 1, D)

    # Write index CSV from center-slice mRNA for thermo compatibility.
    map_seqs_center = map_format_sirna_mrna(data_center, slice_length=slice_length, slice_from='center')
    pd.DataFrame(list(map_seqs_center.items()), columns=["siRNA", "mRNA"]).to_csv(
        outdir + "index_siRNA_mRNA.csv")

    emb_center = _embed_slice_orthrus(data_center, 'center')   # (N, 1, D)
    emb_five   = _embed_slice_orthrus(data_five_end, 'five_prime')
    emb_three  = _embed_slice_orthrus(data_three_start, 'three_prime_start')

    combined = torch.cat([emb_five, emb_center, emb_three], dim=2)  # (N, 1, 3*D)
    torch.save(combined, outdir + f'orthrus_triple_embeddings_mrna__0:{len(combined)}.pth')


# ---------------------------------------------------------------------------
# Token / one-hot savers for LoRA fine-tuning models.
# Instead of running the frozen FM offline, we save the raw sequences so the
# LoRA model can run the FM inside its own forward pass.
# ---------------------------------------------------------------------------

def save_rna_fm_tokens(
        data: dict,
        outdir: str = 'RNAFM_LoRA_inputs/',
        device: str = 'cuda',
        slice_length: int = 50,
        slice_from: str = 'center',
        model=None):
    """Tokenise mRNA slices with RNA-FM's alphabet and save token ID tensors.

    Saved shape: (N, 1, seq_len) as float32, where seq_len = slice_length + 2
    (CLS and EOS tokens).  Stored as float for compatibility with the
    standard reload_fm_outputs / TensorDataset pipeline; the LoRA model
    converts back to long integers in its forward method.
    """
    if model is None:
        _, alphabet = fm.pretrained.rna_fm_t12()
        batch_converter = alphabet.get_batch_converter()
    else:
        _, batch_converter = model

    map_seqs = map_format_sirna_mrna(data, slice_length=slice_length, slice_from=slice_from)
    mrnas = list(map_seqs.items())

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    pd.DataFrame(mrnas, columns=["siRNA", "mRNA"]).to_csv(outdir + "index_siRNA_mRNA.csv")

    all_tokens = []
    batch_size = 500
    for i in range(0, len(mrnas), batch_size):
        _, _, batch_tokens = batch_converter(mrnas[i:i + batch_size])
        all_tokens.append(batch_tokens)  # (B, seq_len) long

    tokens = torch.cat(all_tokens, dim=0)                  # (N, seq_len)
    tokens_3d = tokens.unsqueeze(1).float()                # (N, 1, seq_len)
    torch.save(tokens_3d, outdir + f'rnafm_tokens_mrna__0:{len(tokens_3d)}.pth')


def save_orthrus_one_hot(
        data: dict,
        outdir: str = 'Orthrus_LoRA_inputs/',
        device: str = 'cuda',
        slice_length: int = 150,
        slice_from: str = 'center'):
    """Slice mRNAs and save 4-track one-hot tensors for online Orthrus LoRA.

    Saved shape: (N, seq_len, 4) as float32.  The LoRA model permutes to
    (N, 4, seq_len) before calling orthrus.representation().
    """
    from orthrus.gk_utils import seq_to_oh

    map_seqs = map_format_sirna_mrna(data, slice_length=slice_length, slice_from=slice_from)
    mrnas = list(map_seqs.items())

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    pd.DataFrame(mrnas, columns=["siRNA", "mRNA"]).to_csv(outdir + "index_siRNA_mRNA.csv")

    seqs = [v for _, v in mrnas]
    one_hot = torch.tensor([seq_to_oh(s) for s in seqs], dtype=torch.float32)  # (N, seq_len, 4)
    torch.save(one_hot, outdir + f'orthrus_onehot_mrna__0:{len(one_hot)}.pth')



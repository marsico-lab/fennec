import torch
from torch import nn
from .featurizer import categorize_component, ALL_VOCAB

BASE_VOCAB = {"A": 1, "C": 2, "G": 3, "U": 4, "T": 5, "PAD": 6, "UNK": 0}
SUGAR_VOCAB = {"r": 1, "d": 2, "gn": 3, "x": 4, "PAD": 5, "UNK": 0}
SUGAR_MOD_VOCAB = {"": 0, "2OMe": 1, "f": 2, "s": 3, "hd": 4, "+": 5, "-": 6, "PAD": 7, "UNK": 7}
BACKBONE_VOCAB = {"": 0, "P": 1, "s": 2, "VP": 3, "L96": 4, "PAD": 5, "UNK": 0}


def get_siRNA_embeddings(base_dim=16,
                         sugar_dim=8,
                         sugar_mod_dim=8,
                         backbone_dim=8):
    # Define embedding layers for each component (choose dimensions as desired)
    base_embedding = nn.Embedding(num_embeddings=len(BASE_VOCAB), embedding_dim=base_dim)
    sugar_embedding = nn.Embedding(num_embeddings=len(SUGAR_VOCAB), embedding_dim=sugar_dim)
    sugar_mod_embedding = nn.Embedding(num_embeddings=len(SUGAR_MOD_VOCAB), embedding_dim=sugar_mod_dim)
    backbone_embedding = nn.Embedding(num_embeddings=len(BACKBONE_VOCAB), embedding_dim=backbone_dim)

    return base_embedding, sugar_embedding, sugar_mod_embedding, backbone_embedding


def embed_chem_sequence(seq: str,
                        base_embedding,
                        sugar_embedding,
                        sugar_mod_embedding,
                        backbone_embedding,
                        pad_length: int = None):
    """
    Given a space-separated chemically modified sequence (e.g., "A mC G U mA"),
    this function:
      1. Splits the sequence into tokens.
      2. Uses `categorize_component` to extract the four components for each token.
      3. Converts each component into an index using the provided vocab dictionaries.
      4. Looks up each component’s embedding using the corresponding embedding layer.
      5. Concatenates these embeddings to form a composite token embedding.
      6. Optionally pads/truncates the sequence to pad_length.
      
    Returns a tensor of shape (seq_length, total_embed_dim).
    """

    tokens = seq.strip().split()
    embeddings = []
    
    for token in tokens:
        # Parse the token into its components.
        base, sugar_type, sugar_mod, backbone = categorize_component(token)
        
        # Look up indices from vocabularies. Default to 'UNK' index if not found.
        base_idx = BASE_VOCAB.get(base, BASE_VOCAB.get("UNK", 0))
        sugar_idx = SUGAR_VOCAB.get(sugar_type, SUGAR_VOCAB.get("UNK", 0))
        sugar_mod_idx = SUGAR_MOD_VOCAB.get(sugar_mod, SUGAR_MOD_VOCAB.get("UNK", 0))
        backbone_idx = BACKBONE_VOCAB.get(backbone, BACKBONE_VOCAB.get("UNK", 0))
        
        # Look up the embeddings. (Each embedding layer returns a vector of shape (embed_dim,))
        base_emb = base_embedding(torch.tensor(base_idx, dtype=torch.long))
        sugar_emb = sugar_embedding(torch.tensor(sugar_idx, dtype=torch.long))
        sugar_mod_emb = sugar_mod_embedding(torch.tensor(sugar_mod_idx, dtype=torch.long))
        backbone_emb = backbone_embedding(torch.tensor(backbone_idx, dtype=torch.long))
        
        # Concatenate along the feature dimension.
        token_emb = torch.cat([base_emb, sugar_emb, sugar_mod_emb, backbone_emb], dim=-1)
        embeddings.append(token_emb.unsqueeze(0))  # shape (1, total_embed_dim)
    
    # Optionally pad/truncate the sequence to a fixed length.
    if pad_length is not None:
        current_len = len(embeddings)
        total_dim = embeddings[0].size(-1)
        if current_len < pad_length:
            pad_emb = torch.zeros(1, total_dim)
            embeddings.extend([pad_emb] * (pad_length - current_len))
        else:
            embeddings = embeddings[:pad_length]
    
    # Stack the token embeddings into a sequence tensor.
    sequence_embedding = torch.cat(embeddings, dim=0)  # shape (seq_length, total_embed_dim)
    return sequence_embedding
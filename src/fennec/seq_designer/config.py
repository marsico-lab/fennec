"""
Configuration file for siRNA Design Pipeline

This file contains default parameters and can be customized for different design scenarios.
"""

# Default configuration for siRNA design
DEFAULT_CONFIG = {
    # Sequence generation parameters
    "kmer_sizes": [20, 21, 22, 23, 24],
    "passenger_shifts": [0, 1, 2],
    "gc_min": 0.30,
    "gc_max": 0.62,
    
    # Chemical modification parameters
    "chem_dim": 16,
    "seq_len": 25,
    "pattern_n_min": 30,
    
    # Model parameters
    "model_type": "TCN_FM_multi_head",
    "device": "cuda",
    "batch_size": 128,
    
    # Feature parameters
    "no_passenger": False,
    "with_chem_ablate": False,
    "with_thermo": True,
    "fm_type": "rna_fm",
    "slice_length": 25,
    "only_cls_eos": False,
    
    # Target gene metadata
    "gene_target_species": "Homo sapiens",
    "duration_after_transfection_h": 24,
    "transfection_method": "Lipofectamine",
    "cell_line_donor": "Unknown_cell_line",
    "siRNA_concentration": 10,
    "mRNA_remaining_pct": 50,
    
    # Selection criteria
    "std_cutoff": 12.0,
    "confidence_multiplier": 1.96,
    "strong_knockdown_n": 60,
    "weak_knockdown_n": 25,
    
    # Quality filters
    "exclude_L96": True,
    "exclude_dna": True,
}

# High stringency configuration (more selective)
HIGH_STRINGENCY_CONFIG = DEFAULT_CONFIG.copy()
HIGH_STRINGENCY_CONFIG.update({
    "gc_min": 0.35,
    "gc_max": 0.55,
    "std_cutoff": 8.0,
    "pattern_n_min": 50,
    "strong_knockdown_n": 30,
    "weak_knockdown_n": 10,
})

# Low stringency configuration (more permissive)
LOW_STRINGENCY_CONFIG = DEFAULT_CONFIG.copy()
LOW_STRINGENCY_CONFIG.update({
    "gc_min": 0.25,
    "gc_max": 0.70,
    "std_cutoff": 15.0,
    "pattern_n_min": 20,
    "strong_knockdown_n": 100,
    "weak_knockdown_n": 50,
})

# Fast configuration (fewer kmer sizes for quick testing)
FAST_CONFIG = DEFAULT_CONFIG.copy()
FAST_CONFIG.update({
    "kmer_sizes": [21, 23],
    "passenger_shifts": [0, 1],
    "strong_knockdown_n": 20,
    "weak_knockdown_n": 10,
})

# CPU-only configuration
CPU_CONFIG = DEFAULT_CONFIG.copy()
CPU_CONFIG.update({
    "device": "cpu",
    "batch_size": 32,
})

# Configuration for different cell lines
CELL_LINE_CONFIGS = {
    "HEK293": {
        "cell_line_donor": "HEK293",
        "transfection_method": "Lipofectamine",
        "siRNA_concentration": 25,
    },
    "HeLa": {
        "cell_line_donor": "HeLa",
        "transfection_method": "Lipofectamine 2000",
        "siRNA_concentration": 50,
    },
    "A549": {
        "cell_line_donor": "A549",
        "transfection_method": "Lipofectamine RNAiMAX",
        "siRNA_concentration": 20,
    },
    "MCF7": {
        "cell_line_donor": "MCF-7",
        "transfection_method": "Lipofectamine 3000",
        "siRNA_concentration": 30,
    }
}

# Species-specific configurations
SPECIES_CONFIGS = {
    "human": {
        "gene_target_species": "Homo sapiens",
    },
    "mouse": {
        "gene_target_species": "Mus musculus",
    },
    "rat": {
        "gene_target_species": "Rattus norvegicus",
    }
}

def get_config(config_name: str = "default") -> dict:
    """
    Get a configuration dictionary by name.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        Configuration dictionary
    """
    configs = {
        "default": DEFAULT_CONFIG,
        "high_stringency": HIGH_STRINGENCY_CONFIG,
        "low_stringency": LOW_STRINGENCY_CONFIG,
        "fast": FAST_CONFIG,
        "cpu": CPU_CONFIG,
    }
    
    if config_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    return configs[config_name].copy()

def customize_config(base_config: str = "default", **kwargs) -> dict:
    """
    Create a customized configuration starting from a base configuration.
    
    Args:
        base_config: Name of base configuration
        **kwargs: Parameters to override
        
    Returns:
        Customized configuration dictionary
    """
    config = get_config(base_config)
    config.update(kwargs)
    return config

def get_cell_line_config(cell_line: str) -> dict:
    """Get configuration parameters for a specific cell line."""
    if cell_line not in CELL_LINE_CONFIGS:
        available = ", ".join(CELL_LINE_CONFIGS.keys())
        raise ValueError(f"Unknown cell line '{cell_line}'. Available: {available}")
    return CELL_LINE_CONFIGS[cell_line].copy()

def get_species_config(species: str) -> dict:
    """Get configuration parameters for a specific species."""
    if species not in SPECIES_CONFIGS:
        available = ", ".join(SPECIES_CONFIGS.keys())
        raise ValueError(f"Unknown species '{species}'. Available: {available}")
    return SPECIES_CONFIGS[species].copy()

# Example usage configurations for common scenarios
EXAMPLE_CONFIGS = {
    "ahsa1_human_hek293": customize_config(
        "default",
        **get_cell_line_config("HEK293"),
        **get_species_config("human")
    ),
    
    "fast_screening": customize_config(
        "fast",
        strong_knockdown_n=10,
        weak_knockdown_n=5
    ),
    
    "high_quality": customize_config(
        "high_stringency",
        **get_cell_line_config("HEK293")
    ),
    
    "cpu_testing": customize_config(
        "cpu",
        kmer_sizes=[21],
        passenger_shifts=[0],
        strong_knockdown_n=5,
        weak_knockdown_n=3
    )
}

if __name__ == "__main__":
    # Print available configurations
    print("Available configurations:")
    for name in ["default", "high_stringency", "low_stringency", "fast", "cpu"]:
        config = get_config(name)
        print(f"\n{name.upper()}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print("\nAvailable cell lines:", ", ".join(CELL_LINE_CONFIGS.keys()))
    print("Available species:", ", ".join(SPECIES_CONFIGS.keys()))
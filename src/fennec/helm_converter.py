"""
HELM notation converter for RNA sequences.

Converts custom RNA notation to HELM format for guide and passenger strands.
Uses component categorization to parse components and build HELM notation.
"""

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)


HELM_VERSION_SUFFIX = "$$$$V2.0"

# Import categorize_component from featurizer
try:
    from .featurizer import categorize_component
except ImportError:
    # Fallback: simplified version for standalone use
    from functools import lru_cache
    
    RNA_BASES = frozenset(['A', 'C', 'G', 'U', 'T', 'N', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V'])
    BACKBONE_MODS = frozenset(['p', 'P', '2p', 's', 'VP', 'L96', 'N'])
    
    @lru_cache(maxsize=None)
    def categorize_component(component: str) -> Tuple[str, str, str, str]:
        """
        Categorizes the component based on string parsing.
        Returns a tuple (base, sugar_type, sugar_mod, backbone)
        
        Fallback implementation when featurizer is not available.
        """
        comp = component.strip("() ")
        if comp == "L961":
            comp = "L96"
        if comp == "PAD":
            return "PAD", "PAD", "PAD", "PAD"
        if comp in RNA_BASES:
            return comp, "r", "", "p"
        if comp == "T":
            return comp, "d", "", "p"
        if comp in BACKBONE_MODS:
            return "", "", "", comp
        
        patterns: List[Tuple[callable, callable]] = [
            (lambda c: c.endswith("hd"), lambda c: (str(c[0]), "r", "hd", "p")),
            (lambda c: c.endswith("2p"), lambda c: (str(c[0]), "r", "", "2p")),
            (lambda c: c.endswith("gn"), lambda c: (str(c[0]), "gn", "", "p")),
            (lambda c: c.endswith("x"), lambda c: (str(c[0]), "x", "", "p")),
            (lambda c: c.endswith("f"), lambda c: (str(c[0]), "r", "f", "p")),
            (lambda c: c.endswith("s"), lambda c: (str(c[0]), "r", "s", "p")),
            (lambda c: c.startswith('+'), lambda c: (str(c[1]), "r", "+", "p")),
            (lambda c: c.startswith('-'), lambda c: (str(c[1]), "r", "-", "p")),
            (lambda c: c.startswith("d"), lambda c: (str(c[1]), "d", "", "p")),
            (lambda c: c in {"a", "c", "g", "u", "t"}, lambda c: (str(c.upper()), "r", "2OMe", "p")),
            # Lowercase i is 2'-OMe inosine; approximate as guanosine channel in one-hot features.
            (lambda c: c == "i", lambda c: ("G", "r", "2OMe", "p")),
        ]
        
        for condition, result_func in patterns:
            if condition(comp):
                return result_func(comp)
        
        raise ValueError(f"Unacceptable term {component} used in input sequence, has no relevant encoding")



# HELM sugar type mappings
SUGAR_TYPE_TO_HELM = {
    "r": "",      # Regular ribose (default, often not shown or shown as [mR])
    "d": "[dR]",  # Deoxyribose
    "gn": "[GN]", # Glycol nucleic acid
    "x": "[HEX]", # Hexitol
}

# HELM sugar modification mappings
SUGAR_MOD_TO_HELM = {
    "": "",       # No modification
    "2OMe": "[mR]",  # 2'-O-methyl
    "f": "[fR]",     # 2'-fluoro
    "s": "[sR]",     # 4-thioribose
    "hd": "[HD]",    # Hexadecyl
    "+": "[LR]",     # Locked nucleic acid (LNA)
    "-": "[UNA]",    # Unlocked nucleic acid
    "da": "[daR]",   # Docosyloxy
    "am": "[amR]",   # N-methylacetamide
    "vPm": "[vPm]",  # 5'-Vinylphosphonate-2'-O-methyl
    "vPf": "[vPf]",  # 5'-Vinylphosphonate-2'-fluoro
}

# HELM backbone mappings  
BACKBONE_TO_HELM = {
    "p": "P",      # Normal phosphate
    "P": "P",      # Normal phosphate
    "s": "[sP]",   # Phosphorothioate
    "2p": "[2P]",  # 2'-phosphate
    "VP": "[VP]",  # Vinyl phosphonate
    "L96": "[L96]", # L96 linker
    "L10": "[L10]", # Cholesterol linker
    "N": "",       # No backbone (end of chain)
}

SPECIAL_BASE_TO_HELM = {
    "Y34": "[Y34]",
    "Y44": "[Y44]",
}

HELM_SPECIAL_TO_BASE = {value: key for key, value in SPECIAL_BASE_TO_HELM.items()}


def component_to_helm(base: str, sugar_type: str, sugar_mod: str, backbone: str) -> str:
    """
    Convert component parts to HELM notation.
    
    Args:
        base: Nucleotide base (A, C, G, U, T, etc.)
        sugar_type: Sugar type (r, d, gn, x)
        sugar_mod: Sugar modification (f, 2OMe, s, hd, +, -, etc.)
        backbone: Backbone modification (p, s, VP, L96, etc.)
        
    Returns:
        HELM notation string for this component
    """
    # Handle special case: standalone backbone modifier at start
    if not base and backbone in BACKBONE_TO_HELM:
        return BACKBONE_TO_HELM[backbone]

    # Some featurizer-supported components are best represented as custom HELM monomers.
    if base in SPECIAL_BASE_TO_HELM:
        helm = SPECIAL_BASE_TO_HELM[base]
        if backbone in BACKBONE_TO_HELM:
            helm += BACKBONE_TO_HELM[backbone]
        elif backbone not in ["", "N"]:
            helm += "P"
        return helm
    
    # Build the HELM notation
    helm = ""
    
    # Add sugar modification prefix (like [mR], [fR], [LR])
    if sugar_mod in SUGAR_MOD_TO_HELM and SUGAR_MOD_TO_HELM[sugar_mod]:
        helm += SUGAR_MOD_TO_HELM[sugar_mod]
    # Otherwise check sugar type
    elif sugar_type in SUGAR_TYPE_TO_HELM and SUGAR_TYPE_TO_HELM[sugar_type]:
        helm += SUGAR_TYPE_TO_HELM[sugar_type]
    elif sugar_mod == "" and sugar_type == "r":
        # Unmodified RNA (uppercase bases) - use plain R
        helm = "R"
    else:
        # Default to methylated ribose for 2'OMe (lowercase bases)
        helm += "[mR]"
    
    # Add base
    if base:
        # Handle special bases
        if base == "C" and "meC" in helm:
            helm += "([5meC])"
        else:
            helm += f"({base})"
    
    # Add backbone
    if backbone in BACKBONE_TO_HELM:
        helm += BACKBONE_TO_HELM[backbone]
    else:
        # Default phosphate rules:
        # If backbone is specifically empty string or N, no phosphate (3' end usually)
        if backbone in ["", "N"]:
            helm += ""
        else:
            helm += "P"  # Default phosphate
    
    return helm


def tokenize_sequence(sequence: str) -> List[str]:
    """
    Tokenize a sequence string into individual components.
    Handles space-separated format.
    
    Args:
        sequence: Input sequence string (e.g., "VP Uf s Uf s c Cf")
        
    Returns:
        List of tokenized components
    """
    return sequence.split()


def sequence_to_helm_strand(sequence: str) -> str:
    """
    Convert a sequence to a HELM notation strand.
    
    Uses categorize_component to parse each token and builds HELM notation
    by looking ahead for backbone modifiers.
    
    Args:
        sequence: Input sequence (space-separated tokens)
        
    Returns:
        HELM notation strand (components joined with dots)
    """
    tokens = tokenize_sequence(sequence)
    helm_components = []
    
    five_prime_vp = False
    
    i = 0
    while i < len(tokens):
        token = tokens[i].strip()
        
        if not token:
            i += 1
            continue
        
        try:
            base, sugar_type, sugar_mod, backbone = categorize_component(token)
            
            # Special handling for 5' VP
            if i == 0 and not base and backbone == "VP":
                five_prime_vp = True
                i += 1
                continue
                
            # Apply 5' VP modification if pending
            if five_prime_vp:
                if sugar_mod == "2OMe":
                    sugar_mod = "vPm"
                elif sugar_mod == "f":
                    sugar_mod = "vPf"
                else:
                    # Preserve a leading VP even when the first nucleotide does not have
                    # a dedicated HELM monomer such as [vPm] or [vPf].
                    helm_components.append(BACKBONE_TO_HELM["VP"])
                five_prime_vp = False
            
            # Look ahead for explicit backbone modifier (like 's' for phosphorothioate)
            if i + 1 < len(tokens):
                next_token = tokens[i + 1].strip()
                # Check if next token is a backbone modifier
                try:
                    next_base, next_sugar_type, next_sugar_mod, next_backbone = categorize_component(next_token)
                    # If next component has no base, it's likely a backbone modifier
                    if not next_base and next_backbone in BACKBONE_TO_HELM:
                        backbone = next_backbone
                        i += 1  # Skip the next token since we consumed it
                except (ValueError, KeyError):
                    pass
            else:
                # We are at the last token (3' end)
                # If the backbone is just default "p" (phosphate), we should probably make it empty/OH
                # unless explicitly specified otherwise.
                # In standard HELM for RNA oligos, the 3' end usually has no phosphate "P" suffix 
                # unless it's a cyclic phosphate or similar.
                if backbone == "p":
                    backbone = ""  # Remove default phosphate at 3' end
            
            # Convert to HELM
            helm = component_to_helm(base, sugar_type, sugar_mod, backbone)
            
            # Skip empty components (like standalone backbones, unless it's the start)
            # Allow 5' linkers like L96, but VP should have been consumed if it was 5'
            is_standalone_backbone = helm in ["P", "P", "[sP]", "[VP]", "[2P]"]
            
            if helm and (not is_standalone_backbone or i == 0):
                helm_components.append(helm)
            
        except (ValueError, KeyError) as e:
            logger.warning(f"Error processing token '{token}': {e}")
        
        i += 1
    
    return ".".join(helm_components)


def create_helm_notation(guide: str, passenger: str) -> str:
    """
    Create HELM notation from guide and passenger sequences.
    
    Args:
        guide: Guide strand sequence (RNA1)
        passenger: Passenger strand sequence (RNA2)
        
    Returns:
        HELM notation string with empty connection/group/annotation sections
        and a HELM version suffix.
        
    Example:
        >>> guide = "VP Uf s Uf s c Cf u Uf a Uf c Cf g Gf a Af a Af c Gf a s g s g"
        >>> passenger = "u s c s Gf u Uf u Uf c Cf g Gf a Af u Af a Af g Gf s a s Af"
        >>> helm = create_helm_notation(guide, passenger)
    """
    guide_helm = sequence_to_helm_strand(guide)
    passenger_helm = sequence_to_helm_strand(passenger)
    
    helm_notation = f"RNA1{{{guide_helm}}}|RNA2{{{passenger_helm}}}{HELM_VERSION_SUFFIX}"
    
    return helm_notation


def create_helm_from_list(guide_list: List[str], passenger_list: List[str]) -> str:
    """
    Create HELM notation from lists of components.
    
    Args:
        guide_list: List of guide strand components
        passenger_list: List of passenger strand components

    Returns:
        HELM notation string
    """
    guide = " ".join(guide_list)
    passenger = " ".join(passenger_list)
    return create_helm_notation(guide, passenger)


def batch_convert_to_helm(sequences: List[Tuple[str, str]]) -> List[str]:
    """
    Convert multiple guide/passenger pairs to HELM notation.
    
    Args:
        sequences: List of (guide, passenger) tuples
        
    Returns:
        List of HELM notation strings
    """
    results = []
    for i, (guide, passenger) in enumerate(sequences):
        try:
            helm = create_helm_notation(guide, passenger)
            results.append(helm)
        except Exception as e:
            logger.error(f"Error converting sequence pair {i}: {e}")
            results.append(None)
    
    return results


# ============================================================================
# HELM to Fennec-Format Conversion (Reverse Direction)
# ============================================================================

def parse_helm_component(helm_comp: str) -> Tuple[str, str, str, str]:
    """
    Parse a single HELM component back to (base, sugar_type, sugar_mod, backbone).
    
    Args:
        helm_comp: HELM component like "[fR](U)[sP]" or "[mR](C)P"
        
    Returns:
        Tuple of (base, sugar_type, sugar_mod, backbone)
        
    Example:
        >>> parse_helm_component("[fR](U)[sP]")
        ('U', 'r', 'f', 's')
        >>> parse_helm_component("[mR](C)P")
        ('C', 'r', '2OMe', 'p')
    """
    # Default values
    base = ""
    sugar_type = "r"
    sugar_mod = ""
    backbone = "p"

    for helm_token, special_base in HELM_SPECIAL_TO_BASE.items():
        if helm_token in helm_comp:
            base = special_base
            break
    
    # Extract base - look for (X) pattern
    base_match = re.search(r'\(([A-Z0-9]+|5meC|T)\)', helm_comp)
    if base_match and not base:
        base = base_match.group(1)
        if base == "5meC":
            base = "C"  # Will be handled by sugar mod
    
    # Extract sugar modifications and types
    if "[fR]" in helm_comp:
        sugar_mod = "f"
    elif "[mR]" in helm_comp:
        sugar_mod = "2OMe"
    elif "[dR]" in helm_comp:
        sugar_type = "d"
        sugar_mod = ""
    elif "[LR]" in helm_comp:
        sugar_mod = "+"
    elif "[UNA]" in helm_comp:
        sugar_mod = "-"
    elif "[GN]" in helm_comp:
        sugar_type = "gn"
    elif "[HEX]" in helm_comp:
        sugar_type = "x"
    elif "[HD]" in helm_comp:
        sugar_mod = "hd"
    elif "[daR]" in helm_comp:
        sugar_mod = "da"
    elif "[amR]" in helm_comp:
        sugar_mod = "am"
    elif "[sR]" in helm_comp:
        sugar_mod = "s"  # 4-thioribose
    elif "[vPm]" in helm_comp:
        sugar_mod = "vPm"
    elif "[vPf]" in helm_comp:
        sugar_mod = "vPf"
    elif "[MOE]" in helm_comp:
        sugar_mod = "MOE"
    elif helm_comp.startswith("R(") and "[" not in helm_comp:
        # Plain R(X)P format - unmodified RNA (uppercase)
        sugar_mod = ""
        sugar_type = "r"
    
    # Extract backbone
    if "[sP]" in helm_comp:
        backbone = "s"
    elif "[VP]" in helm_comp:
        backbone = "VP"
    elif "[L96]" in helm_comp:
        backbone = "L96"
    elif "[L10]" in helm_comp:
        backbone = "L10"
    elif "[2P]" in helm_comp:
        backbone = "2p"
    elif "P" in helm_comp:
        backbone = "p"
    
    return base, sugar_type, sugar_mod, backbone


def helm_component_to_fennec(base: str, sugar_type: str, sugar_mod: str, backbone: str) -> str:
    """
    Convert parsed HELM component parts to Fennec-format notation.
    
    Args:
        base: Nucleotide base (A, C, G, U, T)
        sugar_type: Sugar type (r, d, gn, x)
        sugar_mod: Sugar modification (f, 2OMe, +, -, hd, s)
        backbone: Backbone type (p, s, VP, L96, 2p)
        
    Returns:
        Fennec-format component string
    """
    if not base:
        # Standalone backbone modifier
        return backbone if backbone != "p" else ""
    
    # Build the component
    comp = ""
    
    # Handle sugar modifications that are prefixes
    if sugar_mod == "+":
        comp = "+" + base
    elif sugar_mod == "-":
        comp = "-" + base
    elif sugar_mod == "vPm":
        comp = "VP " + base.lower() # VP u
    elif sugar_mod == "vPf":
        comp = "VP " + base + "f"   # VP Uf
    elif sugar_type == "d":
        comp = "d" + base
    else:
        comp = base
    
    # Handle sugar modifications that are suffixes
    if sugar_mod == "f":
        comp += "f"
    elif sugar_mod == "2OMe":
        comp = comp.lower()  # 2'-OMe uses lowercase
    elif sugar_mod == "hd":
        comp += "hd"
    elif sugar_mod == "da":
        comp += "da"
    elif sugar_mod == "am":
        comp += "am"
    elif sugar_mod == "s":
        comp += "s"
    elif sugar_type == "gn":
        comp += "gn"
    elif sugar_type == "x":
        comp += "x"

    return comp


def helm_strand_to_fennec(helm_strand: str) -> str:
    """
    Convert a HELM strand notation to Fennec-format.
    
    Args:
        helm_strand: HELM strand like "[fR](U)[sP].[fR](U)[sP].[mR](C)P"
        
    Returns:
        Fennec-format string with space-separated components
        
    Example:
        >>> helm_strand_to_fennec("[fR](U)[sP].[fR](U)[sP].[mR](C)P")
        'Uf s Uf s c'
    """
    # Split by dots to get individual components
    components = helm_strand.split(".")
    
    fennec_components = []
    
    for helm_comp in components:
        helm_comp = helm_comp.strip()
        if not helm_comp:
            continue
        
        # Parse the HELM component
        base, sugar_type, sugar_mod, backbone = parse_helm_component(helm_comp)
        
        # Convert to Fennec format
        fennec_comp = helm_component_to_fennec(base, sugar_type, sugar_mod, backbone)

        # Standalone backbone modifiers such as a leading [VP] should be emitted once
        # and preserve their original ordering.
        if not base:
            if fennec_comp:
                fennec_components.append(fennec_comp)
            continue
        
        if fennec_comp:
            fennec_components.append(fennec_comp)
        
        # Add separate backbone modifier if it's not the default.
        if backbone not in ["p", ""]:
            fennec_components.append(backbone)
    
    return " ".join(fennec_components)


def helm_to_fennec(helm_notation: str) -> Tuple[str, str]:
    """
    Convert full HELM notation back to Fennec-format (guide, passenger).
    
    Args:
        helm_notation: Full HELM string like "RNA1{...}|RNA2{...}$..." 
                      (base pairing info after $ is ignored)
        
    Returns:
        Tuple of (guide_fennec, passenger_fennec) in Fennec-format
        
    Example:
        >>> helm = "RNA1{[fR](U)[sP].[mR](C)P}|RNA2{[mR](U)[sP].[fR](G)P}"
        >>> guide, passenger = helm_to_fennec(helm)
        >>> print(guide)
        'Uf s c'
        >>> print(passenger)
        'u s Gf'
    """
    # Remove everything after the first $ (base pairing information)
    if "$" in helm_notation:
        helm_notation = helm_notation.split("$")[0]
    
    # Extract RNA1 and RNA2 using regex or careful parsing
    # Look for RNA1{...} and RNA2{...}
    import re
    
    # Match RNA1{...} and RNA2{...}
    rna1_pattern = r'RNA1\{([^}]+)\}'
    rna2_pattern = r'RNA2\{([^}]+)\}'
    
    rna1_match = re.search(rna1_pattern, helm_notation)
    rna2_match = re.search(rna2_pattern, helm_notation)
    
    if not rna1_match or not rna2_match:
        raise ValueError("Invalid HELM notation: could not find RNA1{...} or RNA2{...}")
    
    rna1_content = rna1_match.group(1)
    rna2_content = rna2_match.group(1)
    
    # Convert each strand
    guide_fennec = helm_strand_to_fennec(rna1_content)
    passenger_fennec = helm_strand_to_fennec(rna2_content)
    
    return guide_fennec, passenger_fennec


def batch_convert_from_helm(helm_notations: List[str]) -> List[Tuple[str, str]]:
    """
    Convert multiple HELM notations to Fennec-format.
    
    Args:
        helm_notations: List of HELM notation strings
        
    Returns:
        List of (guide, passenger) tuples in Fennec-format
    """
    results = []
    for i, helm in enumerate(helm_notations):
        try:
            guide, passenger = helm_to_fennec(helm)
            results.append((guide, passenger))
        except Exception as e:
            logger.error(f"Error converting HELM notation {i}: {e}")
            results.append((None, None))
    
    return results

"""Utilities for parsing and detecting modified nucleotide sequences.

This module provides regex patterns and helper functions for tokenizing
and detecting modified oligonucleotide sequences with various chemical
modifications (e.g., 2'-O-methyl, MOE, GNA, etc.).
"""

import re


PATTERN_SEQ_SPLITTER = re.compile(
    r'''
    (?:                                # non-capturing outer group
       \(A2p\) | \(Agn\) | \(Ahd\)   # 2'-phosphate, GNA, hexadecyl paren
     | \(C2p\) | \(Cgn\) | \(Chd\)
     | \(G2p\) | \(Ggn\) | \(Ghd\)
     | \(T2p\) | \(Tgn\) | \(U2p\) | \(Uhd\)
     | \([ACGU]3mx?\)                 # 3'-O-methyl paren (A3m, G3m, ...; +x variants)
     | \([ACGU]ah\)                   # 2'-O-(6-aminohexyl) paren
     | \([AG]sn\)                     # serinol-nucleic acid paren (SNA)
    | \([AGT]am\) | \(m5Cam\)        # N-methylacetamide paren
     | \(m5Ceo\) | \(m5CeO\)         # MOE-5-methylcytidine paren
     | \(5MdC\) | \(m5dC\)           # 5-methyl-deoxycytidine paren
     | \(dT\) | \(dt\)               # paren deoxy-thymine
     | \(pshe\)                       # hydroxyethylphosphorothioate paren
     | m5Ceo                          # MOE-5-methylcytidine (no paren)
     | [ACGUT]e[Oo]                   # methoxyethyl (Aeo/AeO, Ceo, Geo, Teo, Ueo)
     | [ACGU]3mx?                     # 3'-O-methyl (no paren)
     | [ACGU]da                       # docosyl (Ada, Cda, Gda, Uda)
     | [ACGU]hd                       # hexadecyl no-paren (Ahd, Chd, Ghd, Uhd)
     | [ACGT]gn                       # GNA no-paren (Agn, Cgn, Ggn, Tgn)
    | [AGT]am | m5Cam                # N-methylacetamide (no paren)
     | L961 | L96 | L57 | L10        # lipid/conjugate (L961 before L96)
     | Y196 | Y34 | Y44 | Q363       # abasic/conjugate
     | L96                           # nucleoside-L96 conjugates
     | [ACGUT]f                       # 2'-fluoro (Af, Cf, Gf, Tf, Uf)
     | d[ACGTU]                       # deoxy (dA, dC, dG, dT, dU)
     | [ACGTU]b                       # beta-L (Ab, Cb, Gb, Tb, Ub)
     | pshe                           # hydroxyethylphosphorothioate (no paren)
     | VP | vp
     | [ACGU]\+ | [ACGU]\-
     | \([ACGUT]e[Oo]\)               # paren MOE (AeO etc.) — before s fallback
    | [ACGPUNIacgistupPS]            # single-char fallback (incl. N, I, i)
    )
    ''',
    re.VERBOSE
)


def tokenize_seq_to_list(s):
    """Tokenize a modified nucleotide sequence string into a list of tokens.
    
    Args:
        s: String representation of a modified nucleotide sequence
        
    Returns:
        List of tokens representing individual nucleotides and modifications
        
    Example:
        >>> tokenize_seq_to_list("A(C3m)GUL96")
        ['A', '(C3m)', 'G', 'U', 'L96']
    """
    tokens = PATTERN_SEQ_SPLITTER.findall(s)
    return ["L96" if token == "L961" else token for token in tokens]


PATTERN_MOD_SEQ = re.compile(
    r'''
    (?:                                # non-capturing outer group
       \(A2p\) | \(Agn\) | \(Ahd\)   # 2'-phosphate, GNA, hexadecyl paren
     | \(C2p\) | \(Cgn\) | \(Chd\)
     | \(G2p\) | \(Ggn\) | \(Ghd\)
     | \(T2p\) | \(Tgn\) | \(U2p\) | \(Uhd\)
     | \([ACGU]3mx?\)                 # 3'-O-methyl paren (A3m, G3m, ...; +x variants)
     | \([ACGU]ah\)                   # 2'-O-(6-aminohexyl) paren
     | \([AG]sn\)                     # serinol-nucleic acid paren (SNA)
    | \([AGT]am\) | \(m5Cam\)        # N-methylacetamide paren
     | \(m5Ceo\) | \(m5CeO\)         # MOE-5-methylcytidine paren
     | \(5MdC\) | \(m5dC\)           # 5-methyl-deoxycytidine paren
     | \(dT\) | \(dt\)               # paren deoxy-thymine
     | \(pshe\)                       # hydroxyethylphosphorothioate paren
     | m5Ceo                          # MOE-5-methylcytidine (no paren)
     | [ACGUT]e[Oo]                   # methoxyethyl (Aeo/AeO, Ceo, Geo, Teo, Ueo)
     | [ACGU]3mx?                     # 3'-O-methyl (no paren)
     | [ACGU]da                       # docosyl (Ada, Cda, Gda, Uda)
     | [ACGU]hd                       # hexadecyl no-paren (Ahd, Chd, Ghd, Uhd)
     | [ACGT]gn                       # GNA no-paren (Agn, Cgn, Ggn, Tgn)
    | [AGT]am | m5Cam                # N-methylacetamide (no paren)
     | L961 | L96 | L57 | L10        # lipid/conjugate (L961 before L96)
     | Y196 | Y34 | Y44 | Q363       # abasic/conjugate
     | [acgu]L96                      # nucleoside-L96 conjugates
     | [ACGUT]f                       # 2'-fluoro (Af, Cf, Gf, Tf, Uf)
     | d[ACGTU]                       # deoxy (dA, dC, dG, dT, dU)
     | [ACGTU]b                       # beta-L (Ab, Cb, Gb, Tb, Ub)
     | pshe                           # hydroxyethylphosphorothioate (no paren)
     | VP | vp
     | [ACGU]\+ | [ACGU]\-
     | \([ACGUT]e[Oo]\)               # paren MOE (AeO etc.) — before s fallback
    | [IacgistupS]                   # single-char fallback (incl. I, i)
    ){5,}
    ''',
    re.VERBOSE
)


def has_mod_seq(s):
    """Check if a string contains a modified nucleotide sequence.
    
    Detects sequences with 5 or more nucleotides that may include various
    chemical modifications like 2'-O-methyl, MOE, GNA, lipid conjugates, etc.
    
    Args:
        s: String to check for presence of modified sequence
        
    Returns:
        bool: True if a modified sequence pattern is found, False otherwise
        
    Example:
        >>> has_mod_seq("AD-12345: A(C3m)GUUACG")
        True
        >>> has_mod_seq("No sequence here")
        False
    """
    return bool(PATTERN_MOD_SEQ.search(str(s)))

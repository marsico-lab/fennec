# HELM Notation Converter

This module converts **bidirectionally** between **Fennec-format** (your custom RNA notation) and **HELM** (Hierarchical Editing Language for Macromolecules) format for siRNA sequences with guide and passenger strands.

## Features

- **Bidirectional conversion**: Fennec-format ↔ HELM notation
- **Perfect round-trip**: Convert to HELM and back without loss
- Handles various chemical modifications:
  - 2'-fluoro (f)
  - 2'-O-methyl (lowercase letters)
  - Phosphorothioate linkages (s)
  - Vinyl phosphonate (VP)
  - Locked nucleic acids (LNA, +)
  - Unlocked nucleic acids (UNA, -)
  - Glycol nucleic acids (gn)
  - Hexitol modifications (x)
  - Hexadecyl modifications (hd)
  - 2'-phosphate modifications (2p)
  - DNA (d prefix)
  - And more...

## Usage

### Fennec-format → HELM

```python
from fennec.helm_converter import create_helm_notation

# Your Fennec-format with separate tokens
guide = "Uf s Uf s c Cf u Uf"
passenger = "u s c s Gf u Uf Af"

# Convert to HELM
helm = create_helm_notation(guide, passenger)
print(helm)
# Output: RNA1{[fR](U)[sP].[fR](U)[sP].[mR](C)P...}|RNA2{...}
```

### HELM → Fennec-format

```python
from fennec.helm_converter import helm_to_fennec

helm = "RNA1{[fR](U)[sP].[mR](C)P}|RNA2{[mR](U)[sP].[fR](G)P}"

# Convert back to Fennec-format
guide, passenger = helm_to_fennec(helm)
print(f"Guide:     {guide}")
print(f"Passenger: {passenger}")
# Output:
# Guide:     Uf s c
# Passenger: u s Gf
```

### Round-Trip Conversion ✓

```python
from fennec.helm_converter import create_helm_notation, helm_to_fennec

# Original Fennec-format
original_guide = "Uf s c Cf u"
original_passenger = "a g Gf"

# To HELM
helm = create_helm_notation(original_guide, original_passenger)

# Back to Fennec-format
restored_guide, restored_passenger = helm_to_fennec(helm)

assert original_guide == restored_guide  # ✓ Perfect match!
assert original_passenger == restored_passenger  # ✓ Perfect match!
```

## Input Format (Fennec-format)

Fennec-format uses space-separated tokens where:

- **Bases**: Single uppercase letters (A, C, G, U, T) - unmodified RNA
- **2'-O-Methyl**: Lowercase letters (a, c, g, u, t)
- **2'-Fluoro**: Base + 'f' (Af, Cf, Gf, Uf)
- **Phosphorothioate**: Standalone 's' token
- **Vinyl Phosphonate**: Standalone 'VP' token
- **Locked NA**: '+' + base (+A, +C, +G, +U)
- **Unlocked NA**: '-' + base (-A, -C, -G, -U)
- **DNA**: 'd' + base (dA, dC, dG, dT)
- **Glycol NA**: Base + 'gn' (Agn, Cgn, Ggn, Ugn)
- **Hexitol**: Base + 'x' (Ax, Cx, Gx, Ux)
- **Hexadecyl**: Base + 'hd' (Ahd, Chd, Ghd, Uhd)
- **2'-Phosphate**: (Base2p) or Base2p format

## All Functions

### Fennec-format → HELM

- **`create_helm_notation(guide, passenger)`** - Main conversion from strings
- **`create_helm_from_list(guide_list, passenger_list)`** - From lists of tokens
- **`batch_convert_to_helm(sequences)`** - Batch process multiple sequences

### HELM → Fennec-format

- **`helm_to_fennec(helm_notation)`** - Main reverse conversion
- **`helm_strand_to_fennec(helm_strand)`** - Convert single strand
- **`batch_convert_from_helm(helm_notations)`** - Batch reverse conversion

### Component-level Functions

- **`parse_helm_component(helm_comp)`** - Parse HELM component to parts
- **`helm_component_to_fennec(base, sugar_type, sugar_mod, backbone)`** - Build Fennec component

## How It Works

The converter:

1. **Tokenizes** the input sequence (splits by spaces)
2. **Categorizes** each component into:
   - Base (A, C, G, U, T)
   - Sugar type (ribose, deoxyribose, glycol, hexitol)
   - Sugar modification (fluoro, 2'-OMe, locked, etc.)
   - Backbone modification (phosphate, phosphorothioate, VP, etc.)
3. **Looks ahead** for separate backbone modifiers (like standalone 's')
4. **Builds HELM notation** by combining components with appropriate syntax

## HELM Notation Format

The output HELM notation follows this pattern:
- `RNA1{...}` - Guide strand
- `RNA2{...}` - Passenger strand
- Components separated by dots (.)
- Sugar modifications in square brackets: `[fR]`, `[mR]`, `[LR]`, etc.
- Bases in parentheses: `(A)`, `(C)`, `(G)`, `(U)`
- Backbone modifiers: `P` (phosphate), `[sP]` (phosphorothioate), `[VP]` (vinyl phosphonate)

Example component: `[fR](U)[sP]`
- `[fR]` = 2'-fluoro ribose
- `(U)` = Uracil base
- `[sP]` = Phosphorothioate linkage

## Supported Modifications

| Input | Meaning | HELM Output |
|-------|---------|-------------|
| `Uf` | 2'-fluoro U | `[fR](U)P` |
| `u` | 2'-OMe U | `[mR](U)P` |
| `s` | Phosphorothioate | `[sP]` |
| `VP` | Vinyl phosphonate | `[VP]` |
| `+A` | Locked A | `[LR](A)P` |
| `-A` | Unlocked A | `[UNA](A)P` |
| `Agn` | Glycol A | `[GN](A)P` |
| `Ahd` | Hexadecyl A | `[HD](A)P` |
| `dA` | Deoxy A (DNA) | `[dR](A)P` |
| `A2p` | 2'-phosphate A | `[mR](A)[2P]` |

## Notes

- The converter handles the "lookahead" logic for separate backbone modifiers (like `Uf s` becomes `[fR](U)[sP]`)
- Starting modifiers like `VP` at the beginning are skipped in the final output
- The output only includes RNA1 and RNA2 components, not the base pairing information (stops at the `$` in full HELM notation)

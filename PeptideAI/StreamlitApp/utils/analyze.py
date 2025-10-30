from collections import Counter

def aa_composition(sequence):
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    counts = Counter(sequence)
    total = len(sequence)
    return {aa: counts.get(aa, 0) / total for aa in amino_acids}

# Compute sequence properties
def compute_properties(sequence):

    # Property calculations 
    aa_weights = {'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
                  'E': 147.1, 'Q': 146.2, 'G': 75.1, 'H': 155.2, 'I': 131.2,
                  'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
                  'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1}
    mw = sum(aa_weights.get(aa, 0) for aa in sequence)
    hydrophobic = sum(1 for aa in sequence if aa in "AILMFWYV") / len(sequence)
    charge = sum(1 for aa in sequence if aa in "KRH") - sum(1 for aa in sequence if aa in "DE")
    return {"Length": len(sequence), "Molecular Weight (Da)": round(mw, 2),
            "Hydrophobic Fraction": round(hydrophobic, 3), "Net Charge (approx.)": charge}

import pandas as pd
from Bio import SeqIO
from pathlib import Path

amp_fasta = "amps.fasta"         
non_amp_fasta = "non_amps.fasta"  
output_csv = "ampData3.csv"
valid_aas = set("ACDEFGHIKLMNPQRSTVWY")  

# HELPER: clean and validate sequences
def clean_seq(seq):
    seq = seq.strip().upper()
    if not seq or any(aa not in valid_aas for aa in seq):
        return None
    return seq

# LOAD FASTAS
def load_fasta(filepath, label):
    """Load fasta file. Accepts a filename or path. If the path does not exist
    as given, try resolving it relative to this script's directory.
    Returns list of dicts: {"sequence": seq, "label": label}.
    """
    p = Path(filepath)

    if not p.exists():
        p = Path(__file__).resolve().parent / filepath
    if not p.exists():
        raise FileNotFoundError(f"Fasta file not found: '{filepath}' (tried '{p}')")

    records = []
    for record in SeqIO.parse(str(p), "fasta"):
        seq = clean_seq(str(record.seq))
        if seq:
            records.append({"sequence": seq, "label": label})
    return records

amps = load_fasta(amp_fasta, 1)
non_amps = load_fasta(non_amp_fasta, 0)

print(f"Loaded {len(amps)} AMPs and {len(non_amps)} non-AMPs before cleaning.")

# REMOVE DUPLICATES
amp_df = pd.DataFrame(amps).drop_duplicates(subset=["sequence"])
non_amp_df = pd.DataFrame(non_amps).drop_duplicates(subset=["sequence"])

# BALANCE CLASSES
min_count = min(len(amp_df), len(non_amp_df))
amp_balanced = amp_df.sample(n=min_count, random_state=42)
non_amp_balanced = non_amp_df.sample(n=min_count, random_state=42)

# COMBINE AND SHUFFLE 
final_df = pd.concat([amp_balanced, non_amp_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)

# SAVE TO CSV
final_df.to_csv(output_csv, index=False)

print(f"Saved balanced dataset with {len(final_df)} total sequences ({min_count} per class).")
print(f"Output file: {output_csv}")

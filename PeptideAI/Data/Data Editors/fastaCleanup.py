from Bio import SeqIO
import pandas as pd

# CONFIG 
input_fasta = "amps.fasta"       
output_fasta = "amps_clean.fasta" 
output_csv = "amps_clean.csv"     

min_len = 5
max_len = 100
valid_aas = set("ACDEFGHIKLMNPQRSTVWY")

# CLEAN FUNCTION 
def clean_seq(seq):
    seq = seq.strip().upper()
    if not (min_len <= len(seq) <= max_len):
        return None
    if any(aa not in valid_aas for aa in seq):
        return None
    return seq

# READ AND CLEAN
clean_records = []
for record in SeqIO.parse(input_fasta, "fasta"):
    seq = clean_seq(str(record.seq))
    if seq:
        clean_records.append(seq)

# DEDUPLICATE 
clean_records = list(set(clean_records))

# SAVE CLEAN FASTA 
with open(output_fasta, "w") as f:
    for i, seq in enumerate(clean_records, start=1):
        f.write(f">AMP_{i}\n{seq}\n")

# SAVE CSV 
pd.DataFrame({"sequence": clean_records}).to_csv(output_csv, index=False)

print(f"✅ Cleaned {len(clean_records)} sequences saved to '{output_fasta}' and '{output_csv}'.")

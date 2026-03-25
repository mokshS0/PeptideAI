# Heuristic mutation search used by the Optimize page.
import random
from utils.predict import predict_amp

# Residue groups used to propose chemistry-aware substitutions.
HYDROPHOBIC = set("AILMFWVPG")
HYDROPHILIC = set("STNQYCH")
POSITIVE = set("KRH")
NEGATIVE = set("DE")

def mutate_residue(residue):
    # Return a candidate replacement residue and rationale.
    if residue in POSITIVE:
        return residue, "Retained strong positive residue"
    elif residue in NEGATIVE:
        return random.choice(list(POSITIVE)), "Increased positive charge"
    elif residue in HYDROPHILIC:
        return random.choice(list(HYDROPHOBIC)), "Improved hydrophobicity balance"
    elif residue in HYDROPHOBIC:
        return random.choice(list(POSITIVE | HYDROPHILIC)), "Enhanced amphipathicity"
    else:
        return random.choice(list(HYDROPHOBIC)), "Adjusted physicochemical profile"

def optimize_sequence(seq, model, max_rounds=20, confidence_threshold=0.001):
    # Iteratively improve AMP probability by accepting the best mutation per round.
    current_seq = seq
    label, conf = predict_amp(current_seq, model)
    best_conf = conf
    history = [(current_seq, conf, "-", "-", "-", "Original sequence")]

    # Greedy loop: keep only the best confidence-improving mutation each round.
    for _ in range(max_rounds):
        best_mutation = None
        best_mutation_conf = best_conf

        for pos, old_res in enumerate(current_seq):
            new_res, reason = mutate_residue(old_res)
            if new_res == old_res:
                continue
            new_seq = current_seq[:pos] + new_res + current_seq[pos+1:]
            _, new_conf = predict_amp(new_seq, model)

            if new_conf > best_mutation_conf:
                best_mutation_conf = new_conf
                best_mutation = (new_seq, pos, old_res, new_res, reason)

        if best_mutation and best_mutation_conf - best_conf >= confidence_threshold:
            current_seq, pos, old_res, new_res, reason = best_mutation
            best_conf = best_mutation_conf
            change = f"Pos {pos+1}: {old_res} → {new_res}"
            history.append((current_seq, best_conf, change, old_res, new_res, reason))
        else:

            # Stop when no mutation clears the minimum improvement threshold.
            break

    return current_seq, best_conf, history

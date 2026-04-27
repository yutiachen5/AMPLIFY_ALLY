import os
import argparse
import tqdm 

from scipy.stats import spearmanr
import numpy as np
import pandas as pd

import torch

from utils import load_from_mila, load_from_hf

device = "cuda"
compile = False

def get_sequence_window(focus_seq, pos_idx, model_context_len=512):
    seq_len = len(focus_seq)
    if seq_len <= model_context_len:
        return focus_seq, pos_idx

    half = model_context_len // 2
    start = max(0, pos_idx - half)
    end = start + model_context_len

    if end > seq_len:
        end = seq_len
        start = end - model_context_len

    new_pos_idx = pos_idx - start
    return focus_seq[start:end], new_pos_idx

def get_mutation_info(mutant):
    mutations = []
    for mutation in mutant.split(":"):
        from_AA = mutation[0]
        position = int(mutation[1:-1])
        to_AA = mutation[-1]
        mutations.append((from_AA, position, to_AA))
    return mutations

def calc_fitness_masked(model, tokenizer, focus_seq, mutants, device='cuda', model_context_len=512):
    scores = []
    mask_token_id = tokenizer.mask_token_id

    with torch.no_grad():
        for mutant in mutants:
            mutations = get_mutation_info(mutant)
            score = 0.0

            for from_AA, position, to_AA in mutations:
                pos_idx = position - 1  # convert 1-based to 0-based
                assert focus_seq[pos_idx] == from_AA, \
                    f"WT mismatch at pos {position}: expected {from_AA}, got {focus_seq[pos_idx]}"

                seq_window, new_pos_idx = get_sequence_window(focus_seq, pos_idx, model_context_len)

                ids = torch.as_tensor(
                    tokenizer.encode(seq_window)
                ).to(torch.long).unsqueeze(0).to(device)

                masked_ids = ids.clone()
                masked_ids[0, new_pos_idx + 1] = mask_token_id  # +1 for BOS

                logits = model(masked_ids).logits
                log_probs = torch.log_softmax(logits[0, new_pos_idx + 1], dim=-1)

                wt_id = tokenizer.encode(from_AA, add_special_tokens=False)[0]
                mt_id = tokenizer.encode(to_AA, add_special_tokens=False)[0]

                score += (log_probs[mt_id] - log_probs[wt_id]).item()

            scores.append(score)

    return np.array(scores)

def get_mutated_sequence(focus_seq, mutant, start_idx=1, AA_vocab="ACDEFGHIKLMNPQRSTVWY"):
    mutated_seq = list(focus_seq)
    for mutation in mutant.split(":"):
        try:
            from_AA, position, to_AA = mutation[0], int(mutation[1:-1]), mutation[-1]
        except:
            print("Issue with mutant: "+str(mutation))
        relative_position = position - start_idx
        assert (from_AA==focus_seq[relative_position]), "Invalid from_AA or mutant position: "+str(mutation)+" from_AA: "+str(from_AA) + " relative pos: "+str(relative_position) + " focus_seq: "+str(focus_seq)
        assert (to_AA in AA_vocab), "Mutant to_AA is invalid: "+str(mutation)
        mutated_seq[relative_position] = to_AA
    return "".join(mutated_seq)

def main():
    parser = argparse.ArgumentParser(description='AMPLIFY_ALLY masked marginal scoring')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--DMS_reference_file_path', default='/hpc/group/naderilab/eleanor/AMPLIFY_ALLY/ProteinGym/DMS_substitutions.csv', type=str)
    parser.add_argument('--DMS_data_folder', default='/hpc/group/naderilab/eleanor/AMPLIFY_ALLY/ProteinGym/DMS_ProteinGym_substitutions', type=str)
    parser.add_argument('--DMS_index', type=int, default=0)
    parser.add_argument('--output_scores_folder', default='/hpc/group/naderilab/eleanor/AMPLIFY_ALLY/ProteinGym/output', type=str)

    args = parser.parse_args()
    model, tokenizer = load_from_mila(args.model_path, args.config_path)
    # test this script using esm 8M mdoel
    model, tokenizer = load_from_hf(
        model_path="facebook/esm2_t6_8M_UR50D",
        tokenizer_path="facebook/esm2_t6_8M_UR50D",
        fp16=False,
    )
    model.to(device)
    model = torch.compile(model, disable=not compile)

    mapping_protein_seq_DMS = pd.read_csv(args.DMS_reference_file_path)
    list_DMS = mapping_protein_seq_DMS["DMS_id"]
    DMS_id = list_DMS[args.DMS_index]
    DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]
    target_seq = mapping_protein_seq_DMS["target_seq"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0].upper()
    print("Computing scores for: {} with model: {}".format(DMS_id, args.model_path))

    DMS_data = pd.read_csv(args.DMS_data_folder + os.sep + DMS_file_name, low_memory=False)

    model_scores = calc_fitness_masked(
        model=model,
        tokenizer=tokenizer,
        focus_seq=target_seq,
        mutants=DMS_data['mutant'].values,
        device=device
    )

    DMS_data['ALLY_score'] = model_scores
    scoring_filename = args.output_scores_folder + os.sep + DMS_id + '.csv'
    DMS_data[['mutant', 'ALLY_score', 'DMS_score', 'DMS_score_bin']].to_csv(scoring_filename, index=False)

if __name__ == '__main__':
    main()
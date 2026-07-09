import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.notebook import tqdm
tqdm.pandas(leave = False)

from transformers import (
    AutoTokenizer,
    EsmTokenizer,
    EsmForMaskedLM,
    pipeline,
)

from itertools import chain
import torch
import torch.nn.functional as F
import scipy
import condensed_blosum_metric
from condensed_blosum_metric import new_scores
from condensed_blosum_metric import new_blosum62
from condensed_blosum_metric import find_all_scores

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device

def infer_and_group_stats(model, tokenizer, seq, cdr, sequence_id, donor_id):
    losses = []
    predictions = ""
    scores = []
    perplexities = []

    blosum_score = []

    with torch.no_grad():
        sep = "<cls><cls>"
        sep_idx = seq.find(sep)
        heavy = seq[:sep_idx]
        light = seq[sep_idx + len(sep):]
        # print("SEQ: ", seq) 
        # print("CDR: ", cdr)
        
        cdr_mask = str(cdr)[:sep_idx] + str(cdr)[sep_idx + 2:]

        unmasked = tokenizer(seq, return_tensors = "pt").to(device)["input_ids"]
        ranges = [range(sep_idx), range(sep_idx + len(sep), len(seq))]
        total_len = sum(len(i) for i in ranges)

        # model iteratively predicts each residue (skipping over separator tokens)
        for i in chain(*ranges):
        # for i in tqdm(chain(*ranges), total=total_len, leave=False):
            masked = seq[:i] + "<mask>" + seq[i+1:]
            tokenized = tokenizer(masked, return_tensors="pt").to(device)
            mask_pos = (tokenized.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            labels = torch.where(tokenized.input_ids == tokenizer.mask_token_id, unmasked, -100)
            output = model(**tokenized, labels = labels)
            logits = output.logits

            # predicted aa
            pred_token = logits[0, mask_pos].argmax(axis=-1)
            predictions+=tokenizer.decode(pred_token)
            #print(tokenizer.decode(pred_token))
            # print("predicted token: ", pred_token)
            pred_aa = tokenizer.decode(pred_token)
            new_score = new_blosum62[seq[i]][pred_aa]
            scores.append(new_score)

            # loss
            loss = output.loss.item()
            losses.append(loss)
            
            # perplexity
            ce_loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), labels.view(-1)) # i think this is the same as output.loss.item()
            perplexities.append(float(torch.exp(ce_loss)))

        # group stats by region
        # find indices splitting regions (fwrs and cdrs in heavy and light chains)
        cdr_idxs = [0] + [i for i in range(len(cdr_mask)) if cdr_mask[i] != cdr_mask[i-1]] + [len(cdr_mask)]
        cdr_idxs.insert(7, sep_idx)
        
        # accuracy
        predictions_by_region = [predictions[cdr_idxs[n]:cdr_idxs[n+1]] for n in range(len(cdr_idxs)-1)]
        seq_by_region = [seq.replace(sep, "")[cdr_idxs[n]:cdr_idxs[n+1]] for n in range(len(cdr_idxs)-1)]
        region_mean_acc = [sum(true[i] == predict[i] for i in range(len(true)))/len(true) for true, predict in zip(seq_by_region, predictions_by_region)]

        # prediction confidence
        region_mean_scores = [np.mean(scores[cdr_idxs[n]:cdr_idxs[n+1]]) for n in range(len(cdr_idxs)-1)]

        # loss (median)
        region_median_loss = [np.median(losses[cdr_idxs[n]:cdr_idxs[n+1]]) for n in range(len(cdr_idxs)-1)]
        
        # perplexity (median)
        region_median_perplexity = [np.median(perplexities[cdr_idxs[n]:cdr_idxs[n+1]]) for n in range(len(cdr_idxs)-1)]
        
        return {
            "sequence": seq.replace(sep, ""),
            "cdr_mask": cdr_mask,
            "heavy": heavy,
            "light": light,
            "cdr_indices": cdr_idxs,
            "prediction": predictions,
            "accuracy_by_region": region_mean_acc,
            "score_by_region": region_mean_scores,
            "loss_by_region": region_median_loss,
            "perplexity_by_region": region_median_perplexity,
            "score": scores,
            "loss": losses,
            "perplexity": perplexities,
            "id": sequence_id,
            "donor": donor_id
        }

def main():
    
    model_dict = {
    "8M-F": "brineylab/8M_full_checkpoint-500000",
    "35M-F": "brineylab/35M_full_checkpoint-500000",
    "150M-F": "brineylab/150M_full_checkpoint-500000",
    "350M-F": "brineylab/350M_full_checkpoint-500000",
    "650M-F": "brineylab/650M_full_checkpoint-39500000"
    }

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")

    germline_test_df = pd.read_csv("unmutated_donors-combined.csv", dtype=str)
    mutated_test_df = pd.read_csv("mutated_donors-combined.csv", dtype=str)
    data_dict = {
        "germline": germline_test_df, 
        "mutated": mutated_test_df
    }

    for name, model_path in model_dict.items():
        for seq_type, data in data_dict.items():
    
            model = EsmForMaskedLM.from_pretrained(model_path).to(device)
        
            inference_data = []
            sequences = list(data.iterrows())
        
            for _id, row in tqdm(sequences):
                whole_seq = row['sequence_aa_heavy'] + "<cls><cls>" + row["sequence_aa_light"]
                whole_cdr = row["cdr_mask_aa_heavy"] + "00" + row["cdr_mask_aa_light"]
                seq_id = row["sequence_id"]
                donor = row["donor"]
                print("working")
                d = infer_and_group_stats(
                    model, 
                    tokenizer,
                    #row["text"],
                    #row["cdr_mask"]
                    whole_seq, 
                    whole_cdr,
                    seq_id,
                    donor
                )
                
                inference_data.append(d)
      
            inference_df = pd.DataFrame(inference_data)
        inference_df.to_parquet(f"./results/{name}_{seq_type}_{len(inference_data)}.parquet")

if __name__ == "__main__":
    main()

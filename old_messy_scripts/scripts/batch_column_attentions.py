
""" Runs MSA Transformer to create query matrices for each PDB ID in the input string.

Args:
    pdb_ids (str): PDB IDs as a string, seperated by a '+' sign.
    msa_directory (str): The directory where the MSA (.a3m) files are stored.
    output_directory (str): The directory where the query matrix (.pt) files should be created.
"""


from typing import List, Tuple
import os
import string
import sys
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import squareform, pdist, cdist
from Bio import SeqIO
from time import sleep
import esm
torch.set_grad_enabled(False)
from einops import rearrange
import gc


# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)


def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA.
    Source: The ESM-2 notebook 'contact_prediction.ipynb'.
    """
    return sequence.translate(translation)


def read_msa(filename: str) -> List[Tuple[str, str]]:
    """ Reads the sequences from an MSA file, automatically removes insertions.
    Source: The ESM-2 notebook 'contact_prediction.ipynb'.
    """
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]


def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
    """ Select sequences from the MSA to maximize the hamming distance.
    Source: The ESM-2 notebook 'contact_prediction.ipynb'.
    """
    assert mode in ("max", "min")
    if len(msa) <= num_seqs:
        return msa

    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [msa[idx] for idx in indices]


def main(pdb_ids, msa_directory, output_directory, sequence_count, start_index):

    # Select the GPU to run on
    cuda = torch.device('cuda:1')
    torch.set_default_device('cuda:1')
    torch.cuda.set_device(1)

    msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    msa_transformer = msa_transformer.eval().cuda()
    msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()

    pdb_list = pdb_ids.lower().split('+')[start_index:]

    for pdb_id in pdb_list:
        # Make sure the PDB ID is in the right format
        if len(pdb_id) == 5:
            pdb_id = pdb_id[:4] + '_' + pdb_id[4]

        msa_path = os.path.join(msa_directory, 'aligned_' + pdb_id + '.a3m')
        #output_path = os.path.join(output_directory, 'query_col_attentions_' + pdb_id + '_' + str(sequence_count) + '.pt')
        output_path = os.path.join(output_directory, 'convnet_attentions_' + pdb_id + '_' + str(sequence_count) + '.pt')

        # Run MSA Transformer if the MSA input file exists and embeddings output file doesn't exist
        if not os.path.isfile(msa_path):
            print('ERROR: The MSA file for "' + pdb_id + '" does not exist!')
            continue
        elif os.path.isfile(output_path):
            #continue
            print('ERROR: The embeddings file for "' + pdb_id + '" already exists!')
        try:
            torch.cuda.empty_cache()
            inputs = read_msa(msa_path)

            # The script only accepts a MSA sequence count of 1000 or above to ensure some input sequence diversity
            if len(inputs) < 1000:
                #print('ERROR: The MSA file for "' + pdb_id + '" is too small! Expected: >= 1000 Actual:', len(inputs))
                continue

            inputs = greedy_select(inputs, num_seqs=sequence_count)
            msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = msa_transformer_batch_converter([inputs])
            # Free up memory
            gc.collect()
            torch.cuda.empty_cache()
            
            msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(next(msa_transformer.parameters()).device)
            
            with torch.no_grad():
                result = msa_transformer(msa_transformer_batch_tokens, repr_layers=[], need_head_weights=True, return_contacts=True)
            
            column_attentions = result['col_attentions'][0].cpu().numpy()

            query_row_heads = column_attentions[:, :, :, :, 0]
            query_col_heads = column_attentions[:, :, :, 0, :]
            # Flatten the first two dimensions into one
            query_row_heads = np.reshape(query_row_heads, (query_row_heads.shape[0] * query_row_heads.shape[1], query_row_heads.shape[2], query_row_heads.shape[3]))
            query_col_heads = np.reshape(query_col_heads, (query_col_heads.shape[0] * query_col_heads.shape[1], query_col_heads.shape[2], query_col_heads.shape[3]))
            query_heads = np.stack((query_row_heads, query_col_heads), axis=0)
            torch.save(query_heads, output_path)

            torch.cuda.empty_cache()

        except BaseException as e:
            print('ERROR: The attention maps for', pdb_id, 'were not able to be created.')
            print(e)
            torch.cuda.empty_cache()
            sleep(5)


if __name__ == "__main__":
    
    error = False

    if len(sys.argv) < 4:
        print('ERROR: Inputs should be: pdb_ids, msa_directory, output_directory, (sequence_count), (start_index)!')
        error = True

    pdb_ids = sys.argv[1]
    msa_directory = sys.argv[2]
    output_directory = sys.argv[3]
    sequence_count = int(sys.argv[4]) if len(sys.argv) >= 5 else 128
    start_index = int(sys.argv[5]) if len(sys.argv) >= 6 else 0

    print('MSA Directory:', msa_directory, '\nOutput Directory:', output_directory, '\nSequence Count:', sequence_count, '\nStart Index:', start_index)


    if not os.path.isdir(msa_directory):
        print(f'ERROR: The specified MSA directory "{msa_directory}" does not exist!')
        error = True
    if not os.path.isdir(output_directory):
        print(f'ERROR: The specified output directory "{output_directory}" does not exist!')
        error = True
        

    if not error:
        main(pdb_ids, msa_directory, output_directory, sequence_count, start_index)
        print('Done')

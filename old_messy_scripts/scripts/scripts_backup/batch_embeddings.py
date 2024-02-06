
""" Runs MSA Transformer to create embeddings for each PDB ID in the input string.

Args:
    pdb_ids (str): PDB IDs as a string, seperated by a '+' sign.
    msa_directory (str): The directory where the MSA (.a3m) files are stored.
    output_directory (str): The directory where the embedding (.pt) files should be created.
"""


from typing import List, Tuple
import os
import string
import sys
import numpy as np
import torch
from scipy.spatial.distance import squareform, pdist, cdist
from Bio import SeqIO
from time import sleep
import esm
torch.set_grad_enabled(False)


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


def main(pdb_ids, msa_directory, output_directory):

    torch.cuda.set_device(1)

    msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    msa_transformer = msa_transformer.eval().cuda()
    msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()

    for pdb_id in pdb_ids.split('+'):
        # Make sure the PDB ID is in the right format
        pdb_id = pdb_id.lower()
        if len(pdb_id) == 5:
            pdb_id = pdb_id[:4] + '_' + pdb_id[4]

        msa_path = os.path.join(msa_directory, 'aligned_' + pdb_id + '.a3m')
        output_path = os.path.join(output_directory, 'embeddings_' + pdb_id + '.pt')

        # Run MSA Transformer if the MSA input file exists and embeddings output file doesn't exist
        if not os.path.isfile(msa_path):
            print('ERROR: The MSA file for "' + pdb_id + '" does not exist!')
            continue
        elif os.path.isfile(output_path):
            continue
        try:
            inputs = read_msa(msa_path)
            # The script only accepts a MSA sequence count of 1000 or above to ensure some input sequence diversity
            if len(inputs) < 1000:
                print('ERROR: The MSA file for "' + pdb_id + '" is too small! Expected: >= 1000 Actual:', len(inputs))
                continue
            inputs = greedy_select(inputs, num_seqs=512)
            msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = msa_transformer_batch_converter([inputs])
            msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(next(msa_transformer.parameters()).device)
            with torch.no_grad():
                result = msa_transformer(msa_transformer_batch_tokens, repr_layers=[12])
            embeddings = result["representations"][12]
            torch.save(embeddings, output_path)
        except BaseException as e:
            print('ERROR: The embeddings for', pdb_id, 'were not able to be created.')
            print(e)
            torch.cuda.empty_cache()
            sleep(5)


if __name__ == "__main__":
    error = False
    if len(sys.argv) != 4:
        print('ERROR: Inputs should be: pdb_ids, msa_directory, output_directory!')
        error = True
    else:
        if not os.path.isdir(sys.argv[2]):
            print('ERROR: The specified MSA directory "' + sys.argv[2] + '" does not exist!')
            error = True
        if not os.path.isdir(sys.argv[3]):
            print('ERROR: The specified output directory "' + sys.argv[3] + '" does not exist!')
            error = True
    if not error:
        main(sys.argv[1], sys.argv[2], sys.argv[3])

import sys
import glob
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.spatial.distance import squareform, pdist, cdist
from Bio import SeqIO
from Bio.PDB import *
from Bio.Align import PairwiseAligner
import biotite.structure as bs
from biotite.structure.io.pdbx import PDBxFile, get_structure
from biotite.database import rcsb
import esm


torch.set_grad_enabled(False)

parser = PDBParser(PERMISSIVE = True, QUIET = True) 
pdbl = PDBList() 

# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)


def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)


def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)


def read_msa(filename: str) -> List[Tuple[str, str]]:
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]


def extend(a, b, c, L, A, D):
    """
    input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    """

    def normalize(x):
        return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)

    bc = normalize(b - c)
    n = normalize(np.cross(b - a, bc))
    m = [bc, np.cross(n, bc), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    return c + sum([m * d for m, d in zip(m, d)])


def contacts_from_pdb(
    structure: bs.AtomArray,
    distance_threshold: float = 8.0,
    chain: Optional[str] = None,
) -> np.ndarray:
    mask = ~structure.hetero
    if chain is not None:
        mask &= structure.chain_id == chain

    N = structure.coord[mask & (structure.atom_name == "N")]
    CA = structure.coord[mask & (structure.atom_name == "CA")]
    C = structure.coord[mask & (structure.atom_name == "C")]

    Cbeta = extend(C, N, CA, 1.522, 1.927, -2.143)
    dist = squareform(pdist(Cbeta))
    
    contacts = dist < distance_threshold
    contacts = contacts.astype(np.int64)
    contacts[np.isnan(dist)] = -1
    return contacts


# Select sequences from the MSA to maximize the hamming distance
def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
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


def get_pdb_indices(directory):
    fasta_path = glob.glob(directory + "fasta/*.seq")[0]
    reference_1_path = glob.glob(directory + "reference_1/*.pdb")[0]
    reference_2_path = glob.glob(directory + "reference_2/*.pdb")[0]

    # Read the fasta file
    with open(fasta_path, 'r') as file:
        fasta = file.read().split("\n")[1]

    pdb_sequences = []
    for reference_path in (reference_1_path, reference_2_path):
        # Get the amino acid sequence from the pdb file
        structure = parser.get_structure("Structure", reference_path)
        model = list(structure.get_models())[0]
        chains = list(model.get_chains())[0]
        residues = chains.get_residues()
        residues = [res.resname for res in residues if str(res)[17] == ' ']

        # Change the format of the amino acid sequence from 3-letter to 1-letter
        aa_dict = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLU':'E', 'GLN':'Q', 'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'}
        aa_sequence = "".join([aa_dict[res] for res in residues])
        pdb_sequences.append(aa_sequence)

    reference_1_sequence, reference_2_sequence = pdb_sequences
    
    aligner = PairwiseAligner()
    alignment_12 = aligner.align(reference_1_sequence, reference_2_sequence)[0]

    a_12 = list(alignment_12[0])
    a_21 = list(alignment_12[1])

    l_12 = []
    l_21 = []
    index_1 = 0
    index_2 = 0
    for i in range(len(a_12)):
        if a_12[i] == a_21[i]:
            l_12.append(index_1)
            l_21.append(index_2)
        if a_12[i] != '-':
            index_1 += 1
        if a_21[i] != '-':
            index_2 += 1
    seq_12 = "".join([reference_1_sequence[i] for i in l_12])

    alignment_f1 = aligner.align(fasta, seq_12)[0]
    a_f = list(alignment_f1[0])
    a_1 = list(alignment_f1[1])

    l_1 = []
    l_2 = []
    l_f = []
    index_1 = 0
    index_2 = 0
    index_f = 0
    for i in range(len(a_1)):
        if a_f[i] == a_1[i]:
            l_1.append(l_12[index_1])
            l_2.append(l_21[index_2])
            l_f.append(index_f)
        if a_1[i] != '-':
            index_1 += 1
            index_2 += 1
        if a_f[i] != '-':
            index_f += 1

    return l_1, l_2, l_f


def compute_precisions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    src_lengths: Optional[torch.Tensor] = None,
    minsep: int = 6,
    maxsep: Optional[int] = None,
    override_length: Optional[int] = None,  # for casp
):
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)
    override_length = (targets[0, 0] >= 0).sum()

    # Check sizes
    if predictions.size() != targets.size():
        raise ValueError(
            f"Size mismatch. Received predictions of size {predictions.size()}, "
            f"targets of size {targets.size()}"
        )
    device = predictions.device

    batch_size, seqlen, _ = predictions.size()
    seqlen_range = torch.arange(seqlen, device=device)

    sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
    sep = sep.unsqueeze(0)
    valid_mask = sep >= minsep
    valid_mask = valid_mask & (targets >= 0)  # negative targets are invalid

    if maxsep is not None:
        valid_mask &= sep < maxsep

    if src_lengths is not None:
        valid = seqlen_range.unsqueeze(0) < src_lengths.unsqueeze(1)
        valid_mask &= valid.unsqueeze(1) & valid.unsqueeze(2)
    else:
        src_lengths = torch.full([batch_size], seqlen, device=device, dtype=torch.long)

    predictions = predictions.masked_fill(~valid_mask, float("-inf"))

    x_ind, y_ind = np.triu_indices(seqlen, minsep)
    predictions_upper = predictions[:, x_ind, y_ind]
    targets_upper = targets[:, x_ind, y_ind]

    topk = seqlen if override_length is None else max(seqlen, override_length)
    indices = predictions_upper.argsort(dim=-1, descending=True)[:, :topk]
    topk_targets = targets_upper[torch.arange(batch_size).unsqueeze(1), indices]
    if topk_targets.size(1) < topk:
        topk_targets = F.pad(topk_targets, [0, topk - topk_targets.size(1)])

    cumulative_dist = topk_targets.type_as(predictions).cumsum(-1)

    gather_lengths = src_lengths.unsqueeze(1)
    if override_length is not None:
        gather_lengths = override_length * torch.ones_like(
            gather_lengths, device=device
        )

    gather_indices = (
        torch.arange(0.1, 1.1, 0.1, device=device).unsqueeze(0) * gather_lengths
    ).type(torch.long) - 1

    binned_cumulative_dist = cumulative_dist.gather(1, gather_indices)
    binned_precisions = binned_cumulative_dist / (gather_indices + 1).type_as(
        binned_cumulative_dist
    )

    pl5 = binned_precisions[:, 1]
    pl2 = binned_precisions[:, 4]
    pl = binned_precisions[:, 9]
    auc = binned_precisions.mean(-1)

    return {"AUC": auc, "P@L": pl, "P@L2": pl2, "P@L5": pl5}


def evaluate_prediction(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    contact_ranges = [
        ("local", 3, 6),
        ("short", 6, 12),
        ("medium", 12, 24),
        ("long", 24, None),
    ]
    metrics = {}
    targets = targets.to(predictions.device)
    for name, minsep, maxsep in contact_ranges:
        rangemetrics = compute_precisions(
            predictions,
            targets,
            minsep=minsep,
            maxsep=maxsep,
        )
        for key, val in rangemetrics.items():
            metrics[f"{name}_{key}"] = val.item()
    return metrics


def get_results(input_directory, num_seqs=128):
    """ Returns MSA Transformer contact perdiction accuracy for a chosen protein."""

    reference_1_path = glob.glob(input_directory + "original/reference_1/*.cif")[0]
    reference_2_path = glob.glob(input_directory + "original/reference_2/*.cif")[0]
    reference_1 = glob.glob(input_directory + "reference_1/*.pdb")[0].split("/")[-1].split(".")[0]
    reference_2 = glob.glob(input_directory + "reference_2/*.pdb")[0].split("/")[-1].split(".")[0]

    structure_reference_1 = get_structure(PDBxFile.read(reference_1_path))[0]
    structure_reference_2 = get_structure(PDBxFile.read(reference_2_path))[0]
    print(reference_1, reference_2)
    contacts_reference_1 = contacts_from_pdb(structure_reference_1)
    contacts_reference_2 = contacts_from_pdb(structure_reference_2)
    print(contacts_reference_1.shape, contacts_reference_2.shape)
    
    msa_transformer_predictions = {}
    msa_transformer_results_1 = []
    msa_transformer_results_2 = []
    sequences = []

    msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    msa_transformer = msa_transformer.eval().cuda()
    msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()

    clusters = glob.glob(input_directory + "original/msa_clusters/*")
    print(len(clusters))

    for cluster_path in clusters:
        if cluster_path.find("_U1") != -1 or not cluster_path.endswith(".a3m"):
            continue
        name = cluster_path.split("/")[-1]
        inputs = read_msa(cluster_path)
        torch.cuda.empty_cache()
        sequences.append(len(inputs))
        if len(inputs) > num_seqs:
            inputs = greedy_select(inputs, num_seqs=num_seqs)
        msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = msa_transformer_batch_converter([inputs])
        msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(next(msa_transformer.parameters()).device)
        msa_transformer_predictions[name] = msa_transformer.predict_contacts(msa_transformer_batch_tokens)[0].cpu()
        
        prediction = msa_transformer_predictions[name]
        metrics = {"id": name, "reference":reference_1, "model": "MSA Transformer (Unsupervised)"}
        metrics.update(evaluate_prediction(prediction, contacts_reference_1))
        msa_transformer_results_1.append(metrics)

        prediction = msa_transformer_predictions[name]
        metrics = {"id": name, "reference":reference_2, "model": "MSA Transformer (Unsupervised)"}
        metrics.update(evaluate_prediction(prediction, contacts_reference_2))
        msa_transformer_results_2.append(metrics)
    
    msa_transformer_results_1 = pd.DataFrame(msa_transformer_results_1)
    msa_transformer_results_2 = pd.DataFrame(msa_transformer_results_2)

    msa_transformer_results_1["num_seqs"] = sequences
    msa_transformer_results_2["num_seqs"] = sequences

    return msa_transformer_results_1, msa_transformer_results_2


def display_contact_results(results_0, results_1, title=None, color_by_seqs=True):
    """ Displays the contact prediction results compared with reference proteins in a 2D plot."""

    reference_0 = results_0["reference"][0]
    reference_1 = results_1["reference"][0]

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    if title is not None:
        fig.suptitle(title)

    # Loop over the subplots and plot the data
    for i, metric in enumerate(["long_P@L", "long_P@L2", "long_P@L5", "medium_P@L", "short_P@L", "local_P@L"]):
        row = i // 3
        col = i % 3
        axs[row, col].set_xlabel(f"{metric} for {reference_0}")
        axs[row, col].set_ylabel(f"{metric} for {reference_1}")
        axs[row, col].set_title(f"{metric} for {reference_0} and {reference_1}")
        
        axs[row, col].set_xlim([0, 1])
        axs[row, col].set_ylim([0, 1])
        
        # Adjusted num_seqs to give a roof of 128
        adjusted_num_seqs = results_0["num_seqs"].apply(lambda x: min(x, 128))
        
        if color_by_seqs:
            # Color the points by the number of sequences in the subset
            axs[row, col].scatter(results_0[metric][:-1], results_1[metric][:-1], s=10, c=adjusted_num_seqs[:-1], cmap='cool')
            axs[row, col].scatter(results_0[metric][-1:], results_1[metric][-1:], s=10, c='black', marker='s')
            # Add a color bar
            if col == 2:
                fig.colorbar(axs[row, col].collections[0], ax=axs[row, col])
        else:
            axs[row, col].scatter(results_0[metric][:-1], results_1[metric][:-1], s=10)

    fig.tight_layout()
    return fig


def main(protein_dir, image_dir, identifier):
    print("Getting results...")
    results_1, results_2 = get_results(protein_dir, num_seqs=128)
    
    print("Plotting results...")
    plot_title = identifier + "_" + protein_dir.split("/")[-2]
    plot = display_contact_results(results_1, results_2, title=plot_title, color_by_seqs=False)
    plot.savefig(image_dir + plot_title + ".png")
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('ERROR: Inputs should be: protein_directory, image_directory, identifier')
    main(sys.argv[1], sys.argv[2], sys.argv[3])
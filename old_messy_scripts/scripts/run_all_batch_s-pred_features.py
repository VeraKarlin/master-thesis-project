""" Runs MSA Transformer to create s-pred features for each PDB ID in the input text file.
The features are created in parallel in a specified number of batches.

Args:
    pdb_list_path (str): The (.txt) file location of the list of PDB IDs.
    batches (int) : Number of batches to run in parallel. Keep at 1 on madoka due to memory constraints.
    msa_directory (str): The directory where the MSA (.a3m) files are stored.
    output_directory (str): The directory where the feature (.pt) files should be created.
"""


import os
import sys
import subprocess


def main(pdb_list_path, batches, msa_directory, output_directory):

    with open(pdb_list_path) as input_file:
        pdb_list = input_file.read().splitlines()

    batch_list = [pdb_list[i::batches] for i in range(batches)]
    script_path = os.path.abspath(__file__).replace('run_all_batch_', 'batch_')

    for i in range(len(batch_list)):
        command = ['python', script_path, '+'.join(batch_list[i]), msa_directory, output_directory]
        subprocess.Popen(command)


if __name__ == "__main__":
    error = False
    if len(sys.argv) != 5:
        print('ERROR: Inputs should be: pdb_list_path, batches, msa_directory, output_directory!')
        error = True
    else:
        if not os.path.isfile(sys.argv[1]):
            print('ERROR: The specified PDB ID text file "' + sys.argv[1] + '" does not exist!')
            error = True
        if not sys.argv[2].isnumeric():
            print('ERROR: The specified batch count "' + sys.argv[2] + '" is not an integer!')
            error = True
        if not os.path.isdir(sys.argv[3]):
            print('ERROR: The specified MSA directory "' + sys.argv[3] + '" does not exist!')
            error = True
        if not os.path.isdir(sys.argv[4]):
            print('ERROR: The specified output directory "' + sys.argv[4] + '" does not exist!')
            error = True
    if not error:
        main(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
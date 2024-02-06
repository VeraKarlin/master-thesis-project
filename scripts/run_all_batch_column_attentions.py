""" Runs MSA Transformer to create query matrices for each PDB ID in the input text file.
The embeddings are created in parallel in a specified number of batches.

Args:
    pdb_list_path (str): The (.txt) file location of the list of PDB IDs.
    batches (int) : Number of batches to run in parallel. Keep at 1 on madoka due to memory constraints.
    msa_directory (str): The directory where the MSA (.a3m) files are stored.
    output_directory (str): The directory where the query matrix (.pt) files should be created.
"""


import os
import sys
import subprocess


def validate_inputs(args):
    if not os.path.isfile(args[0]):
        print(f'ERROR: The specified PDB ID text file "{args[0]}" does not exist!')
        return False
    if not args[1].isnumeric():
        print(f'ERROR: The specified batch count "{args[1]}" is not an integer!')
        return False
    if not os.path.isdir(args[2]):
        print(f'ERROR: The specified MSA directory "{args[2]}" does not exist!')
        return False
    if not os.path.isdir(args[3]):
        print(f'ERROR: The specified output directory "{args[3]}" does not exist!')
        return False
    if len(args) >= 5 and not args[4].isnumeric():
        print(f'ERROR: The specified sequence count "{args[5]}" is not an integer!')
        return False
    if len(args) >= 6 and not args[4].isnumeric():
        print(f'ERROR: The starting index count "{args[6]}" is not an integer!')
        return False
    return True


def main(args):
    if not validate_inputs(args):
        print('ERROR: Inputs should be: pdb_list_path, batches, msa_directory, output_directory, (sequence_count), (start_index)!')
        return

    pdb_list_path, batches, msa_directory, output_directory = args[:4]
    sequence_count = int(args[4]) if len(args) >= 5 else 128
    start_index = int(args[5]) if len(args) == 6 else 0

    with open(pdb_list_path) as input_file:
        pdb_list = input_file.read().splitlines()

    batch_list = [pdb_list[i::int(batches)] for i in range(int(batches))]
    script_path = os.path.abspath(__file__).replace('run_all_batch_', 'batch_')

    for i in range(len(batch_list)):
        command = ['python', script_path, '+'.join(batch_list[i]), msa_directory, output_directory, str(sequence_count), str(start_index)]
        subprocess.Popen(command)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print('ERROR: Inputs should be: pdb_list_path, batches, msa_directory, output_directory, (sequence_count), (start_index)!')
    else:
        main(sys.argv[1:])
        print('Done')


""" Creates FASTA.seq files for the input pdb_IDs in the specified directory.

Args:
    pdb_list_path (str): The (.txt) file location of the list of PDB IDs.
    fasta_list_path (str) The file location of the FASTA database pdb_seqres.txt.
    output_directory (str): The directory where the fasta files should be created.
"""

import sys
import os


def main(pdb_list_path, fasta_list_path, output_directory):

    # Read in a list of PDB IDs
    with open(pdb_list_path) as input_file:
        pdb_ids = input_file.read().splitlines()

    # Create a dictionary of FASTA sequences
    with open(fasta_list_path) as fasta_list:
        lines = fasta_list.readlines()
    entries = [''.join(x) for x in zip(lines[0::2], lines[1::2])]
    pdb_keys = [entry.split(' ')[0][1:].lower() for entry in entries]
    fasta_dict = dict(zip(pdb_keys, entries))

    # Create a FASTA.seq file for each pdb_id, provided it exists in the FASTA dictionary
    i = 0
    for pdb_id in pdb_ids:
        pdb_id = pdb_id.lower()
        if len(pdb_id) == 5:
            pdb_id = pdb_id[:4] + '_' + pdb_id[4]
        if pdb_id not in pdb_keys:
            i += 1
            print('ERROR: PDB ID not found:', i, pdb_id)
            continue
        fasta_text = fasta_dict[pdb_id]
        fasta_file = os.path.join(output_directory, 'query_' + pdb_id + '.seq')
        with open(fasta_file, "w") as fasta_file_open:
            fasta_file_open.write(fasta_text)
    return


if __name__ == "__main__":
    error = False
    if len(sys.argv) != 4:
        print('ERROR: Inputs should be: pdb_list_path, fasta_list_path, output_directory!')
        error = True
    else:
        if not os.path.isfile(sys.argv[1]):
            print('ERROR: The specified PDB ID text file "' + sys.argv[1] + '" does not exist!')
            error = True
        if not os.path.isfile(sys.argv[2]):
            print('ERROR: The FASTA database text file "' + sys.argv[2] + '" does not exist!')
            error = True
        if not os.path.isdir(sys.argv[3]):
            print('ERROR: The specified output directory "' + sys.argv[3] + '" does not exist!')
            error = True
    if not error:
        main(sys.argv[1], sys.argv[2], sys.argv[3])


""" Creates a multiple sequence alignment .a3m file using HHblits for each PDB ID in the input.

Args:
    pdb_ids (str): PDB IDs as a string, seperated by a '+' sign.
    fasta_directory (str): The directory where the fasta (.seq) files are stored.
    db_directory (str): Path to the uniclust30 database, usually 'hhsuite-3.3.0-AVX2-Linux/databases/uniclust30_2018_08/uniclust30_2018_08'.
    output_directory (str): The directory where the MSA (.a3m) files should be created.
"""


import sys
import os
import subprocess


def main(pdb_ids, fasta_directory, db_directory, output_directory):

    for pdb_id in pdb_ids.split('+'):

        # Make sure the PDB ID is in the right format
        pdb_id = pdb_id.lower()
        if len(pdb_id) == 5:
            pdb_id = pdb_id[:4] + '_' + pdb_id[4]

        fasta_path = os.path.join(fasta_directory, 'query_' + pdb_id + '.seq')
        output_path = os.path.join(output_directory, 'aligned_' + pdb_id + '.a3m')

        # Run HHblits if the fasta file exists
        if not os.path.isfile(fasta_path):
            print('ERROR: The FASTA file for "' + pdb_id + '" does not exist!')
            continue
        command = 'hhblits -cpu 4 -i ' + fasta_path + ' -d ' + db_directory + ' -oa3m ' + output_path + ' -n 3'
        print(command)
        subprocess.run(command.split(' '), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    error = False
    if len(sys.argv) != 5:
        print('ERROR: Inputs should be: pdb_ids, fasta_directory, db_directory, output_directory!')
        error = True
    else:
        if not os.path.isdir(sys.argv[2]):
            print('ERROR: The specified FASTA directory "' + sys.argv[2] + '" does not exist!')
            error = True
        if not os.path.isdir(sys.argv[4]):
            print('ERROR: The specified output directory "' + sys.argv[4] + '" does not exist!')
            error = True
    if not error:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

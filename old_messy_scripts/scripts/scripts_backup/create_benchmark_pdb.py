import os
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
from Bio import SeqIO
from Bio.PDB import *
from Bio import pairwise2
from Bio.Seq import Seq
import biotite.structure as bs
from biotite.structure.io.pdbx import PDBxFile, get_structure
from biotite.database import rcsb
import sys

parser = PDBParser(PERMISSIVE = True, QUIET = True) 
pdbl = PDBList() 

def download_pdb_chain(pdb_id, path):
    """ Downloads a PDB file for a given PDB ID and chain ID."""
    pdb = pdb_id.upper()[:4]
    chain_id = pdb_id.upper()[-1]

    # Download the PDB file
    parser = PDBParser()
    pdbl.retrieve_pdb_file(pdb, pdir = path, file_format = 'pdb')
    structure = parser.get_structure(pdb, path + "pdb" + pdb.lower() + ".ent")
    os.remove(path + "pdb" + pdb.lower() + ".ent")

    model = copy.deepcopy(structure[0])
    for i, chain in reversed([c for c in enumerate(model.child_list)]):
        # Remove all chains except the one we want
        if chain.id != chain_id:
            model.child_list.remove(model.child_list[i])
        else:
            # Remove all hetatoms
            for i, residue in reversed([r for r in enumerate(chain.child_list)]):
                if residue.id[0] != ' ':
                    chain.child_list.remove(chain.child_list[i])

    # Create a new PDB file
    io=PDBIO()
    io.set_structure(model)
    io.save(path + pdb_id + ".pdb")
    return



def main(directory, pdb_id):
    parser = PDBParser(PERMISSIVE = True, QUIET = True) 
    pdbl = PDBList() 

    pdb = pdb_id.lower()[:4]
    chain_id = pdb_id.upper()[-1]
    
    # Get the data from the pdb file
    pdbl.retrieve_pdb_file(pdb, pdir = directory, file_format = 'pdb')
    structure = parser.get_structure(pdb, directory + "pdb" + pdb + ".ent")
    os.remove(directory + "pdb" + pdb + ".ent")

    for model in structure:
        for chain in model:
            class ChainSelect(Select):
                        def accept_chain(self, chain):
                            return chain.get_id().upper() == chain_id
            
    io = PDBIO()
    io.set_structure(structure)
    io.save(directory + pdb_id + ".pdb" , ChainSelect())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('ERROR: Inputs should be: directory, pdb_id!')
    #main(sys.argv[1], sys.argv[2])
    download_pdb_chain(sys.argv[2], sys.argv[1])

import biotite.structure as bs
from biotite.structure.io.pdbx import PDBxFile, get_structure
from biotite.database import rcsb


PDB_IDS = ["6uyd_f", "1a81_a", "1ae7_a"]


structures = {
    name.lower(): get_structure(PDBxFile.read(rcsb.fetch(name[:4], "cif")))[0]
    for name in PDB_IDS
}

print(structures)

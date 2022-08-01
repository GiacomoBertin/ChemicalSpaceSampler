#!/bin/bash
mk_prepare_ligand.py -i /home/giacomo/Documents/LCP_runs/RUN_1/docking/ligands/2993/LIG.sdf -o /home/giacomo/Documents/LCP_runs/RUN_1/docking/complexes/2993/6fff_2993_ligand.pdbqt 
prepare_receptor -r /home/giacomo/Documents/LCP_runs/RUN_1/docking/complexes/2993/6fff_2993_protein.pdb -o /home/giacomo/Documents/LCP_runs/RUN_1/docking/complexes/2993/6fff_2993_protein.pdbqt

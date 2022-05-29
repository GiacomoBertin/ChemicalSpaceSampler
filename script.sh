#!/bin/bash
mk_prepare_ligand.py -i /home/giacomo/Documents/LCP_runs/docking/ligands/173/LIG.sdf -o /home/giacomo/Documents/LCP_runs/docking/complexes/173/4UYG_173_ligand.pdbqt 
prepare_receptor -r /home/giacomo/Documents/LCP_runs/docking/complexes/173/4UYG_173_protein.pdb -o /home/giacomo/Documents/LCP_runs/docking/complexes/173/4UYG_173_protein.pdbqt

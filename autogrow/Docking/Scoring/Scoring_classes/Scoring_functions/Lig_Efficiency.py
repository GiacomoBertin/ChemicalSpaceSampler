

import __future__

import glob
import os

import rdkit
import rdkit.Chem as Chem
#Disable the unnecessary RDKit warnings
rdkit.RDLogger.DisableLog('rdApp.*')

from autogrow.Docking.Scoring.Scoring_classes.ParentScoringClass import ParentScoring
from autogrow.Docking.Scoring.Scoring_classes.Scoring_functions.VINA import VINA 

class Lig_Efficiency(VINA):
    """
    This will Score a given ligand for its binding affinity based on VINA or QuickVina02 type docking.

    This inherits many functions from the VINA scoring function. The only difference is that this scoring function
    uses the ligand efficiency instead of docking score. 
    ligand efficiency is the docking score divided by the number of heavy atoms (non-hydrogens)

    Inputs:
    :param class VINA: the VINA scoring function which this class inherits from.
    """        

    def __init__(self, vars, smiles_dict):
        """
        This will take vars and a list of smiles.

        Inputs:
        :param dict vars: Dictionary of User variables
        :param dict smiles_dict: a dict of ligand info of SMILES, IDS, and short ID
        """
        self.vars = vars
        
        self.smiles_dict = smiles_dict
    # 

    def get_score_from_a_file(self, file_path):
        """
        Make a list of a ligands information including its docking score.


        Inputs:
        :param str file_path: the path to the file to be scored

        Returns:
        :returns: list lig_info: a list of the ligands short_id_name and the docking score from the best pose.
        """

        # grab the index of the ligand for the score
        basefile = os.path.basename(file_path)
        basefile_strip = basefile.replace(".pdbqt.vina","")
        basefile_split = basefile.split("__")
        ligand_short_name = basefile_split[0]
        ligand_pose = basefile_split[1]

        affinity = None

        with open(file_path, 'r') as f:
            for line in f.readlines():
                if "REMARK VINA" in line:
                    line_stripped = line.replace("REMARK VINA RESULT:","").replace("\n","")
                    line_split = line_stripped.split()
                    
                    if affinity is None:
                        affinity = float(line_split[0])
                    else:
                        if affinity > float(line_split[0]):
                            affinity = float(line_split[0])
        if affinity is None:
            # This file lacks a pose to use
            return None

        lig_info = [ligand_short_name, affinity]

        if ligand_short_name not in list(self.smiles_dict.keys()):
            return None
        
        lig_info = self.merge_smile_info_w_affinity_info(lig_info)
        
        # For saftey remove Nones and empty lists
        if type(lig_info) is not type([]) or len(lig_info) == 0:
            return None

        lig_info = append_lig_effeciency(lig_info)
        if lig_info == None:
            return None
        lig_info = [str(x) for x in lig_info]
       
        return lig_info

    #

# These Functions are placed outside the class for multithreading reasons. Multithreading doesn't like being executed within the class.

def get_number_heavy_atoms(SMILES_str):
    """
    Get the number of non Hydrogens in a SMILE


    Inputs:
    :param str SMILES_str: a str representing a molecule
    Returns:
    :returns: int num_heavy_atoms: a int of the count of heavy atoms 
    """
    if SMILES_str is None:
        return None
    # easiest nearly everything should get through 

    try:
        mol = Chem.MolFromSmiles(SMILES_str, sanitize=False)
    except:
        mol = None

    if mol is None:
        return None

    atom_list = mol.GetAtoms()
    num_heavy_atoms = 0
    for atom in atom_list:
        if atom.GetAtomicNum() != 1:
            num_heavy_atoms = num_heavy_atoms + 1

    return num_heavy_atoms
#
    
def append_lig_effeciency(list_of_lig_info):
    """
    Determine the ligand efficiency and append it to the end of a list which has the ligand information.


    Inputs:
    :param list list_of_lig_info: a list containing ligand informations with idx=0 as the SMILES str and idx=-1 is the docking score

    Returns:
    :returns: list list_of_lig_info: the same list as list_of_lig_info, 
                                but with each sublist now having the ligand efficiency score appended to the end.
    
    """

    if type(list_of_lig_info) == None:
        return None
    elif type(list_of_lig_info) == list:
        if None in list_of_lig_info:
            return None
            
    # Unpack ligand info
    lig_smiles_str = str(list_of_lig_info[0])
    affinity = float(list_of_lig_info[-1])
    
    # Get num of heavy atoms
    heavy_atom_count = get_number_heavy_atoms(lig_smiles_str)
    
    if heavy_atom_count is None or heavy_atom_count == 0:
        return None

    # Convert to Lig efficiency (aka affinity/heavy_atom_count )
    lig_efficieny = float(affinity) / float(heavy_atom_count)

    # Append lig_efficiency to list_of_lig_info
    list_of_lig_info.append(str(lig_efficieny))
    
    return list_of_lig_info
#


if __name__ == "__main__":
    vars = {}
    # vars['scoring_function'] = 'VINA'
    # folder_to_search = os.sep + os.path.join("home", "jspiegel", "DataB", "jspiegel", "projects", "output_autogrow_testing", "Run_11", "generation_0", "PDBs") + os.sep
    # run_scoring_common(vars, folder_to_search)
    smile_dict = {'Gen_0_Mutant_5_203493': ['COC1OC(CO)C(O)C(O)C1n1nnc(CCO)c1-c1ccc(-c2cccs2)s1', '(ZINC04530731+ZINC01529972)Gen_0_Mutant_5_203493'], 'Gen_0_Cross_452996': ['CC(=O)OCC(O)CN=[N+]=[N-]', '(ZINC44117885+ZINC34601304)Gen_0_Cross_452996'], 'ZINC13526729': ['[N-]=[N+]=NCC1OC(O)CC1O', 'ZINC13526729']}

    print(append_lig_effeciency(smile_dict['Gen_0_Mutant_5_203493']))
    
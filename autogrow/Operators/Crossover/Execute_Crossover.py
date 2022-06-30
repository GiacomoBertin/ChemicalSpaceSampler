"""Pre-SmileMerge_MCS_Filter

This script should take an input of a randomly selected file containing a list of smiles.
        
A random number function will be used to pick 2 non-identical numbers for 0 to the len(smile_list)
Then those numbers are used to grab 2 non-identical molecules from smile_list
Those 2 smiles are tested using the MCS function to find the most common structure (MCS).
If MCS returns None (ie. no shared structures)
then mol2 is reassigned using the random function generator.
This iterates until a shared structure is returned.
"""
import  __future__ 

import random
import copy

import rdkit
from rdkit import Chem
from rdkit.Chem import rdFMCS
#Disable the unnecessary RDKit warnings
rdkit.RDLogger.DisableLog('rdApp.*')

import autogrow.Operators.Filter.ExecuteFilters as Filter
import autogrow.Operators.Crossover.SmilesMerge.SmilesMerge as SmilesMerge
import autogrow.Operators.ConvertFiles.gypsum_dl.gypsum_dl.MolObjectHandling as MOH


def test_for_MCS(vars, mol_1, mol_2):
    """
    Takes a ligand and a random selected molecule and looks
    for the Most common structure. rdFMCS.FindMCS(mols) flags an error if there
    are no common structures being compared. Try/except statement used to prevent
    program crash when 2 unlike molecules are compared.
    mols is a list of the molecules to be compare using rdFMCS.FindMCS
    
    This can function with mol_1 and mol_2 having H's but we recommend using this function
    with molecules with H's removed. If implicit H's are added they will be recoginized as part of
    MCS. This means 1 atom in common with 3H's in common would pass a atom similarity count of 4 atoms
    shared, but really its only 1 non-H atom...

    Inputs:
    :param dict vars: User variables which will govern how the programs runs
    :param rdkit.Chem.rdchem.Mol mol_1: the 1st rdkit molecule 
    :param rdkit.Chem.rdchem.Mol mol_2: the 2nd rdkit molecule 
    Returns:
    :returns: <class 'rdkit.Chem.rdFMCS.MCSResult'> results: an MCSResults object
                    returns None if it fails to find MCS sufficient with the User defined parameters.
    """

    
    mols = [mol_1, mol_2]  
    time_timeout = vars["max_time_MCS_prescreen"]
    min_number_atoms_matched = vars["min_atom_match_MCS"]

    try: 
        result = rdFMCS.FindMCS(mols, matchValences = False, ringMatchesRingOnly = True, completeRingsOnly = False, timeout = time_timeout)
    except:
        return None
    
    # could be used for a theoretical timeout prefilter was to be implement (but this isn't implimented)
    # (ie. if it times out the prefilter dont use in thorough MCS ligmerge)
    # canceled: if True, the MCS calculation did not finish
    
    
    # filter by UserDefined minimum number of atoms found
    # The higher the number the more similar 2 ligands are but the more restrictive for finding mergable ligands
    # number of atoms in common found
    if result.numAtoms < min_number_atoms_matched:
        return None
    if result.canceled == True:
        return None
    else:
        return result
# 

def find_random_lig2(vars, ligands_list, Ligand1_pair):
    """
    Pick a random molecule from the list and check that it can be
    converted into a rdkit mol object and then test for a satistifactory 
    Most common substructure (MCS) which satisifies the User specified minimum shared substructure

    NECESSARY INCASE THE SMILE CANNOT BE USED (ie. valence issue) 
    Inputs:
    :param dict vars: User variables which will govern how the programs runs
    :param list ligands_list: list of all the lignads to chose from
    :param list Ligand1_pair: information for the Ligand 1. This info includes the name and SMILES string

    Returns:
    :returns: list mol2_pair: a set of information for a 2nd ligand (Lig2) This includes the name and SMILES string 
                            this mol is from the ligands_list
    :returns: bool bool: returns False if no satistifactory matches were found it returns False
    """
    count = 0
    shuffled_num_list = list(range(0,len(ligands_list)-1))
    random.shuffle(shuffled_num_list)

    # Convert Lig_1 into an RDkit mol
    Lig_1_string = Ligand1_pair[0]
    lig1_mol = convert_MolFromSmiles(Lig_1_string)    

    while count < len(ligands_list) - 1:
        rand_num = shuffled_num_list[count]
        mol2_pair = ligands_list[rand_num]

        if mol2_pair[0] == Lig_1_string:
            count = count + 1
            continue
        
        # Convert Lig_1 into an RDkit mol
        Lig_2_string = mol2_pair[0]
        lig2_mol = convert_MolFromSmiles(Lig_2_string)
        
        if lig2_mol is None:
            count = count + 1
            continue
        else:
            # it converts and it is not Ligand1 
            # now lets test for a common substructure
            
            if test_for_MCS(vars, lig1_mol, lig2_mol) is None:
                count = count + 1
                continue
            else:
                # We found a good pair of Ligands
                return mol2_pair

    return False
#

def convert_MolFromSmiles(SmilesString):
    """
    Test a SMILES string can be converted into an rdkit molecule (rdkit.Chem.rdchem.Mol)  
    and be sanitize. 
    This also deprotanates them

    Inputs:
    :param str SmilesString: a single SMILES String
    Returns:
    :returns: rdkit.Chem.rdchem.Mol mol: an rdkit molecule object if it properly converts from the SMILE
                        and None
    """
    try:
        mol = Chem.MolFromSmiles(SmilesString, sanitize = False)  
    except:
        return None
    
    mol = MOH.check_sanitization(mol)
    if mol is None:
        return None

    mol = MOH.try_deprotanation(mol)
    if mol == None:
        return False
    return mol
#

######################### 
#### RUN MAIN PARTS #####
#########################  
def make_Crossovers(vars, generation_num, number_of_processors, num_crossovers_to_make, list_previous_gen_smiles, new_crossover_smiles_list):
    """
    Make crossover compounds in a list to be returned.
    
    This runs SmileClick and returns a list of new molecules.
    
    Input:
    :param dict vars: User variables which will govern how the programs runs
    :param int generation_num: the generation number indexed by 0
    :param int number_of_processors: number of processors to multithread with
    :param int num_crossovers_to_make: number of crossovers to make User specified
    :param list list_previous_gen_smiles: a list of molecules to seed the current generations crossovers
    :param list new_crossover_smiles_list: a list of ligands made by crossover for this generation but in
                                            a previous run of crossover
                                            -ie. if filtering ligands removed some of the ligands generated by
                                                crossover, it requires another loop of crossover to fill out the list
                                                so this is used to prevent creating the same mol multiple times
    Returns:
    :returns: list new_ligands_list: list of new unique ligands with unique names/IDS
                                    ["CCC" (zinc123+zinc456)Gen_1_Cross_123456"]
                            return None if no new mol gets generated
    """
    if len(new_crossover_smiles_list) == 0:
        new_ligands_list = []
    else:
        new_ligands_list = copy.deepcopy(new_crossover_smiles_list)

    # Use a temp vars dict so you don't put mpi multiprocess info through itself...
    temp_vars = {}
    for key in list(vars.keys()):
        if key =="Parallelizer":
            continue
        temp_vars[key]= vars[key]

    new_ligands_list = []
    number_of_processors = int(vars['Parallelizer'].return_node())
    
    loop_counter = 0 
    while loop_counter < 2000 and len(new_ligands_list) < num_crossovers_to_make:

        react_list = copy.deepcopy(list_previous_gen_smiles)
        
        
        while len(new_ligands_list) < num_crossovers_to_make and len(react_list) > 0:
            
            num_to_grab = num_crossovers_to_make - len(new_ligands_list)
            num_to_make = num_to_grab

            # to minimize a big loop of running a single crossover at a time we will make 1 new lig/processor
            # This will help to prevent wasting reasources and time. 
            if num_to_make < number_of_processors:
                num_to_make = number_of_processors
                
            smile_pairs = [react_list.pop() for x in range(num_to_make) if len(react_list) > 0]

            smile_inputs = [x[0] for x in smile_pairs]
            smile_names = [x[1] for x in smile_pairs]
                                
            # make a list of tuples for multi-processing Crossover
            job_input = []            
            for i in smile_pairs:
                temp = tuple([temp_vars, i, list_previous_gen_smiles])
                job_input.append(temp)
            job_input = tuple(job_input)
            # Example information:
            # result is a list of lists
            # result = [[Ligand_new_smiles, Lig1_smile_pair,Lig_2_pair],...]
            # Ligand_new_smiles is the smiles string of a new ligand from crossover
            # Lig1_smile_pair = ["NCCCCCC","zinc123"]
            # Lig2_smile_pair = ["NCCCO","zinc456"]
            # Lig1 and lig 2 were used to generate the Ligand_new_smiles

            results = vars['Parallelizer'].run(job_input, do_Crossovers_smiles_Merge)
            results =  [x for x in results if x is not None]

            for index, i in enumerate(results):
                if i is None:
                    continue

                # Get the new molecule's (aka the Child lig) Smile string
                child_lig_smile = i[0]

                # get the ID for the parent of a child mol
                parent_lig_1_ID = i[1][1]
                parent_lig_2_ID = i[2][1]

                # get the unique ID (last few diget ID of the parent mol)
                parent_lig_1_ID = parent_lig_1_ID.split(")")[-1]
                parent_lig_2_ID = parent_lig_2_ID.split(")")[-1]

                # Make a list of all smiles and smile_ID's of all previously made smiles in this generation
                list_of_already_made_smiles = []
                list_of_already_made_ID = []

                # fill lists of all smiles and smile_ID's of all previously made smiles in this generation
                for x in new_ligands_list:
                    list_of_already_made_smiles.append(x[0])
                    list_of_already_made_ID.append(x[1])

                if child_lig_smile not in list_of_already_made_smiles:
                    # if the smiles string is unique to the list of previous smile strings
                    # in this round of reactions then we append it to the list of newly created ligands
                    # we append it with a unique ID, which also tracks the progress of the reactant
                    is_name_unique = False
                    while is_name_unique is False:     

                        # make unique ID with the 1st number being the ligand_ID_Name for the derived mol
                        # second being the lig2 number
                        # Followed by Cross 
                        # folowed by the generationnumber
                        # followed by a  unique 

                        random_id_num = random.randint(100,1000000)
                        new_lig_ID = "({}+{})Gen_{}_Cross_{}".format(parent_lig_1_ID,parent_lig_2_ID,generation_num,random_id_num)
                        
                        # check name is unique
                        if new_lig_ID not in list_of_already_made_ID:
                            is_name_unique = True

                    # make a temporary list containing the smiles string of the new product
                    # and the unique ID
                    ligand_info = [child_lig_smile, new_lig_ID]

                    # append the new ligand smile and ID to the list of all newly made ligands
                    new_ligands_list.append(ligand_info)
            
        loop_counter = loop_counter + 1

    # 
    if len(new_ligands_list) < num_crossovers_to_make:
        return None
    else:
        # once the number of mutants we need is generated
        # return the list

        return new_ligands_list
#

def run_SmilesMerge_prescreen(vars, ligands_list, Ligand1_pair):
    """
    This function runs a series of functions to find two molecules
    with a sufficient amount of shared common structure (most common structure = MCS)
    
    These two parent molecules will be derived from a list of molecules from the 
    parent generation.
    
    Inputs:
    :param dict vars: User variables which will govern how the programs runs
    :param list ligands_list: list of all the lignads to chose from
    :param list Ligand1_pair: information for the Ligand 1. This info includes the name and SMILES string
    Returns:
    :returns: list Lig_2_pair: a set of information for a 2nd ligand (Lig2) This includes the name and SMILES string 
                            this mol is from the ligands_list
                            returns None if a ligand with a sufficient MCS can not be found return None
    """
    Ligand_1_string = Ligand1_pair[0]

    # check if Ligand_1 can be converted to an rdkit mol 
    Lig_1 = convert_MolFromSmiles(Ligand_1_string)
    if Lig_1 is False:
        # Ligand1_string failed to be converted to rdkit mol format
        return None
    
    # GET TWO UNIQUE LIGANDS TO WITH A SHARED SUBSTRUCTURE
    Lig_2_pair = find_random_lig2(vars, ligands_list, Ligand1_pair)
       
    if Lig_2_pair is False:
        # Ligand_1 has no matches 
        return None
    
    else:
        return Lig_2_pair
#
 
def do_Crossovers_smiles_Merge(vars, Lig1_smile_pair, ligands_list):
    """
    This function will take the list of ligands to work on and the number in that list
    for the Ligand 1. 

    It will then prescreen the Ligand 1 using the run_SmilesMerge_prescreen which should return 
    either a None if no matches are found, or the Smile string of a second ligand (ligand 2)
    which has some share common substructure with Ligand 1.

    This pair of ligands will be passed off to the SmilesMerge.run_main_SmilesMerge function
    which will execute a crossover and return a new molecule 

    Input:
    :param dict vars: User variables which will govern how the programs runs
    :param list Lig1_smile_pair: a list with the SMILES string and info for lig1
    :param list ligands_list: a list of all the seed ligands from the previous generation

    Returns:
    :returns: str Ligand_new_smiles: a new mol's SMILES string
    :returns: list Lig1_smile_pair: a list of parent ligand 1's information
    :returns: list Lig_2_pair: a list of the parent ligand 2's information
    :returns: bool None: returns three Nones if there are no sufficient matches
    """
    # Run the run_SmilesMerge_prescreen of the ligand
    # This gets a new a lig2 which passed the prescreen
    Lig_2_pair = run_SmilesMerge_prescreen(vars, ligands_list, Lig1_smile_pair)

    if Lig_2_pair is None:
        return None
    else:
        Ligand_1_string = Lig1_smile_pair[0]
        Ligand_2_string = Lig_2_pair[0]

        counter=0
        while counter<3:
            # run SmilesMerge
            Ligand_new_smiles = SmilesMerge.run_main_SmilesMerge(vars, Ligand_1_string, Ligand_2_string)
        
            if Ligand_new_smiles is None:
                counter = counter+1
            else:
                # Filter Here
                pass_or_not = Filter.run_filter_on_just_smiles(Ligand_new_smiles, vars["Filter_Object_Dict"])
                if pass_or_not == False:

                    counter = counter+1
                else:
                    return [Ligand_new_smiles, Lig1_smile_pair, Lig_2_pair]
    return None
#
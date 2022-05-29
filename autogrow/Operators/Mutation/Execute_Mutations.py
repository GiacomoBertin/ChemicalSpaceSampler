"""
This function should contain all the info for executing the mutation functions
"""

import __future__

import random
import copy
import sys


import autogrow.Operators.Mutation.SmileClickChem.SmileClickClass as SmileClickClass



####################################### 
# Functions for creating molecular models                                 
##########################################
def make_mutants(vars, generation_num, number_of_processors, num_mutants_to_make, ligands_list, new_mutation_smiles_list, rxn_library_variables):
    """
    Make mutant compounds in a list to be returned
    
    This runs SmileClick and returns a list of new molecules

    Inputs:
    :param dict vars: a dictionary of all user variables
    :param int generation_num:    generation number
    :param int number_of_processors: number of processors as specified by the user
    :param int num_mutants_to_make: number of mutants to return
    :param list ligands_list: list of ligand/name pairs which are the order in which to be sampled
    :param list new_mutation_smiles_list: is the list of mutants made for the current generation being populated 
                                            but in a previous iteration of the loop in Operations
    :param list rxn_library_variables: a list of user variables which define the Rxn_library, rxn_library_file,     
                                    and function_group_library. 
                                    ie. rxn_library_variables = [vars['Rxn_library'], vars['rxn_library_file'], vars['function_group_library']]
    returns:
    :returns: list new_ligands_list: ligand/name pairs OR returns None if sufficient number was not generated.
              None: bol    if mutations failed
    """
    
    if len(new_mutation_smiles_list) == 0:
        new_ligands_list = []
    else:
        new_ligands_list = new_mutation_smiles_list
    
    loop_counter = 0 

    number_of_processors = int(vars['Parallelizer'].return_node())

    # initialize the smileclickclass
    aSmilesClickChem = SmileClickClass.SmilesClickChem(rxn_library_variables, new_mutation_smiles_list, vars["Filter_Object_Dict"])

    while loop_counter < 2000 and len(new_ligands_list) < num_mutants_to_make:

        react_list = copy.deepcopy(ligands_list)
        
        while len(new_ligands_list) < num_mutants_to_make and len(react_list) > 0:
            
            aSmilesClickChem.update_list_of_already_made_smiles(new_ligands_list)
            num_to_grab = num_mutants_to_make - len(new_ligands_list)
            num_to_make = num_to_grab

            # to minimize a big loop of running a single mutation at a time we will make 1 new lig/processor
            # This will help to prevent wasting reasources and time. 
            if num_to_make < number_of_processors:
                num_to_make = number_of_processors
                
            smile_pairs = [react_list.pop() for x in range(num_to_make) if len(react_list) > 0]
            
            smile_inputs = [x[0] for x in smile_pairs]
            smile_names = [x[1] for x in smile_pairs]

            job_input = tuple([tuple([smile, aSmilesClickChem]) for smile in smile_inputs])

            ######################################## 
            results = vars['Parallelizer'].run(job_input, run_Smile_Click_for_multithread)
            
            for index,i in enumerate(results):
                if i is not None:
                    
                    # Get the new molecule's (aka the Child lig) Smile string
                    child_lig_smile = i[0]

                    # get the reaction id number
                    reaction_id_number = i[1]

                    # get the ID for the parent of a child mol and the complimentary parent mol
                    # comp mol could be None or a zinc database ID
                    parent_lig_ID = smile_names[index]
                    zinc_ID_comp_mol = i[2]

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
                            # make unique ID with the 1st number being the parent_lig_ID for the derived mol
                            # Followed by Mutant 
                            # folowed by the generationnumber
                            # followed by a  unique 
                            
                            # get the unique ID (last few diget ID of the parent mol 
                            parent_lig_ID = parent_lig_ID.split(")")[-1]

                            random_id_num = random.randint(100,1000000)
                            if zinc_ID_comp_mol is None:
                                new_lig_ID = "({})Gen_{}_Mutant_{}_{}".format(parent_lig_ID,generation_num, reaction_id_number, random_id_num)
                            else:
                                new_lig_ID = "({}+{})Gen_{}_Mutant_{}_{}".format(parent_lig_ID,zinc_ID_comp_mol,generation_num,reaction_id_number, random_id_num)
                            
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

    if len(new_ligands_list) < num_mutants_to_make:
        return None
    else:
        # once the number of mutants we need is generated
        # return the list
        return new_ligands_list
#

def run_Smile_Click_for_multithread(smile, aSmilesClickChem):
    """
    This function takes a single smilestring and performs SmileClick on it.

    This is necessary for Multithreading as it is unable to execute multithread
    on a class function, but can thread a class run within a function.

    Inputs:
    :param str smile: a SMILES string
    Return:
    :returns: str result_of_run: either a smile string of a child mol or None if the reactions failed
    """
    
    result_of_run = aSmilesClickChem.run_Smile_Click(smile)

    return result_of_run
#



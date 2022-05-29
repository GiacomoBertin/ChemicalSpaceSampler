import __future__

import rdkit
from rdkit import Chem
#Disable the unnecessary RDKit warnings
rdkit.RDLogger.DisableLog('rdApp.*')

from autogrow.Operators.Filter.Filter_classes.ParentFilterClass import ParentFilter
from autogrow.Operators.Filter.Filter_classes.get_child_filter_class import get_all_subclasses
import autogrow.Operators.ConvertFiles.gypsum_dl.gypsum_dl.MolObjectHandling as MOH
from autogrow.Operators.Filter.Filter_classes.FilterClasses import *

 
def make_run_class_dict(filters_to_use):
    """
    This will retrieve all the names of every child class of the parent class ParentFilter

    Input:
    :param list filters_to_use: list of filters to be used.
            defined in vars["Chosen_Ligand_Filters"]
    
    Returns:
    :returns: dict child_dict: This dictionary contains all the names of the chosen filters as keys and the
                                    the filter objects as the items
                                returns None if no filters are specified by user
    """   

    if filters_to_use is None:
        # if the user turned off filters
        return None

    children = get_all_subclasses(ParentFilter)

    child_dict = {}
    for child in children:
        childObject = child()
        childName = childObject.get_name()

        if childName in filters_to_use:
            child_dict[childName] = childObject

    return child_dict
#

def run_filter(vars, list_of_new_ligands):
    """
    This will run a filter of the Users chosing.

    This will take a list of lists of ligands to filter.
    list_of_new_ligands = [["CCC","Zinc123],["CCCC","Zinc1234]]
   
    Input:
    :param dict vars: User variables which will govern how the programs runs
    :param list list_of_new_ligands: list of lists containing all the newly generated ligands and their names
    
    Returns:
    :returns: list ligands_which_passed_filter: a list of only the molecules which passed the filter
                                                -excludes all molecules which failed
    """
    # Get the already generated dictionary of filter objects
    Filter_Object_Dict =vars["Filter_Object_Dict"]

    # make a list of tuples for multi-processing Filter
    job_input = []            
    for smile_info in list_of_new_ligands:
        temp_tuple = tuple([smile_info, Filter_Object_Dict])
        job_input.append(temp_tuple)
    job_input = tuple(job_input)
        
    ######################################## 
    results = vars['Parallelizer'].run(job_input, run_filter_mol)

    # remove mols which fail the filter
    ligands_which_passed_filter = [x for x in results if x!=None]

    return ligands_which_passed_filter
#

def run_filter_mol(smile_info, child_dict):
    """
    This takes a smiles_string and the selected filter list (child_dict) and runs it through the selected filters.

    Inputs:
    :param list smile_info: A list with info about a ligand, the SMILES string is idx=0 and the name/ID is idx=1
                     example:   smile_info ["CCCCCCC","zinc123"]
    :param dict child_dict: This dictionary contains all the names of the chosen filters as keys and the
                                the filter objects as the items
                            Or None if User specifies no filters
    Returns:
    :returns: list smile_info: list of the smile_info if it passed the filter.
                            returns None If the mol fails a filter.
    """

    smiles_string = smile_info[0]

    mol = Chem.MolFromSmiles(smiles_string, sanitize = False) 
    # try sanitizing, which is necessary later
    mol = MOH.check_sanitization(mol)
    if mol is None:
        return None
 
    mol = MOH.try_deprotanation(mol)
    if mol is None:
        return None
 
    if child_dict is not None:
        # run through the filters
        filter_result = run_all_selected_filters(mol, child_dict)
         
        # see if passed
        if filter_result is False:
            return None
        else:
            return smile_info
    else:
        return smile_info
#

def run_filter_on_just_smiles(smile_string, child_dict):
    """
    This takes a smiles_string and the selected filter list (child_dict) and runs it through the selected filters.

    Inputs:
    :param str smile_string: A smiles_string
                     example:   smile_info ["CCCCCCC","zinc123"]
    :param dict child_dict: This dictionary contains all the names of the chosen filters as keys and the
                                the filter objects as the items
                            Or None if User specifies no filters
    Returns:
    :returns: str smile_string: smile_string if it passed the filter.
                            returns False If the mol fails a filter.
    """
    
    mol = Chem.MolFromSmiles(smile_string, sanitize = False) 
    # try sanitizing, which is necessary later
    mol = MOH.check_sanitization(mol)
    if mol is None:
        return False
 
    mol = MOH.try_deprotanation(mol)
    if mol is None:
        return False
 
    if child_dict is not None:
        # run through the filters
        filter_result = run_all_selected_filters(mol, child_dict)
         
        # see if passed
        if filter_result is False:
            return False
        else:
            return smile_string
    else:
        return smile_string
#
def run_all_selected_filters(mol, child_dict):
    """
    Iterate through all of the filters specified by the user for a single molecule
        -returns True if the mol passes all the chosen filters
        -returns False if the mol fails any of the filters

    Input:
    :param rdkit.Chem.rdchem.Mol object mol: An rdkit mol object to be tested if it passes the filters
    :param dict child_dict: This dictionary contains all the names of the chosen filters as keys and the
                                        the filter objects as the items

    Return:
    returns bol bol: True if the mol passes all the filters
                    False if the mol fails any filters
    """
    filters_failed = 0
    for child in list(child_dict.keys()):
        filter_function = child_dict[child].run_filter
        if filter_function(mol) is False:
            filters_failed = filters_failed + 1

    if filters_failed == 0:
        return True
    else:
        return False
    #

# if __name__ == "__main__":
#     print(run_test())

#     vars = {}
#     vars['Lipinski_Strict'] = True
#     vars['Lipinski_Lenient'] = False
#     vars['Ghose'] = False
#     vars['Mozziconacci'] = False
#     vars['VandeWaterbeemd'] = False
#     vars['Alternative_filters'] = ['Ghose']
#

# Dictionary and Dictionary handling functions

import __future__

import copy

import rdkit
from rdkit import Chem
#Disable the unnecessary RDKit warnings
rdkit.RDLogger.DisableLog('rdApp.*')

import autogrow.Operators.Crossover.SmilesMerge.MergeFunctions.MappingClass as MappingClass


def handle_dicts_and_select_B_groups(mol_1, mol_2, MCS_Mol):
    """
    this takes 3 rdkit.Chem.rdchem.Mol objects 1 for lig_1,lig_2, 
    and the common core(MCS_Mol). It creates all the necessary dictionaries, mapping,
    and selects the ligands that will be added to make the final molecule.
    
    Inputs:
    :param rdkit.Chem.rdchem.Mol mol_1: rdkit mol for ligand 1
    :param rdkit.Chem.rdchem.Mol mol_2: rdkit mol for ligand 2
    :param rdkit.Chem.rdchem.Mol MCS_Mol: rdkit mol for shared common core between mol_1 and mol_2

    Returns:
    :returns: list Rs_chosen_smiles: smiles strings for the the R groups which correspond to the chosen B's
                                    returns None if it fails
    """

    # Confirm that MCS_mol can be replaced in mol_1 and mol_2
    # around 0.8% of the time this function fails so we will filter this 1st
    they_pass = check_replace_mol(mol_1, mol_2, MCS_Mol)
    if they_pass == False:
        return None

    R_Smiles_dict_1, B_to_R_master_dict_1, B_to_Anchor_master_dict_1 = mol_handling_of_fragmenting_labeling_and_indexing(mol_1, MCS_Mol, 1)
    # check that this worked (ie if it failed they will return None)
    if R_Smiles_dict_1 is None:
        return None
    
    R_Smiles_dict_2, B_to_R_master_dict_2, B_to_Anchor_master_dict_2 = mol_handling_of_fragmenting_labeling_and_indexing(mol_2, MCS_Mol, 2)
    # check that this worked (ie if it failed they will return None)
    if R_Smiles_dict_2 is None:
        return None

    # Merg B_to_Anchor_master_dict into 1 master dictionary of B_to_anchors
    # the keys will be all be unique so we can add these dictionaries together
    # without worry of overrighting an entry. We will invert the dict after to
    # get anchors as the keys and the B's as the items
    # example B_to_Anchor_master {'1B1':[10008,10007],'1B2':[10000],'1B3':[10006],
    #                '2B3':[10006,10007],'2B2':[10000],'2B1':[10008]}
    B_to_Anchor_master = B_to_Anchor_master_dict_1
    for i in list(B_to_Anchor_master_dict_2.keys()):
        B_to_Anchor_master[i] = B_to_Anchor_master_dict_2[i]

    # Invert B_dictionary to produce a master I dicitonary
    # example Anchor_to_B_master = {10008:['1B1','2B1'],10000:['1B2','2B2']}
    Anchor_to_B_master = invert_dictionary(B_to_Anchor_master)

    Bs_chosen = MappingClass.run_mapping(B_to_Anchor_master, Anchor_to_B_master)

    # Get the R groups which correspond to the chosen B's
    # ['1R1', '1R5', '2R2']
    Rs_chosen = get_Rs_chosen_from_Bs(Bs_chosen, B_to_R_master_dict_1, B_to_R_master_dict_2)


    # Get the smiles strings for the the R groups which correspond to the chosen B's
    Rs_chosen_smiles = get_Rs_chosen_smiles(Rs_chosen, R_Smiles_dict_1, R_Smiles_dict_2)
    
    return Rs_chosen_smiles
#
  
def mol_handling_of_fragmenting_labeling_and_indexing(mol, MCS_Mol, lig_number):
    """
    This takes an rdkit mol for a ligand and 1 for the MCS_mol
    It fragments the ligand by replacing the MCS. and it determines
    which anchors are in each fragment. 
    These fragments are our R-groups and the assignment of anchors 
    is how we determine which R-group goes where relative to the MCS.
    
    lig_number  int    is the number of the ligand that is mol
                        ie if mol is mol_1 lig_number = 1
                        ie if mol is mol_2 lig_number = 2


    Inputs:
    :param rdkit.Chem.rdchem.Mol mol: an rdkit mol (either mol_1 or mol_2)
    :param rdkit.Chem.rdchem.Mol MCS_Mol: rdkit mol for shared common core between mol_1 and mol_2
    :param int lig_number: an int either 1 or 2 for (mol_1 or mol_2 respectively)

    Returns:
    :returns: dict R_Smiles_dictionary: a dictionary of the R-groups which branch off the common core
                                        keys are the R-groups; items are the SMILES strings of that R-groups 
                                        returns None if it fails
                                        ie) {'1R1': '[10003*][1007N]=[1013O]', '1R2': '[10000*][1011CH2]=[1008O]'}
    :returns: dict B_to_R_master_dict: A dictionary which tracks the R groups which belong to a B-group
                                        keys are the B-groups; items are the R-groups which belong to the B-group
                                        returns None if it fails
                                        ie) {'1B1': ['1R2'], '1B2': ['1R1']}
    :returns: dict B_to_Anchor_master_dict: A dictionary which tracks the iso label of the anchor atoms for B-group
                                        keys are the B-groups; items are the iso label of the anchor atoms for B-group
                                        returns None if it fails
                                        ie) {'1B1': [10000], '1B2': [10003]}
    """
    ################################################ 
    
    # Find which MCS Atoms Rs branch from
    # Function to find all neighbors for a set of molecules touching an Isolabeled core  
    MCS_touches = get_atoms_touch_MCS(mol)  
    # invert dictionary
    lig_R_atoms_touch_MCS = invert_dictionary(MCS_touches)

    ###########################################################
    ###########################################################
    # remove the Core atoms from each ligand this gives us the R-groups
    Replace_core = r_group_list(mol, MCS_Mol)
    if Replace_core is None:
        # Replace_core failed to handle fragments"
        return None, None, None
        

    Replace_core = replace_core_mol_dummy_atoms(mol, MCS_Mol, Replace_core)
    if Replace_core is None:
        # Replace_core failed to handle fragments"
        return None, None, None


    ####### A single anchor (isotope label) may now be present multiple times in
    ############## the Replace_core_mols as they are fragmented 
    # replace_frag_w_anchor_isolabels can return a None if failed so lets check for None before we move on
    if Replace_core is None:
        # Replace_core failed to handle fragments"
        return None, None, None
    
    # MAKE NEW MOL FRAGS FROM LABELED REPLACE_CORE
    mol_frags = Chem.GetMolFrags(Replace_core, asMols = True, sanitizeFrags = False)
    list_R_groups = []
    i = 0
    while i < len(mol_frags):
        val = Chem.MolToSmiles(mol_frags[i],isomericSmiles = True)
        list_R_groups.append(val)
        i = i + 1
        
    # Generate all the R-libraries with full R-groups 
    # using the index of its respective Lig
    # R_chain_dictionary is the master dictionary for R-groups
    R_chain_dictionary, R_Smiles_dictionary = R_groups_dict(mol_frags, lig_number)
    
    # R_dict is a secondary dictionary for searching I's in R's
    # this dictionary is limited to only the R-group and anchor(I)
    R_dict = get_R_dict(R_chain_dictionary, lig_R_atoms_touch_MCS)       
    
    # make inversion of R_dict
    # keys are the Anchor atom iso_labels while the items are the R-group numbers
    # which are attached to that anchor atom
    # ie {10008: ['2R3'], 10000: ['2R2'], 10006: ['2R1'], 10007: ['2R1']}
    I_dict = invert_dictionary(R_dict)
    """
    B-dictionaries:
    Ligmerge will randomly select R-groups to append to a shared common core from 2 seperate ligands
    but so anchor atoms in the common core may have more than 1 R-group attached to it.
            ie. if an anchor carbon has di-methyls attached to it (which are not part of the shared core)
                are these di-methyls 2 seperate R-groups or is the contextual chemical enviorment created by
                having both methyls different from having one alone and thus should be treated as a single R-group
                which just happen to branch. 
                This author would argue that context is important here and so this version of Ligmerge 
                treats anything attached to an anchor atom in the common core as a singular contextual
                functional group which shall be refered to as a B-groups.
                    ie. a B-group consists of 1 or more R-groups which are attached to an anchor atom in the
                        shared common core. This makes a significant difference in how we select for which pieces
                        are added to build our child molecule. Additionally this has significance in the 
                        decision tree use to build a child molecule. An R/B group can be connected to multiple anchor
                        atoms so once we chose a B group we will need to know which anchor atoms are affected by that
                        decision. This is something handled more in the Mapping class, but this is why 
                        the nominclature change from R-groups to B-groups and why the next several steps are important.
    make_B_dictionaries (B is the name we gave to R-groups sets)
    """
    B_to_R_master_dict, B_to_Anchor_master_dict = make_B_dic(I_dict, R_dict, lig_number)

    return R_Smiles_dictionary, B_to_R_master_dict, B_to_Anchor_master_dict
#

def check_replace_mol(mol_1,mol_2, MCS_mol):
    """
    Confirm that MCS_mol can be replaced in mol_1 and mol_2
    around 0.8% of the time this function fails so we will filter this 1st
    Inputs:
    :param rdkit.Chem.rdchem.Mol mol_1: an rdkit mol 
    :param rdkit.Chem.rdchem.Mol mol_2: an rdkit mol 
    :param rdkit.Chem.rdchem.Mol MCS_mol: rdkit mol for shared common core between mol_1 and mol_2
    Returns:
    :returns: bool True/False: Returns True if it passes for both mol_1 and mol_2
                            returns False if either fails.
    
    
    """
    temp = r_group_list(mol_1, MCS_mol)
    if temp == None:
        return False
    temp = r_group_list(mol_2, MCS_mol)
    if temp == None:
        return False
    return True
  
###########################################################
###########################################################
# HANDLE THE OBTAINING THE R-Groups for a given mol

def r_group_list(mol, core_mol):
    """
    This takes a mol and the common core and finds all the R-groups by replacing the 
    atoms in the ligand (which make up the common core) with nothing.
    
    This fragments the ligand and from those fragments we are able to determine what our
    R-groups are. for any common core atom which touched the fragment a * 
    will replace that atom in the fragments.


    Inputs: 
    :param rdkit.Chem.rdchem.Mol mol: an rdkit molecule
    :param rdkit.Chem.rdchem.Mol core_mol: an rdkit molecule for the shared common core

    Returns:
    :returns: rdkit.Chem.rdchem.Mol Replace_core_mol: an rdkit molecule with the common core removed from a ligand
                                                fragments the mol which can be used to make lists of R-groups
    """
    # This returns all the mol frags for a particular compound against the core molecule
    Replace_core_mol = Chem.ReplaceCore(mol, core_mol,labelByIndex = True,replaceDummies = True,requireDummyMatch = False)
    
    if len(Replace_core_mol.GetAtoms()) == 0:
        # This means that the mol either did not contain the core_mol or
        # the core_mol is the same mol as the mol
        # ie) if mol_string ="[10000N-]=[10001N+]=[10002N][10003CH]1[10004O][10005CH]([10006CH2][10007OH])[10008CH]([10013OH])[10009CH]([10012OH])[10010CH]1[10011OH]"
        #   and core_string ="[10000NH]=[10001N+]=[10002N][10003CH]1[10004O][10005CH]([10006CH2][10007OH])[10008CH]([10013OH])[10009CH]([10012OH])[10010CH]1[10011OH]"
        #   the only difference is the H's which means it can be replaced within because its the same mol
        #   This is rare but does occur.
        return None

    return Replace_core_mol
#
 
def replace_core_mol_dummy_atoms(mol, MCS, Replace_core_mol):
    """
    This function will replace the dummy atoms (*) with the isotope label from the core atoms.
    example:
        mol = Chem.MolFromSmiles("[10000N-]=[10001N+]=[10002N][10003CH2][2004CH]1[2005NH2+][2006CH2][2007CH]([2008OH])[2009CH]([2010OH])[2011CH]1[2012OH]")
        MCS = Chem.MolFromSmiles("[10003CH3][10002N]=[10001N+]=[10000NH]")
        Replace_core = Chem.MolFromSmiles("[3*][2004CH]1[2005NH2+][2006CH2][2007CH]([2008OH])[2009CH]([2010OH])[2011CH]1[2012OH]")
        
        resulting Replace_core = '[10003*][2004CH]1[2005NH2+][2006CH2][2007CH]([2008OH])[2009CH]([2010OH])[2011CH]1[2012OH]'

    Inputs: 
    :param rdkit.Chem.rdchem.Mol mol: an rdkit molecule
    :param rdkit.Chem.rdchem.Mol MCS: an rdkit molecule for the shared common core
    :param rdkit.Chem.rdchem.Mol Replace_core_mol: the mol with the MCS anchors labeled
                with * and an isotope label of the idx of the core anchor atom
            
    Returns:
    :returns: rdkit.Chem.rdchem.Mol Replace_core_mol: an rdkit molecule
            with the common core removed from a ligand fragments the mol which 
            can be used to make lists of R-groups. The * atoms will be isotope labeled with
            the isotope label from the core.
    """

    Replace_core_mol_original = copy.deepcopy(Replace_core_mol)
    anchor_dict = {}
    anchor_to_set_dict = {}
    for atom in Replace_core_mol.GetAtoms():
        if atom.GetAtomicNum()==0:
            anchor_iso = atom.GetIsotope() + 10000
            neighbors= atom.GetNeighbors()
            tmp = []
            for n_atom in neighbors:
                tmp.append(n_atom.GetIsotope())
            anchor_dict[anchor_iso] = tmp
                              
            anchor_to_set_dict[atom.GetIdx()] = anchor_iso
        
    for idx in list(anchor_to_set_dict.keys()):

        atom = Replace_core_mol.GetAtomWithIdx(idx)
        anchor_iso = anchor_to_set_dict[idx]
        atom.SetIsotope(anchor_iso)

    return Replace_core_mol

###########################################################
###########################################################

  
def R_groups_dict(mol_frags, lig_number_for_multiplier):
    """
    given a set of mol_frags and the ligand_number (ie. 1 for mol_1 and 2 for mol_2)
    this will make dictionaries of all the Rgroup and all the smiles for each Rgroup
    
    Input
    :param rdkit.Chem.rdchem.Mol mol_frags: a rdkit mol containing fragments
    :param int lig_number_for_multiplier: an int either 1 for mol_1 or 2 for mol_2, used to make
                                            labels which are tracable to the ligand being used
    Returns:
    :returns: dict R_chain_dictionary: a dictionary with the R-groups and the anchor atoms they connect to
                                        ie) {'1R1':[13,14],'1R2':[21,22],'1R3':[25]}
    :returns: dict R_Smiles_dictionary: a dictionary with the R-groups and the SMILES strings of those groups
                                        ie {'1R1':'[1*]:[1013c]([1020H])[1014c]([1019H])[1015c]([1018H])[1016c](:[2*])[1017H]',
                                        '1R2':'[3*][1024C]([1026H])([1027H])[1023N] = [1022N+] = [1021N-]',
                                        '1R3':'[4*][1025O][1029H]'}
    """
    num_frags = len(mol_frags)
    i = 0
    R_chain_dictionary = {}
    R_Smiles_dictionary = {}
    k = int(lig_number_for_multiplier)
    while i < num_frags:
        frag = mol_frags[i]
        R_list_idx = []
        r_list_temp = []
        r_list_smiles = []
        r_list_smiles = Chem.MolToSmiles(frag, isomericSmiles = True)
        for atoms in frag.GetAtoms():
            iso = atoms.GetIsotope()
            if 3000> iso >100:
                r_list_temp.append(iso - (1000*k))
                atoms.SetIsotope(0)
            if iso >3000:
                name = "I{}".format(iso-10000)
                r_list_temp.append(iso)
            lignum_R_Rnum = "{}R{}".format(k,i+1)
            R_chain_dictionary[lignum_R_Rnum] = r_list_temp
            R_Smiles_dictionary[lignum_R_Rnum] = r_list_smiles
        i = i+1

    return R_chain_dictionary, R_Smiles_dictionary
#
  
def get_R_dict(R_chain_dict, Lig_R_atom_touch_MCS):
    """
    This will take the R_chain_dict and the dict of all the atoms which touch the core
    and return a dict of Rs groups as keys and their nodes as values
    Inputs: 
    :param dict R_chain_dict: dict of all the atom isolabels for in an R-group
                                        keys are R-groups;  items are iso-labels of atoms in the R-group
                                        ie) {'1R1': [3, 4, 5, 6, 7, 8, 9, 10, 11, 10000]}
    :param dict Lig_R_atom_touch_MCS:   dict of all the atoms which directly touch the core and what anchor they touch
                                            keys are atom isolabels of atoms touching an anchor;   
                                            items are iso-labels of anchor atoms
                                        ie) {3: [10000]}
    Returns:
    :returns: dict Rs_dict:  dictionary of R-groups and anchor atoms they are connected to
                                keys are R-groups
                                items are isolabel of anchor atoms
                            ie) {'1R1': [10000]}
    """
    Rs_dict = {}
    for key in list(R_chain_dict.keys()):
        temp_R_list = R_chain_dict[key]
        node_list = []
        for atom in R_chain_dict[key]:
            for key_id in list(Lig_R_atom_touch_MCS.keys()):
                if atom == key_id:
                    for x in Lig_R_atom_touch_MCS[key_id]:
                        node_list.append(x)
                    Rs_dict[key] = node_list 

    return Rs_dict    
#

##########
# Mapping functions and finding neighbors
######### 
def get_idx_using_unique_iso(mol, iso_val):
    """
    This function takes a value for an isotope label and finds the atom in a mol
    which has that isotope label. This assumes there is only 1 atom in a mol
    with the same isotope value
    
    Input:
    :param rdkit.Chem.rdchem.Mol mol: a molecule whose atom's have unique isotope labels
    :param int iso_val:  the isotope value to search by
    
    Returns:
    :returns: int idx:  the Idx index number of the atom whose isotope label is the same as
                        iso_val
                        Returns None if iso_val not in mol
    """
    for atom in mol.GetAtoms():
        if atom.GetIsotope() == iso_val:
            idx=atom.GetIdx()
            return idx
    return None
#

def make_B_dic(I_dictionary, R_dict_num, lig_number):
    """
    This generates the dictionaries for the B-groups. 
    one is to track the R-groups which a B-group represents (this is the B_to_R_master_dict)
    one is to track the anchor atoms a B-group branches from (this is the B_to_Anchor_master_dict)
        
    Inputs:
    :param dict I_dictionary:dictionary for R groups bound to nodes (aka I's)
                                    ie) {'10008':[1R1,1R2],'10009':[1R2,1R3]}
    :param dict R_dict_num: dictionary for anchors which are attached to an R group
                                    ie) {'1R1':[10008],'1R2':[10008,10009],'1R3':[10009]}
    :param int lig_number: an int either 1 or 2 for (mol_1 or mol_2 respectively)

    Returns:
    :returns: dict B_to_R_master_dict: key is unique B-name and the R-groups it represents.
                        example {'1B1':['1R1'],'1B2':['1R2','1R3','1R4'],'1B3': ['1R5']}
    :returns: dict B_to_Anchor_master_dict: key is unique B-name and items are anchors that B connects to.      
                        example {'1B1':[10008,10007],'1B2':[10000],'1B3':[10006]}
       
    """
    k = lig_number
    B_to_R_master_dict = {}
    B_to_Anchor_master_dict = {}
    counter = 1
    anchor_list = list(I_dictionary.keys())
    # anchor_list = [10008, 10000, 10006, 10007]
    
    while len(anchor_list) > 0:
        anchor = anchor_list[0]
        B_key = "{}B{}".format(k, counter)
        temp_R_list = []
        temp_anchor_list = []
        
        for Rs in I_dictionary[anchor]:
            # example Rs in I_dictionary[anchor]: '1R1')
            temp_R_list.append(Rs)
            R_dict_Is = R_dict_num[Rs]
            for I in R_dict_Is:
                # example Rs in I_dictionary[anchor]: '1R1')
                temp_anchor_list.append(I)
        # remove any redundancies in the list by list(set(list_of_things))
        temp_anchor_list = list(set(temp_anchor_list))  
        temp_R_list = list(set(temp_R_list))    
        
        # make new B-group entry in the dictionaries
        B_to_R_master_dict[B_key] = temp_R_list    # This B-represents these R-groups
        B_to_Anchor_master_dict[B_key] = temp_anchor_list # This B connects to these anchor atoms
        
        counter = counter+1
        
        # make a list of atoms to remove in the next iteration if they are 
        # in both the temp_anchor_list and anchor_list
        for i in temp_anchor_list:
            if i in anchor_list:
                anchor_list.remove(i)
    
    # example B_to_R_master_dict:{'1B1':['1R1'],'1B2':['1R2','1R3','1R4'],'1B3': ['1R5']}
    # example B_to_Anchor_master_dict:{'1B1':[10008,10007],'1B2':[10000],'1B3':[10006]}
    return B_to_R_master_dict, B_to_Anchor_master_dict
# 

def invert_dictionary(old_dic):
    """
    This will invert any dictionary so that the keys are the values and the values are the keys.
    Inputs: 
    :param dict old_dic: a dictionary to invert 
    Return: 
    :returns: dict inverted_dic: old_dict dict inverted so the keys are the items and the items are the keys
    """
    
    # inverted_dic = {}
    # for k, v in old_dic.iteritems():
    # keys = inverted_dic.setdefault(v, [])
    # keys.append(k)
    values = set([a for b in list(old_dic.values()) for a in b])
    values = list(values)
    inverted_dic = dict((new_key, [key for key, value in list(old_dic.items()) if new_key in value]) for new_key in values)
     
    return inverted_dic
#

def get_atoms_touch_MCS(mol):
    """
    Function to find all neighbors for a set of molecules touching 
    Isolabeled core atoms
    
    Inputs: 
    :param rdkit.Chem.rdchem.Mol mol: isolabeled with atoms in the core having isotope
                                    labels set as their idx number + 10000
                                    and atoms not shared in the common core isotope
                                    labels set as 
                                    for lig_1: atom idx number + 1000  
                                    for lig_1: atom idx number + 2000  
                                    
    :returns: 
    :returns: dict MCS_touches dict:  a dictionary with keys being the isotope label of core atoms
                                    and the items being the idx's of all non-core atoms which touch it
                                    If a core atom touch no non-core atoms it will not be added to
                                    the dictionary
    """
    MCS_touches = {}
    all_atoms = mol.GetAtoms()
    
    for atom in all_atoms:
        # find all atoms in the mol which are also in the Core using iso labels
        
        iso = atom.GetIsotope()
        if iso > 9999:
            # then its a core atom
            neighbors = atom.GetNeighbors()
            values = []

            for neighbor_atom in neighbors:
                # compile list of the Indexes of all neighbor of the atom
                Idx_neighbor = neighbor_atom.GetIdx()        

                # Select for only neighbors which are not in the core using iso
                iso_neighbor_x = neighbor_atom.GetIsotope()
                if iso_neighbor_x < 9999:
                    # Then this is an atom which is not in the core but touches the core
                    idx_of_neighbor = neighbor_atom.GetIdx()
                    values.append(idx_of_neighbor)
                    MCS_touches[iso] = values   
        
    return MCS_touches
#
 
##########
# Handling after B-groups are chosen
########## 
def get_Rs_chosen_from_Bs(Bs_chosen, B_to_R_master_dict_1, B_to_R_master_dict_2):
    """
    this function returns a list of R-groups chosen based on the list of chosen B's.
    It requires the B_to_R_master_dictt for both ligands to function.

    Inputs: 
    :param list Bs_chosen: A list of the chosen B-groups
                    ie) ['1B1', 1B2', '2B3']
    :param dict B_to_R_master_dict_1: a Dictionary to reference B and R-groups from mol_1 
                        keys are names of B-groups; items are R-groups that 
                            a B-group represents
                        ie) {'1B1':['1R1'],'1B2':['1R2','1R3','1R4'],'1B3': ['1R5']}
    :param dict B_to_R_master_dict_2: a Dictionary to reference B and R-groups from mol_2
                        keys are names of B-groups; items are R-groups that 
                            a B-group represents
                        ie) {'2B1':['2R1'],'2B2':['2R2','2R3','2R4'],'2B3': ['2R5','2R6]}
    
    Returns:
    :returns: list Rs_chosen: a list containing all the R-groups represented by the chosen B-groups
                                ie) ['1R1', '1R2', '1R3','1R4', '2R5', '2R6']
    """
    Rs_chosen = []
    for B in Bs_chosen:
        Rs_for_the_B = []
        lig_number = B[0]
        B_number = B[2]
        if lig_number == str(1):
            for i in B_to_R_master_dict_1[B]:
                Rs_for_the_B.append(i)      

        elif lig_number == str(2): 
            for i in B_to_R_master_dict_2[B]:
                Rs_for_the_B.append(i)      
        for i in Rs_for_the_B:
            Rs_chosen.append(i)
    
    # Rs_chosen looks like ['1R1', '1R5', '2R2']  
    return Rs_chosen  
#

def get_Rs_chosen_smiles(Rs_chosen, R_Smiles_dict_1, R_Smiles_dict_2):
    """
    This function returns a list of SMILES strings for every R-group chosen.
    It requires the R_smile_dictionary for both ligands to function.
    
    Inputs:
    :param list Rs_chosen: A list of the chosen R-groups which will be used to generate a new mol
                                ie) ['2R2', '1R1']
    :param dict R_Smiles_dict_1: A dictionary which has can find the SMILES string for each R-group of Ligand 1
                                ie) {'1R1': '[10006*][1009N]=[1008N+]=[1007N-]'}

    :param dict R_Smiles_dict_2: A dictionary which has can find the SMILES string for each R-group of Ligand 2
                                ie) {'2R2': '[10006*][2009OH]', '2R1': '[10003*][2007CH2][2008OH]'}

    :returns: 
    :returns: list Rs_chosen_smiles: A list of all the SMILES string which are to be added to
                                    make the child ligand. Each SMILES is a sublist.
                                ie)[['[10006*][1009N]=[1008N+]=[1007N-]'],['[10006*][2009OH]']]
    """
    Rs_chosen_smiles = []
    for R in Rs_chosen:
        Rs_for_the_R = []
        lig_number = R[0]
        R_number = R[2]
        if lig_number == str(1):
            Rs_for_the_R.append(R_Smiles_dict_1[R])      
        elif lig_number == str(2): 
            Rs_for_the_R.append(R_Smiles_dict_2[R])
                
        Rs_chosen_smiles.append(Rs_for_the_R)

    return Rs_chosen_smiles  
#


#  BUT good function
def map_all_neighbors_w_iso(mol):
    """
    maps any neighbors by IDx, return dictionary of all atom indexes in a molecule and
    the index of any atom it touches as the values in the dictionary
    
    Inputs: 
    :param rdkit.Chem.rdchem.Mol mol: rdkit molecule with isolabels
    
    Returns:
    :returns: dict atom_touches dict: dict of all atoms in the mol and everything they touch.
                                    the keys and items are Isotope labels of the atom
                                    they represent
                examples of non_core atoms in dict: 1023: [1022], 1024: [1023, 10000],
                examples of core atoms in dict: 10000:[1024, 10001, 1028],10001:[10000, 10002],
    """
    atom_touches = {}
    all_atoms = mol.GetAtoms()
    for atom in all_atoms:
        neighbors = atom.GetNeighbors()
        iso = atom.GetIsotope()
        values = []
        for neighbor_atom in neighbors:
            neighbor_iso = neighbor_atom.GetIsotope()
            values.append(neighbor_iso)    
        atom_touches[iso] = values
    return atom_touches
#
#####################


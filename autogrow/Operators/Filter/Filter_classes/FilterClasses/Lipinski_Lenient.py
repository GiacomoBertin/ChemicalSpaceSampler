"""Lipinski Lenient
This runs a Lenient Lipinski filter.
Lipinski filter refines for orally available drugs. 
It filters molecules by Molecular weight (MW),
the number of hydrogen donors, the number hydrogen acceptors,
and the logP value.

To pass the Lipinski filter a molecule must be
    MW: Max 500 dalton
    Number of H acceptors: Max 10
    Number of H donors: Max 5
    logP Max +5.0

If you use the Lipinski Filter please cite:
C.A. Lipinski et al.
Experimental and computational approaches to estimate 
solubility and permeability in drug discovery 
and development settings
Advanced Drug Delivery Reviews, 46 (2001), pp. 3-26
"""
import __future__

import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.Lipinski as Lipinski
import rdkit.Chem.Crippen as Crippen
import rdkit.Chem.Descriptors as Descriptors
#Disable the unnecessary RDKit warnings
rdkit.RDLogger.DisableLog('rdApp.*')

from autogrow.Operators.Filter.Filter_classes.ParentFilterClass import ParentFilter


class Lipinski_Lenient(ParentFilter):
    """
    This runs a Lenient Lipinski filter.
    Lipinski filter refines for orally available drugs. 
    It filters molecules by Molecular weight (MW),
    the number of hydrogen donors, the number hydrogen acceptors,
    and the logP value.

    This is a Lenient Lipinski which means a ligand 
    is allowed one violation exception to 
    the Lipinski Rule of 5 restraints.

    To pass the Lipinski filter a molecule must be
        MW: Max 500 dalton
        Number of H acceptors: Max 10
        Number of H donors: Max 5
        logP Max +5.0

    If you use the Lipinski Filter please cite:
    C.A. Lipinski et al.
    Experimental and computational approaches to estimate 
    solubility and permeability in drug discovery 
    and development settings
    Advanced Drug Delivery Reviews, 46 (2001), pp. 3-26

    Inputs:
    :param class ParentFilter: a parent class to initialize off
    """        
    def run_filter(self, mol):
        """
        This runs the Lenient Lipinski filter.
        Lipinski filter refines for orally available drugs. 
        It filters molecules by Molecular weight (MW),
        the number of hydrogen donors, the number hydrogen acceptors,
        and the logP value.
        
        This is a Lenient Lipinski which means a ligand 
        is allowed one violation exception to 
        the Lipinski Rule of 5 restraints.

        To pass the Lipinski filter a molecule must be
            MW: Max 500 dalton
            Number of H acceptors: Max 10
            Number of H donors: Max 5
            logP Max +5.0
            
        Inputs:
        :param rdkit.Chem.rdchem.Mol object mol: An rdkit mol object to be tested if it passes the filters
        Returns:
        :returns: bool bool: True if the mol passes the filter; False if it fails the filter
        """    
        violation_counter = 0

        ExactMWt = Descriptors.ExactMolWt(mol)
        if ExactMWt > 500:
            violation_counter = violation_counter + 1
        
        num_H_bond_donors = Lipinski.NumHDonors(mol)
        if num_H_bond_donors > 5:
            violation_counter = violation_counter + 1
        
        num_H_bond_acceptors = Lipinski.NumHAcceptors(mol)
        if num_H_bond_acceptors > 10:
            violation_counter = violation_counter + 1
        
        mol_logP = Crippen.MolLogP(mol)
        if mol_logP > 5:
            violation_counter = violation_counter + 1
        
        if violation_counter < 2:
            return True
        else:
            return False
    #
#

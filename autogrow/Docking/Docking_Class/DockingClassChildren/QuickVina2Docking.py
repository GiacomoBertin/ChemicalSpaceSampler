"""
The child classes from ParentExample
"""
import __future__

import os
import sys
import glob
import string
import subprocess
import time

import autogrow.Docking.Delete_failed_mol as Deletion
from autogrow.Docking.Docking_Class.DockingClassChildren.VinaDocking import VinaDocking


class QuickVina2Docking(VinaDocking):
    """
    RUN QuickVina2 Docking
    Inputs:
    :param class ParentDocking: Parent docking class to inherit from 
    """    
    def __init__(self, vars, receptor_file):
        """
        get the specifications for Vina/QuickVina2 from vars
        load them into the self variables we will need
        and convert the receptor to the proper file format (ie pdb-> pdbqt)

        Inputs:
        :param dict vars: Dictionary of User variables
        :param str receptor_file: the path for the receptor pdb
        """
        self.vars = vars

        # VINA SPECIFIC VARS
        receptor_file = vars['filename_of_receptor']
        mgl_python = vars['mgl_python']
        receptor_template = vars['prepare_receptor4.py']
        number_of_processors = vars['number_of_processors']
        docking_executable = vars['docking_executable']

        ########################### 

        # convert Receptor from PDB to PDBQT
        self.convert_receptor_pdb_files_to_pdbqt(receptor_file, mgl_python, receptor_template, number_of_processors)

        self.receptor_PDBQT_file = receptor_file + "qt"

        self.vars['docking_executable'] = self.get_docking_executable_file(self.vars)
    #

    def get_docking_executable_file(self, vars):
        """
        This retrieves the docking executable files Path. 

        Inputs:
        :param dict vars: Dictionary of User variables
        Returns:
        :returns: str docking_executable: String for the docking executable file path
        """
        # This must already be true if we are here vars["Dock_choice"] == "QuickVina2Docking"
        
        if vars["docking_executable"] is None: 
            # get default docking_executable for QuickVina2
            script_dir = str(os.path.dirname(os.path.realpath(__file__)))
            docking_executable_directory = script_dir.split(os.sep + "Docking_Class")[0] + os.sep + "Docking_Executables" + os.sep
    
            docking_executable = docking_executable_directory + "QVina02{}qvina2.1".format(os.sep) 
            
        else:
            # if user specifies a different QuickVina executable
            docking_executable = vars["docking_executable"]
          
        return docking_executable
    # 


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

import rdkit
import rdkit.Chem as Chem

#Disable the unnecessary RDKit warnings
rdkit.RDLogger.DisableLog('rdApp.*')

import autogrow.Docking.Delete_failed_mol as Delete
import autogrow.Docking.Ranking.Ranking_mol as Ranking
from autogrow.Docking.Docking_Class.ParentDockClass import ParentDocking
import autogrow.Operators.ConvertFiles.gypsum_dl.gypsum_dl.MolObjectHandling as MOH
import autogrow.Docking.Scoring.Execute_scoring_mol as Scoring

class VinaDocking(ParentDocking):
    """
    RUN VINA DOCKING
    Inputs:
    :param class ParentDocking: Parent docking class to inherit from
    """    
    def __init__(self, vars, receptor_file):
        """
        get the specifications for Vina from vars
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

    def run_ligand_handling_for_docking(self, pdb_file):
        """
        this function converts the ligands from PDB to PDBQT format.
        Returns NONE if it worked and the name if it failed to convert.

        Inputs:
        :param str pdb_file: the pdb file of a ligand to format, dock and score        
        Returns:
        :returns: str smile_name: name of smiles if it failed to dock
                                returns None if it docked properly
        """
        # convert ligands to pdbqt format
        # log("\nConverting ligand PDB files to PDBQT format...")
        did_it_convert, smile_name = self.convert_ligand_pdb_file_to_pdbqt(pdb_file)
        
        if did_it_convert is False:
            # conversion failed
            return smile_name
        else:
            return None      
    #

    def run_dock(self, pdbqt_filename):
        """
        this function runs the docking. Returns None if it worked and the name if it failed to dock.

        Inputs:
        :param str pdbqt_filename: the pdbqt file of a ligand to dock and score        
        Returns:
        :returns: str smile_name: name of smiles if it failed to dock
                                returns None if it docked properly
        """

        # log("Docking compounds using AutoDock Vina...")
        self.dock_ligand(pdbqt_filename)
        
        # check that it docked       
        pdb_filename = pdbqt_filename.replace("qt","")

        did_it_Dock, smile_name = self.check_docked(pdb_filename)
        
        if did_it_Dock is False:
            # Docking failed

            if smile_name == None:
                print("Missing pdb and pdbqt files for : ", pdbqt_filename)

            
            return smile_name

        return None
    #

    ####################################### 
    # STUFF DONE BY THE INIT       
    ##########################################   
    def get_docking_executable_file(self, vars):
        """
        This retrieves the docking executable files Path. 
        Inputs:
        :param dict vars: Dictionary of User variables
        Returns:
        :returns: str docking_executable: String for the docking executable file path
        """
        if vars["docking_executable"] is None: 
            # get default docking_executable for vina
            script_dir = str(os.path.dirname(os.path.realpath(__file__)))
            docking_executable_directory = script_dir.split(os.sep + "Docking_Class")[0] + os.sep + "Docking_Executables" + os.sep

            if sys.platform == "linux" or sys.platform == "linux2":
                # Use linux version of Autodock Vina
                docking_executable = docking_executable_directory + "Vina" + os.sep + "autodock_vina_1_1_2_linux_x86" + os.sep + "bin" + os.sep + "vina"
        
            elif sys.platform == "darwin":
                # Use OS X version of Autodock Vina
                docking_executable = docking_executable_directory + "Vina" + os.sep + "autodock_vina_1_1_2_mac" + os.sep + "bin" + os.sep + "vina"

            elif sys.platform == "win32":
                # Windows...
                raise Exception("Windows is currently not supported")
            else:
                raise Exception("This OS is currently not supported")

        else:
            # if user specifies a different vina executable
            docking_executable = vars["docking_executable"]

        return docking_executable
    #

    def convert_receptor_pdb_files_to_pdbqt(self, receptor_file, mgl_python, receptor_template, number_of_processors):
        """
        Make sure a PDB file is properly formatted for conversion to pdbqt
        
        Inputs:
        :param str receptor_file:  the file path of the receptor
        :param str mgl_python: file path of the pythonsh file of mgl tools
        :param str receptor_template: the receptor4.py file path from mgl tools.
        :param int number_of_processors: number of processors to multithread
        """    
        count = 0
        
        while not os.path.exists(receptor_file + "qt"): 
        
            count = count + 1
            if count > 10000:
                print("ERROR: I've tried 10,000 times to convert the file \"" + receptor_file + "\" to the PDBQT format. Aborting program...")
                raise ValueError("ERROR: I've tried 10,000 times to convert the file \"" + receptor_file + "\" to the PDBQT format. Aborting program...")
            
            # make sure the receptors have been converted to PDBQT. If not, do the conversion.
            receptors = [receptor_file] 
            need_to_covert_receptor_to_pdbqt = []
            for filename in receptors: 
                if not os.path.exists(filename + "qt"): need_to_covert_receptor_to_pdbqt.append(filename)
            
            # This should only be 1 receptor. If Autogrow is expanded to handle multiple receptor, one will need 
            # Multiprocess this.
            # Fix this to 1 processor, so no overwriting issues, but could be expanded if we ever wanted to do multiple receptors

            # create a file to run the pdbqt
            for i in need_to_covert_receptor_to_pdbqt:

                output = self.prepare_receptor_multiprocessing(mgl_python, receptor_template, i)
    #

    def prepare_receptor_multiprocessing(self, mgl_python, prepare_script, mol_filename):
        """
        This prepares the receptor for multiprocessing. 
        Inputs:
        :param str mgl_python: file path of the pythonsh file of mgl tools
        :param str prepare_script:  the file path for the mgltool receptor prep file receptor4
        :param str mol_filename:  the file path of the receptor
        """
        command = mgl_python + " " + prepare_script + " -r " + mol_filename + " -o " + mol_filename + "qt"
        try:
            os.system(command)
        except:
            raise Exception("Could not convert receptor with MGL_tools")
    #

    ####################################### 
    # Convert the Ligand from PDB to PDBQT                                                                 # DockingModel
    ########################################## 
    def convert_ligand_pdb_file_to_pdbqt(self, pdb_file):
        """
        Convert the ligands of a given directory from pdb to pdbqt format
        
        Inputs:
        :param str pdb_file: the file name, a string.
        Returns:
        :returns: bool bool: True if it worked; 
                            False if its the gypsum param file or if it failed to make PDBQT
        :returns: str smile_name: name of the SMILES string from a pdb file
                                    None if its the param file 
        """
        smile_name = self.get_smile_name_from_PDB(pdb_file)

        ligand4_template = self.vars['prepare_ligand4.py']
        mgl_python = self.vars['mgl_python']
                
        # gypsum makes 1 files labeled params which is not a valid pdb, but is actually a log
        # Do not convert the params files
        if "params" in pdb_file:
            return False, None

        # if the file already has been converted to a .pbdqt skip this file take all other pdb file names
        if not os.path.exists(pdb_file + "qt"): 

            # make sure its in proper format
            self.convert_pdb_to_pdbqt_acceptable_format(pdb_file)
            self.prepare_ligand_processing(mgl_python, ligand4_template,pdb_file)
            if not os.path.exists(pdb_file + "qt"): 
                # FILE FAILED TO CONVERT TO PDBQT DELETE PDB AND RETURN FALSE
                print("PDBQT not generated: Deleting " + os.path.basename(pdb_file) + "...")
                
                #REMOVED FOR LIGANDS WHICH FAILED TO CONVERT TO PDBQT
                Delete.delete_all_associated_files(pdb_file)
                return False, smile_name

        return True, smile_name
    # 

    # Convert Ligand from PDB to PDBQT conversion
    def prepare_ligand_processing(self, mgl_python, prepare_script, mol_filename):
        """
        This function will convert a single ligand from PDB to PDBQT using MGLTools.
        It has 10seconds to sucessfull convert this. It will try to convert the 
            ligand up to 3 times
        If it fails to do so 3 times, whether because it timed out  
            or because MGLTools failed or because of an MGLTools Glitch,
            it will stop and the ligand won't be docked. 

        It will print the ligand if it fails 3 times. It will also fail if the
        molecule is unable to be imported into rdkit and sanitized. This is because MGLTools
        is sensitive to issues like atoms replaced with *, formating errors, and improper valences.
        Because MGLTools will crash with these issues the RDKit check is especially useful to prevent hard crashes.

        Inputs:
        :param str mgl_python: file path of the pythonsh file of mgl tools
        :param str prepare_script:  the file path for the mgltool ligand prep file receptor4
        :param str mol_filename:  the file path of the ligand
        """

        vars = self.vars
        timeout_option = vars["timeout_vs_gtimeout"]

        # Check that the PDB is a valid PDB file in RDKIT
        try:
            mol = Chem.MolFromPDBFile(mol_filename,sanitize=False,removeHs=False)
            if mol is not None:
                mol = MOH.check_sanitization(mol)
        except:
            mol = None
            
        temp_file = "{}_temp".format(mol_filename)
        if mol is not None:
            count = 0
            # timeout or gtimeout
            command = timeout_option + " 10 " + mgl_python + " " + prepare_script + " -g -l " + mol_filename + " -o " + mol_filename + "qt"

            while not os.path.exists(mol_filename + "qt"):
                if count < 3:         
                # We will try up to 3 times 
                    try:
                        subprocess.check_output(command +" 2> "+ temp_file, shell=True) 
                    except:
                        try:
                            os.system(command +" 2> "+ temp_file)
                        except:
                            pass
                        if os.path.exists( mol_filename + "qt") == False:
                            printout = "Failed to convert {} times: {}".format(count,mol_filename)
                            print(printout)
                            
                    count = count + 1
                            
                else:
                    printout = "COMPLETELY FAILED TO CONVERT: {}".format(mol_filename)
                    print(printout)
                    break 

        if os.path.exists(temp_file)==True:
            command = "rm {}".format(temp_file)
            try:
                os.system(command)
            except:
                print("Check permissions. Could not delete {}".format(temp_file))
    #

    # Convert PDB to acceptable PDBQT file format before converting
    def convert_pdb_to_pdbqt_acceptable_format(self, filename): 
        """
        Make sure a PDB file is properly formatted for conversion to pdbqt
        
        Inputs:
        :param str filename: the file path of the pdb file to be converted
        """  
        # read in the file
        output_lines = []
        with open(filename,'r') as f:
            for line in f.readlines():
                line = line.replace("\n","")
                if line[:5] == "ATOM " or line[:7] == "HETATM ":
                    # fix things like atom names with two letters
                    first = line[:11]
                    middle = line[11:17].upper().strip() #Need to remove whitespaces on both ends
                    last = line[17:]
                    
                    middle_firstpart = ""
                    middle_lastpart = middle
                    
                    for i in range(len(middle_lastpart)):
                        if middle_lastpart[:1].isupper() == True:
                            middle_firstpart = middle_firstpart + middle_lastpart[:1]
                            middle_lastpart = middle_lastpart[1:]
                        else: break # you reached the first number
                    
                    # now if there are more than two letters in middle_firstpart, keep just two
                    if len(middle_firstpart) > 2:
                        middle_lastpart = middle_firstpart[2:] + middle_lastpart
                        middle_firstpart = middle_firstpart[:2]
                    
                    if not (middle_firstpart == "BR" or middle_firstpart == "ZN" or middle_firstpart == "FE" or middle_firstpart == "MN" or middle_firstpart == "CL" or middle_firstpart == "MG"):
                        # so just keep the first letter for the element part of the atom name
                        middle_lastpart = middle_firstpart[1:] + middle_lastpart
                        middle_firstpart = middle_firstpart[:1]
                    
                    middle = middle_firstpart.rjust(3) + middle_lastpart.ljust(3)
                    
                    line = first + middle + last
                    
                    # make sure all parts of the molecule belong to the same chain and resid
                    line = line[:17] + "LIG X 999" + line[26:]
                    
                    output_lines.append(line)
                else:
                    output_lines.append(line)
                    

        with open(filename,'w') as f:
            
            for line in output_lines:
                f.write(line + "\n")
    #

    # Finding PDBs for ligands in a folder
    def find_pdb_ligands(self, current_generation_PDB_dir):    
        """
        This finds all the pdb files of ligands in a directory

        Inputs:
        :param str current_generation_PDB_dir: the dir path which contains the pdb files
                                                of ligands to be converted
        Returns:
        :returns: list pdbs_in_folder: a list of all PDB's in the dir 
        """

        # make list of every pdb in the current generations pdb folder
        pdbs_in_folder = []
        for filename in glob.glob(current_generation_PDB_dir + "*.pdb"): 
            pdbs_in_folder.append(filename)

        return pdbs_in_folder
    #

    # Find ligands which converted to PDBQT
    def find_converted_ligands(self, current_generation_PDB_dir):        
        """
        This finds all the pdbqt files of ligands in a directory

        Inputs:
        :param str current_generation_PDB_dir: the dir path which contains the pdbqt files
                                                of ligands to be docked
        Returns:
        :returns: list pdbqts_in_folder: a list of all PDBqt's in the dir 
        """
        # make list of every pdbqt in the current generations pdb folder
        pdbqts_in_folder = []
        for filename in glob.glob(current_generation_PDB_dir + "*.pdbqt"): 
            pdbqts_in_folder.append(filename)

        return pdbqts_in_folder
    #
    
    ####################################### 
    # DOCK USING VINA                     # 
    #######################################
    def dock_ligand(self, lig_pdbqt_filename):
        """
        Dock the ligand pdbqt files in a given directory using AutoDock Vina
        
        Inputs:
        :param str lig_pdbqt_filename: the ligand pdbqt filename
        """
        vars = self.vars
        timeout_option = vars["timeout_vs_gtimeout"]
        docking_timeout_limit = vars["docking_timeout_limit"]
        # do the docking of the ligand
        # Run with a timeout_option limit. Default setting is 5 minutes. This is excessive as most things run within 30seconds
        # This will prevent stalling out.
        # timeout or gtimeout
        torun = "{} {} {}".format(timeout_option, docking_timeout_limit, vars['docking_executable']) + \
            " --center_x " + str(vars['center_x']) + " --center_y " + str(vars['center_y']) + " --center_z " + \
            str(vars['center_z']) + " --size_x " + str(vars['size_x']) + " --size_y " + str(vars['size_y']) + \
            " --size_z " + str(vars['size_z']) + " --receptor " + self.receptor_PDBQT_file + \
            " --ligand " + lig_pdbqt_filename + " --out " + lig_pdbqt_filename + ".vina --cpu 1"

        # Add optional user variables additional variable 
        if vars['docking_exhaustiveness'] != None and vars['docking_exhaustiveness']!="None":
            if type(vars['docking_exhaustiveness']) == int or type(vars['docking_exhaustiveness']) == float:
                torun = torun + " --exhaustiveness " + str(int(vars['docking_exhaustiveness']))
        if vars['docking_num_modes'] != None and vars['docking_num_modes']!="None":
            if type(vars['docking_num_modes']) == int or type(vars['docking_num_modes']) == float:
                torun = torun + " --num_modes " + str(int(vars['docking_num_modes']))

        # Add output line MUST ALWAYS INCLUDE THIS LINE
        torun = torun + " >>" + lig_pdbqt_filename + "_docking_output.txt " + " 2>>" + lig_pdbqt_filename + "_docking_output.txt"

        print("\tDocking: {}".format(lig_pdbqt_filename))
        results = self.execute_docking_vina(torun)
        
        if results== None or results== None or results==256:
            made_changes = self.replace_atoms_not_handled_by_forcefield(lig_pdbqt_filename)
            if made_changes == True:
                results = self.execute_docking_vina(torun)
                if results == 256 or results == None:
                    print("\nLigand failed to dock after corrections: {}\n".format(lig_pdbqt_filename))
            else:
                print("\tFinished Docking: {}".format(lig_pdbqt_filename))
        else:
            print("\tFinished Docking: {}".format(lig_pdbqt_filename))
    #
    
    def replace_atoms_not_handled_by_forcefield(self, lig_pdbqt_filename):
        """
        Replaces atoms not handled by the forcefield to prevent errors. Atoms include B and Si.
        
        Inputs:
        :param str lig_pdbqt_filename: the ligand pdbqt filename
        Returns:
        :returns: bool retry: If True it will be ligand will be redocked, if False its dones and wont be docked again.
        """

        # VINA/QuickVINA and MGL have problems with the forcefields for certain atom types
        # To correct this, Autodock Vina suggests replacing the 
        atoms_to_replace = ["B \n","B\n","Si \n","Si\n"]  #add the \n at the end so we replace the end portion of the line
        printout_of_file = ""
        printout_info = ""
        retry = False
        line_count = 0
        with open(lig_pdbqt_filename, 'r') as f:
            for line in f.readlines():
                line_count = line_count + 1
                if "HETATM" in line:
                    for x in atoms_to_replace:
                        if x in line:
                            line = line.replace(x,"A \n")
                            retry = True
                            
                            printout_info = printout_info + "Changing '{}' to 'A ' in line: {} of {}".format(str(x.strip()),line_count,lig_pdbqt_filename)   #x Need to remove whitespaces on both ends
                printout_of_file = printout_of_file + line

        if retry == True:
            print(printout_info)
            with open(lig_pdbqt_filename, 'w') as f:
                f.write(printout_of_file)
        else:
            printout_info = "\nCheck the docking message for 'Parse error on'"
            printout_info = printout_info + "\n\t This ligand failed to dock. Please check that all atoms are covered by the docking forcefield"
            printout_info = printout_info + "\n\t Any atoms not covered by the forcefield should be added to atoms_to_replace in the function replace_atoms_not_handled_by_forcefield"    
            printout_info = printout_info + "\n\t Verify for this ligand: {}\n".format(lig_pdbqt_filename)
            print(printout_info)
        return retry 
    #

    def execute_docking_vina(self, command):
        """
        Run a single docking execution command
        
        Inputs:
        :param str command: string of command to run.
        Return:
        :returns: int result: the exit output for the command. If its None of 256 it failed.
        """
        
        try:
            result = os.system(command)
        except:
            result = None
            print("Failed to execute: " + command)    
        return result

    #

    def check_docked(self, pdb_file):
        """
        given a pdb_file name, test if a pdbqt.vina was created.
        If it failed to dock delete the file pdb and pdbqt file for that ligand
            -then return false

        if it docked properly return True
        Inputs:
        :param str pdb_file: pdb file path

        Returns:
        :returns: bool bool: false if not vina was unsuccessful 
        :returns: str smile_name: name of the pdb file
        """
        if not os.path.exists(pdb_file):
            # PDB file doesn't exist
            return False, None
        else:
            smile_name = self.get_smile_name_from_PDB(pdb_file)

        if not os.path.exists(pdb_file + "qt.vina"): # so this pdbqt.vina file didn't exist
            print("Docking unsuccessful: Deleting " + os.path.basename(pdb_file) + "...")
            
            #REMOVE Failed molecules
            # delete ones that were not docked successfully
            Delete.delete_all_associated_files(pdb_file)
            # # delete pdbqt_file
            pdbqt_file = pdb_file + "qt"
            Delete.delete_all_associated_files(pdbqt_file)

            return False, smile_name
        else:
            return True, smile_name
    #

    ####################################### 
    # Handle Failed PDBS                  #
    ####################################### 
    def get_smile_name_from_PDB(self, pdb_file):
        """
        This will return the unique identifier name for the compound
                
        Inputs:
        :param str pdb_file: pdb file path
        Returns:
        :returns: str line_stripped: the name of the SMILES string 
                                with the new lines and COMPND removed
        """   
        if os.path.exists(pdb_file):  
            with open(pdb_file, 'r') as f:
                for line in f.readlines():
                    if "COMPND" in line:
                        line_stripped = line.replace("COMPND","").strip()   #Need to remove whitespaces on both ends
                        line_stripped = line_stripped.replace("\n","").strip() #Need to remove whitespaces on both ends
                        compound_name = line_stripped

            # line_stripped is now the name of the smile for this compound
        else:
            line_stripped = "unknown"
        return line_stripped
    #


    ########################################## 
    # Convert the dock outputs to a usable formatted .smi file
    # This is mandatory for all Docking classes but 
    # implimentation and approach varies by docking and scoring choice                                       
    ########################################## 
      
    def rank_and_save_output_smi(self, vars, current_generation_dir, current_gen_int, smile_file, deleted_smiles_names_list):
        """
        Given a folder with PDBQT's, rank all the SMILES based on docking score (High to low).
        Then format it into a .smi file.
        Then save the file.

        Inputs:
        :param dict vars: vars needs to be threaded here because it has the paralizer object which is needed within Scoring.run_scoring_common
        :param str current_generation_dir: path of directory of current generation
        :param int current_gen_int: the interger of the current generation indexed to zero
        :param str smile_file:  File path for the file with the ligands for the generation which will be a .smi file
        :param list deleted_smiles_names_list: list of SMILES which may have failed the conversion process

        Return:
        :returns: str output_ranked_smile_file: the path of the output ranked .smi file
        """

        # Get directory string of PDB files for Ligands
        folder_with_pdbqts = current_generation_dir + "PDBs" + os.sep   

        # Run any compatible Scoring Function
        smiles_list = Scoring.run_scoring_common(vars, smile_file, folder_with_pdbqts)

        # Before ranking these we need to handle Pass-Through ligands from the last generation
        # If it's current_gen_int==1 or if vars['redock_advance_from_previous_gen']==True
        #       -Both of these states dock all ligands from the last generation so all of the pass-through lig
        #           are already in the PDB's folder thus they should be accounted for in smiles_list 
        # If vars['redock_advance_from_previous_gen']==False and current_gen_int != 1
        #       - We need to append the scores form the last gen to smiles_list
        
        # Only add these when we haven't already redocked the ligand
        if self.vars['redock_advance_from_previous_gen'] == False and current_gen_int != 0:
            # Go to previous generation folder
            prev_gen_num = str(current_gen_int - 1)
            run_folder = self.vars['output_directory']
            previous_gen_folder = run_folder + "generation_{}{}".format(str(prev_gen_num), os.sep)
            ranked_smi_file_prev_gen = previous_gen_folder + "generation_{}_ranked.smi".format(str(prev_gen_num))

            # Also check sometimes Generation 1 won't have a previous generation to do this with and sometimes it will
            if current_gen_int == 1 and os.path.exists(ranked_smi_file_prev_gen) == False:
                pass
            else:
                print("Getting ligand scores from the previous generation")

                # Shouldn't happen but to be safe.
                if os.path.exists(ranked_smi_file_prev_gen) == False:
                    raise Exception("Previous generation ranked .smi file does not exist. Check if output folder has been moved")

                # Get the data for all ligands from previous generation ranked file
                prev_gen_data_list = Ranking.get_usable_fomat(ranked_smi_file_prev_gen)

                # Get the list of pass through ligands
                current_gen_pass_through_smi = current_generation_dir + "SeedFolder{}Chosen_To_Advance_Gen_{}.smi".format(os.sep, str(current_gen_int))
                pass_through_list = Ranking.get_usable_fomat(current_gen_pass_through_smi)

                # Convert lists to searchable Dictionaries.
                prev_gen_data_dict = Ranking.convert_usable_list_to_lig_dict(prev_gen_data_list)

                pass_through_data = []
                for lig in pass_through_list:
                    smile_plus_id = str(lig[0] + lig[1])
                    lig_data = prev_gen_data_dict[smile_plus_id]
                    lig_info_remove_diversity = [lig_data[x] for x in range(0, len(lig_data)-1)]
                    pass_through_data.append(lig_info_remove_diversity) 
                
                smiles_list.extend(pass_through_data)
        
        # Output format of the .smi file will be:
        # SMILES    Full_lig_name   shorthandname   ...Anycustominfo... Fitness_metric  diversity
        # Normally the docking score is the fitness metric but if we use a custom metric than dock score gets
        # moved to index -3 and the new fitness metric gets -2

        # sort list by the affinity of each sublist (which is the last index of sublist) 
        smiles_list.sort(key = lambda x: float(x[-1]),reverse = False)
       
        # score the diversity of each ligand compared to the rest of the ligands in the group
        # this adds on a float in the last column for the sum of pairwise comparisons
        # the lower the diversity score the more unique a molecule is from the other mols 
        # in the same generation
        smiles_list = Ranking.score_and_append_diversity_scores(smiles_list)

        # name for the output file
        output_ranked_smile_file = smile_file.replace(".smi","") + "_ranked.smi"

        # save to a new output smiles file.
        # ie. save to ranked_smiles_file

        with open(output_ranked_smile_file, "w") as output:
            for ligand_info_list in smiles_list:
                str_ligand_info_list = [str(x) for x in ligand_info_list]
                output_line = "\t".join(str_ligand_info_list) + "\n"
                output.write(output_line)

        return output_ranked_smile_file
    #

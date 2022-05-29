"""UserVars
This should contain the functions for defining input variables.
Both the default variables and the user input variables.
This should also validate them.
"""

import __future__

import glob
import os
import datetime
import time
import json
import sys
import platform

def program_info():
	"""
	Get the program version number, etc.
	
	Returns:
	:returns: str program_output: a string for the print of the program information
	"""
	program_output = "\nAutoGrow Version 4.0.0\n"
	program_output = program_output + " ================== \n"
	program_output = program_output + "If you use AutoGrow 4.0.0 in your research, please cite the following reference:\n"
	program_output = program_output + "Spiegel, J, Ropp, P. J., Durrant, J. D. \n"
	program_output = program_output + " ================== \n\n"
	
	return program_output
#

def multiprocess_handling(vars):
    """
    This function handles the multiprocessing functions. It establishes a Paralellizer object
    and adds it to the vars dictionary.
    Input:
    :param dict vars: dict of user variables which will govern how the programs runs
    Returns:
	:returns: dict vars: dict of user variables which will govern how the programs runs
	"""   

    # Handle Serial overriding number_of_processors
    # serial fixes it to 1 processor
    if vars["multithread_mode"] == "serial" or vars["multithread_mode"]=="Serial":
        
        if vars["number_of_processors"] != 1:
            print("Because --multithread_mode was set to serial, this will be run on a single processor.")
        vars["number_of_processors"] = 1

    # Handle mpi errors if mpi4py isn't installed
    if vars["multithread_mode"] == "mpi" or vars["multithread_mode"] == "MPI":
        try:
            import mpi4py
        except:
            printout = "mpi4py not installed but --multithread_mode is set to mpi. \n Either install mpi4py or switch multithread_mode to multithreading or serial"
            raise ImportError(printout)

        try:
            import func_timeout
            from func_timeout import func_timeout, FunctionTimedOut
        except:
            printout = "func_timeout not installed but --multithread_mode is set to mpi. \n Either install func_timeout or switch multithread_mode to multithreading or serial"
            raise ImportError(printout)
      
    # # # launch mpi workers
    if vars["multithread_mode"] == 'mpi':
        # Avoid EOF error
        from autogrow.Operators.ConvertFiles.gypsum_dl.gypsum_dl.Parallelizer import Parallelizer

        vars["Parallelizer"] = Parallelizer(vars["multithread_mode"], vars["number_of_processors"])

        if vars["Parallelizer"] == None:
            printout = "EOF ERRORS FAILED TO CREATE A PARALLIZER OBJECT"
            print(printout)
            raise Exception(printout)

    else:
        # Lower level mpi (ie making a new Parallelizer within an mpi) 
        #   has problems with importing the MPI enviorment and mpi4py
        #   So we will flag it to skip the MPI mode and just go to multithread/serial
        # This is a saftey precaution
        from autogrow.Operators.ConvertFiles.gypsum_dl.gypsum_dl.Parallelizer import Parallelizer
                
        vars["Parallelizer"] = Parallelizer(vars["multithread_mode"], vars["number_of_processors"], True)



    # For Debugging
    # print("")
    # print("###########################")
    # print("number_of_processors  :  ", vars["number_of_processors"])
    # print("chosen mode  :  ", vars["multithread_mode"])
    # print("Parallel style:  ", vars["Parallelizer"].return_mode())
    # print("Number Nodes:  ", vars["Parallelizer"].return_node())
    # print("###########################")
    # print("")
    return vars
#

############################################
###### Variables Handlining Settings #######
############################################
def check_for_required_inputs(input_params):
    """
    Confirm all the required inputs were provided.

    Required Variables go here.

    Inputs:
    :param dict input_params: The parameters. A dictionary of {parameter name: value}.
    """
    keys_from_input = list(input_params.keys())

    list_of_required_inputs = ["filename_of_receptor","center_x","center_y","center_z",\
        "size_x","size_y","size_z","root_output_folder","source_compound_file",\
        "mgltools_directory"]

    missing_variables = []
    for variable in list_of_required_inputs:
        if variable in keys_from_input:
            continue
        else:
            missing_variables.append(variable)

    if len(missing_variables) != 0:
        printout = "\nRequired variables are missing from the input. A description of each of these can be found by running python ./RunAutogrow -h"
        printout = printout + "\nThe following required variables are missing: "
        for variable in missing_variables:
            printout = printout + "\n\t" + variable
        print("")
        print(printout)
        print("")
        raise NotImplementedError("\n"+printout+"\n")

    # Make sure the dimmensions are in floats. If in int convert to float.
    for x in ["center_x","center_y","center_z", "size_x","size_y","size_z"]:
        if type(input_params[x]) == float:
            continue
        elif type(input_params[x]) == int:
            input_params[x] == float(input_params[x])   
        else:
            printout = "\n{} must be a float value.\n".format(x)
            print(printout)
            raise Exception(printout)

    # Check Docking Exhaustiveness and modes...
    if "docking_exhaustiveness" in list(input_params.keys()):
        if input_params["docking_exhaustiveness"] == "None":
            input_params["docking_exhaustiveness"] = None
        if input_params["docking_exhaustiveness"] != None:

            try:
                input_params["docking_exhaustiveness"] = int(input_params["docking_exhaustiveness"])
            except:
                pass
            if type(input_params["docking_exhaustiveness"]) != int and type(input_params["docking_exhaustiveness"]) != float:
                raise Exception("docking_exhaustiveness needs to be an interger. \
                    If you do not know what to use, leave this blank and the default for the docking software will be used.")
    if "docking_num_modes" in list(input_params.keys()):
        if input_params["docking_num_modes"] == "None":
            input_params["docking_num_modes"] = None
        if input_params["docking_num_modes"] != None:
            try:
                input_params["docking_num_modes"] = int(input_params["docking_num_modes"])
            except:
                pass

            if type(input_params["docking_num_modes"]) != int and type(input_params["docking_num_modes"]) != float:
                raise Exception("docking_num_modes needs to be an interger. \
                    If you do not know what to use, leave this blank and the default for the docking software will be used.")

    # Check numbers which may be defined by first generation
    if "top_mols_to_seed_next_generation_first_generation" not in list(input_params.keys()):
        if "top_mols_to_seed_next_generation" not in list(input_params.keys()):
            # Use defined default of 10
            input_params["top_mols_to_seed_next_generation"] = 10
            input_params["top_mols_to_seed_next_generation_first_generation"] = 10
        else:
            input_params["top_mols_to_seed_next_generation_first_generation"] = input_params["top_mols_to_seed_next_generation"]
    
    if "number_of_crossovers_first_generation" not in list(input_params.keys()):
        if "number_of_crossovers" not in list(input_params.keys()):
            # Use defined default of 10
            input_params["number_of_crossovers"] = 10
            input_params["number_of_crossovers_first_generation"] = 10
        else:
            input_params["number_of_crossovers_first_generation"] = input_params["number_of_crossovers"]
    
    if "number_of_mutants_first_generation" not in list(input_params.keys()):
        if "number_of_mutants" not in list(input_params.keys()):
            # Use defined default of 10
            input_params["number_of_mutants"] = 10
            input_params["number_of_mutants_first_generation"] = 10
        else:
            input_params["number_of_mutants_first_generation"] = input_params["number_of_mutants"]

    if "number_to_advance_from_previous_gen_first_generation" not in list(input_params.keys()):
        if "number_to_advance_from_previous_gen" not in list(input_params.keys()):
            # Use defined default of 10
            input_params["number_to_advance_from_previous_gen"] = 10
            input_params["number_to_advance_from_previous_gen_first_generation"] = 10
        else:
            input_params["number_to_advance_from_previous_gen_first_generation"] = input_params["number_to_advance_from_previous_gen"]

    #######################################
    # Check that all required files exist #
    #######################################
    
    # convert paths to abspath, in case necessary
    input_params['filename_of_receptor'] = os.path.abspath(input_params['filename_of_receptor'])
    input_params['root_output_folder'] = os.path.abspath(input_params['root_output_folder'])
    input_params['source_compound_file'] = os.path.abspath(input_params['source_compound_file'])
    input_params['mgltools_directory'] = os.path.abspath(input_params['mgltools_directory'])

    # Check filename_of_receptor exists
    if os.path.isfile(input_params["filename_of_receptor"]) == False:
        raise NotImplementedError("Receptor file can not be found. File must be a .PDB file.")
    if ".pdb" not in input_params["filename_of_receptor"]:
        raise NotImplementedError("filename_of_receptor must be a .PDB file.")


    # Check root_output_folder exists
    if os.path.exists(input_params["root_output_folder"]) == False:
        # If the output directory doesn't exist, then make ithe output directory doesn't exist, then make it
        try:
            os.makedirs(vars['root_output_folder'])
        except:
            raise NotImplementedError("root_output_folder could not be created. Please manual create desired directory or check input parameters")
            
        if os.path.exists(input_params["root_output_folder"]) == False:
            raise NotImplementedError("root_output_folder does not exist")
    if os.path.isdir(input_params["root_output_folder"]) == False:
        raise NotImplementedError("root_output_folder is not a directory. Check your input parameters.")

    # Check source_compound_file exists
    if os.path.isfile(input_params["source_compound_file"]) == False:
        raise NotImplementedError("source_compound_file can not be found. File must be a tab delineated .smi file.")
    if ".smi" not in input_params["source_compound_file"]:
        raise NotImplementedError("source_compound_file must be a tab delineated .smi file.")

    # Check mgltools_directory exists
    if os.path.exists(input_params["mgltools_directory"]) == False:
        raise NotImplementedError("mgltools_directory does not exist")
    if os.path.isdir(input_params["mgltools_directory"]) == False:
        raise NotImplementedError("mgltools_directory is not a directory. Check your input parameters.")
#

def determine_bash_timeout_vs_gtimeout():
    """
    This function tests whether we should use the BASH command "timeout" (for linux) or
        the homebrew function "gtimeout" for MacOS

    Returns:
    :returns: str timeout_option: A string either "timeout" or "gtimeout" describing whether
            the bash terminal is able to use the bash function timeout or gtimeout
    """

    if platform.system() == "Linux":
        # Should be true and default installed in all Linux machines
        return "timeout"

    command = 'timeout 1 echo " "'
    # Running the os.system command for command will return 0,1, or 32512
        # 0 means that the timeout function works (most likely this is a linux os)
        # 32512 means that the timeout function DOES NOT Work (most likely this is MacOS)

    try:  # timeout or gtimeout
        timeout_result = os.system(command)
    except:
        raise Exception("Something is very wrong. This OS may not be supported by Autogrow or you may need to execute through Bash.")
    if timeout_result == 0:
        timeout_option = "timeout"
        return timeout_option

    else:
        try:  # timeout or gtimeout
            timeout_result = os.system("g" + command)
        except:
            raise Exception("Something is very wrong. This OS may not be supported by Autogrow or you may need to execute through Bash.")
        if timeout_result == 0:
            timeout_option = "gtimeout"
            return timeout_option
        else:
            printout = "Need to install GNU tools for Bash to work. \n"
            printout = printout + "This is essential to use Bash Timeout function in Autogrow. \n"
            printout = printout + "\t This will require 1st installing homebrew. \n"
            printout = printout + "\t\t Instructions found at: https://brew.sh/ \n"
            printout = printout + "\t Once brew is installed, please run: sudo brew install coreutils \n\n"
            print(printout)
            raise Exception(printout)
# 

def check_dependencies():
    """
    This function will try to import all the installed dependencies that will be
    used in Autogrow. If it fails to import it will raise an ImportError
    """
  
    # Check Bash Timeout function (There's a difference between MacOS and linux)
    # Linux uses timeout while MacOS uses gtimeout
    timeout_option = determine_bash_timeout_vs_gtimeout()
    if timeout_option != "timeout" and timeout_option != "gtimeout":
        raise Exception("Something is very wrong. This OS may not be supported by Autogrow or you may need to execute through Bash.")
        
    try:
        import rdkit
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D
        from rdkit.Chem.Draw import PrepareMolForDrawing
        from rdkit.Chem import rdFMCS
        from rdkit.Chem import FilterCatalog
        from rdkit.Chem.FilterCatalog import FilterCatalogParams
        import rdkit.Chem.Lipinski as Lipinski
        import rdkit.Chem.Crippen as Crippen
        import rdkit.Chem.Descriptors as Descriptors
        import rdkit.Chem.MolSurf as MolSurf

    except:
        print("You need to install rdkit and its dependencies.")
        raise ImportError("You need to install rdkit and its dependencies.")

    # molvs is prepackaged within gypsum_dl
    # try:
    #     from molvs import standardize_smiles as ssmiles
    # except:
    #     print("You need to install molvs and its dependencies.")
    #     raise ImportError("You need to install molvs and its dependencies.")

    try:
        import numpy
    except:
        print("You need to install numpy and its dependencies.")
        raise ImportError("You need to install numpy and its dependencies.")

    try:
        from scipy.cluster.vq import kmeans2
    except:
        print("You need to install scipy and its dependencies.")
        raise ImportError("You need to install scipy and its dependencies.")

    try:
        import os
        import sys
        import glob
        import subprocess
        import multiprocessing
        import time
    except:
            print("Missing a Python Dependency. Could be import: os,sys,glob,subprocess,multiprocess, time.")
            raise ImportError("Missing a Python Dependency. Could be import: os,sys,glob,subprocess,multiprocess, time.")
            
    try:
        import copy
        import random
        import string
        import math
    except:
        print("Missing a Python Dependency. Could be import: copy,random, string,math")
        raise ImportError("Missing a Python Dependency. Could be import: copy,random, string,math")

    try:
        from collections import OrderedDict
        import webbrowser
        import argparse
        import itertools
        import unittest
    except:
        print("Missing a Python Dependency. Could be import: collections,webbrowser,argparse,itertools,unittest")
        raise ImportError("Missing a Python Dependency. Could be import: collections,webbrowser,argparse,itertools,unittest")

    try:
        import textwrap
        import pickle
        import json
    except:
        print("Missing a Python Dependency. Could be import: textwrap, pickle,json")
        raise ImportError("Missing a Python Dependency. Could be import: textwrap, pickle,json")
#
 
def define_defaults(): 
    """
    Sets the command-line parameters to their default values.

    Returns:
    :returns: dict vars: a dictionary of all default variables
    """
    
    vars = {}
    
    # where we are currently (absolute filepath from route)
    # used for relative pathings
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Some variables which can be manually replaced but defaults point to prepackaged locations.
    ## Neural Network executable for scoring binding
    vars['nn1_script'] = os.path.join(script_dir,"Docking","Scoring","NNScore_exe", "nnscore1", "NNScore.py")
    # Example: vars['nn1_script'] = "/PATH/autogrow/autogrow/Docking/Scoring/NNScore_exe/nnscore1/NNScore.py"
    
    vars['nn2_script'] =  os.path.join(script_dir,"Docking","Scoring","NNScore_exe", "nnscore2", "NNScore2.py")
    # Example: vars['nn2_script'] = "/PATH/autogrow/autogrow/Docking/Scoring/nnscore2/NNScore2.py"

    #### OPTIONAL FILE-LOCATION VARIABLES #### 
    # (RECOMMEND SETTING TO "" SO AUTOGROW CAN AUTOLOCATE THESE FILES)# 
    vars['prepare_ligand4.py'] = ""    # vars['prepare_ligand4.py'] = "/PATH/MGLTools-1.5.4/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"
    vars['prepare_receptor4.py'] = ""  # vars['prepare_receptor4.py'] = "/PATH/MGLTools-1.5.4/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"
    vars['mgl_python'] = ""        # vars['mgl_python'] = "/PATH/MGLTools-1.5.4/bin/pythonsh"
    

    # Crossover function
    vars["start_a_new_run"] = False
    vars["max_time_MCS_prescreen"] = 1
    vars["max_time_MCS_thorough"] = 1
    vars["min_atom_match_MCS"] = 3
    vars["protanate_step"] = False

    # Mutation Settings
    vars["Rxn_library"] ="ClickChem"
    vars["rxn_library_file"] =""
    vars["function_group_library"] = ""
    vars["complimentary_mol_directory"] = ""

    # processors
    vars["number_of_processors"] = 1
    vars["multithread_mode"] = "multithreading"

    # Genetic Algorithm Components
    vars["Selector_Choice"] = "Roulette_Selector"
    vars["tourn_size"] = 0.1

    # Seeding next gen and diversity
    vars["top_mols_to_seed_next_generation_first_generation"] = 10
    vars["top_mols_to_seed_next_generation"] = 10
    vars["diversity_mols_to_seed_first_generation"] = 10
    vars["diversity_seed_depreciation_per_gen"] = 2

    # Populations settings
    vars["filter_source_compounds"] = True
    vars["use_docked_source_compounds"] = True
    vars["num_generations"] = 10
    vars["number_of_crossovers_first_generation"] = 10
    vars["number_of_mutants_first_generation"] = 10
    vars["number_of_crossovers"] = 10
    vars["number_of_mutants"] = 10
    vars["number_to_advance_from_previous_gen"] = 10
    vars["number_to_advance_from_previous_gen_first_generation"] = 10
    vars["redock_advance_from_previous_gen"] = False
    
    # Filters
    vars["Lipinski_Strict"] = False
    vars["Lipinski_Lenient"] = False
    vars["Ghose"] = False
    vars["Mozziconacci"] = False
    vars["VandeWaterbeemd"] = False
    vars["PAINS_Filter"] = False
    vars["NIH_Filter"] = False
    vars["BRENK_Filter"] = False
    vars["No_Filters"] = False
    vars["Alternative_filter"] = None

    # docking
    vars["Dock_choice"] = "QuickVina2Docking"
    vars["docking_executable"] = None
    vars["docking_exhaustiveness"] = None
    vars["docking_num_modes"] = None
    vars["docking_timeout_limit"] = 200

    # scoring
    vars["scoring_function"] = "VINA"
    vars["rescore_Lig_Efficiency"] = False
    vars["custom_scoring_script"] = ""

    # gypsum # max variance is the number of conformers made per ligand
    vars["max_variants_per_compound"] = 3
    vars["gypsum_thoroughness"] = 3
    vars["min_ph"] = 6.4
    vars["max_ph"] = 8.4
    vars["pka_precision"] = 1.0
    vars["gypsum_timeout_limit"] = 30
    

    # Other vars
    vars["debug_mode"] = False
    vars["reduce_files_sizes"] = False
    vars["generate_plot"] = True
    
    # Check Bash Timeout function (There's a difference between MacOS and linux)
    # Linux uses timeout while MacOS uses gtimeout
    timeout_option = determine_bash_timeout_vs_gtimeout()
    if timeout_option == "timeout" or timeout_option == "gtimeout":
        vars["timeout_vs_gtimeout"] = timeout_option
    else:
        raise Exception("Something is very wrong. This OS may not be supported by Autogrow or you may need to execute through Bash.")
        
    return vars
# 

############################################
######## Input Handlining Settings #########
############################################
def convert_json_params_from_unicode(params_unicode):
    """
    Set the parameters that will control this ConfGenerator object.

    :param dict params_unicode: The parameters. A dictionary of {parameter name:
                value}.
    Returns:
    :returns: dict params: Dictionary of User variables
    """
    # Also, rdkit doesn't play nice with unicode, so convert to ascii

    # Because Python2 & Python3 use different string objects, we separate their
    # usecases here.
    params = {}
    if sys.version_info < (3,):
        for param in params_unicode:
            val = params_unicode[param]
            if isinstance(val, unicode):
                val = str(val).encode("utf8")
            key = param.encode("utf8")
            params[key] = val
    else:
        for param in params_unicode:
            val = params_unicode[param]
            key = param
            params[key] = val
    return params
#

def check_value_types(vars, argv):
    """
    This checks that all the user variables loaded in use that same or comparable datatypes
    as the defaults in vars. This prevents type issues later in the simulation.
    
    Given the many uservars and the possibility for intentional differences, especially as the program is developed,
    this function tries to be NOT OPINIONATED, only correcting for several obvious and easy to correct issues
    of type discrepencies occur between argv[key] and vars[key]
        ie 
            1) argv[key] = "true" and vars[key] = False 
                this script will not change argv[key] to False... it will convert "true" to True
                ---> argv[key]=True
            2) argv[key] = "1.01" and vars[key] = 2.1 
                this script will change argv[key] from "1.01" to float(1.01)       

    Input:
    :param dict vars: Dictionary of program defaults, which will later be overwriten by argv values
    :param dict argv: Dictionary of User specified variables
    Returns:
    :param dict vars: Dictionary of program defaults, which will later be overwriten by argv values
    :param dict argv: Dictionary of User specified variables
    """
    for key in list(argv.keys()):
        if key not in list(vars.keys()):
            # Examples may be things like filename_of_receptor or dimensions of the docking box
            #   Just skip these
            continue
        
        if type(argv[key]) != type(vars[key]):
            # Several variable default is None which means checks are processed elsewhere...
            if vars[key] == None:
                # check argv[key] is "none" or "None"
                if type(argv[key])==str:
                    if argv[key].lower() == "none":
                        argv[key] = None
                else: continue

            #Handle number types
            elif type(vars[key]) == int or type(vars[key]) == float:
                if type(argv[key]) == int or type(argv[key]) == float:
                    # this is fine 
                    continue
                elif type(argv[key]) == str:
                    try: 
                        temp_item = float(argv[key])
                        if type(temp_item) == float:
                            argv[key] = temp_item
                        else:
                            printout = "This parameter is the wrong type. \n \t Check : {} type={}\n".format(key,type(argv[key]))
                            printout = printout + "\t Should be type={}\n\tPlease check Autogrow documentation using -h".format(type(vars[key]))
                            raise IOError(printout)
                    except:
                        printout = "This parameter is the wrong type. \n \t Check : {} type={}\n".format(key,type(argv[key]))
                        printout = printout + "\t Should be type={}\n\tPlease check Autogrow documentation using -h".format(type(vars[key]))
                        raise IOError(printout)
                else:
                    printout = "This parameter is the wrong type. \n \t Check : {} type={}\n".format(key,type(argv[key]))
                    printout = printout + "\t Should be type={}\n\tPlease check Autogrow documentation using -h".format(type(vars[key]))
                    raise IOError(printout)
            elif type(vars[key]) == bool:
                if argv[key] == None:
                    # Do not try to handle this. May make sense.
                    continue
                if type(argv[key]) == str:
                    if argv[key].lower() in ["true", '1']:
                        argv[key] = True
                    elif argv[key].lower() in ["false", '0']:
                        argv[key] = False
                    elif argv[key].lower() in ["none"]:
                        argv[key] = None
                    else: 
                        printout = "This parameter appears to be the wrong type. \n \t Check : {} type={}\n".format(key,type(argv[key]))
                        printout = printout + "\t Should be type={}\n\tPlease check Autogrow documentation using -h".format(type(vars[key]))
                        raise IOError(printout)
                else: 
                    printout = "This parameter appears to be the wrong type. \n \t Check : {} type={}\n".format(key,type(argv[key]))
                    printout = printout + "\t Should be type={}\n\tPlease check Autogrow documentation using -h".format(type(vars[key]))
                    raise IOError(printout)
    return vars, argv
#

def load_in_commandline_parameters(argv):
    """
    Load in the command-line parameters
    
    Input:
    :param dict argv: Dictionary of User specified variables

    Returns:
    :returns: dict vars: Dictionary of User variables
    :returns: str printout: a string to be printed to screen and saved to output file
    """

    vars = define_defaults()
    
    # Load the parameters from the json
    if 'json' in argv:
        json_vars = json.load(open(argv['json']))
        json_vars = convert_json_params_from_unicode(json_vars)
        check_for_required_inputs(json_vars)
        vars, json_vars = check_value_types(vars, json_vars)
        for key in list(json_vars.keys()):
            vars[key] = json_vars[key]
    else:
        check_for_required_inputs(argv)
        vars, argv = check_value_types(vars, argv)
        for key in list(argv.keys()):
            vars[key] = argv[key]

    vars = multiprocess_handling(vars)
    
    printout =  "(RE)STARTING AUTOGROW 4.0: " + str(datetime.datetime.now())
    printout = printout + program_info()
    printout = printout + "\nUse the -h tag to get detailed help regarding program usage.\n"
    print(printout)
    sys.stdout.flush()
    # Check all Dependencies are installed
    check_dependencies()

    vars = filter_choice_handling(vars)
    

    ###########################################
    ########## Check variables Exist ##########
    ###########################################

    # Check if custom docking option if so there's a few things which need to also be specified
    # if not lets flag the error
    if vars["Dock_choice"] == "Custom":
        if vars["docking_executable"] is None:
            raise ValueError("TO USE CUSTOM DOCKING OPTION, MUST SPECIFY THE PATH \
                            TO THE DOCKING_EXECUTABLE AND THE DOCKING_CLASS") 

    # Mutation Settings
    if vars['Rxn_library'] == "CUSTOM":
        if vars['rxn_library_file'] == "":
            raise ValueError("TO USE CUSTOM REACTION LIBRARY OPTION, ONE MUST SPECIFY THE PATH \
                            TO THE REACTION LIBRARY USING INPUT PARAMETER Rxn_library") 
        else:
            if os.path.exists(vars['rxn_library_file']) == False:
                raise ValueError("TO USE CUSTOM REACTION LIBRARY OPTION, ONE MUST SPECIFY THE PATH \
                                TO THE REACTION LIBRARY USING INPUT PARAMETER Rxn_library") 

        if vars['function_group_library'] == "":
                raise ValueError("TO USE CUSTOM REACTION LIBRARY OPTION, ONE MUST SPECIFY THE PATH \
                            TO THE REACTION LIBRARY USING INPUT PARAMETER function_group_library") 
        else:
            if os.path.exists(vars['rxn_library_file']) == False:
                raise ValueError("TO USE CUSTOM REACTION LIBRARY OPTION, ONE MUST SPECIFY THE PATH \
                            TO THE REACTION LIBRARY USING INPUT PARAMETER function_group_library") 


        if vars['complimentary_mol_directory'] == "":
            raise ValueError("TO USE CUSTOM REACTION LIBRARY OPTION, ONE MUST SPECIFY THE PATH \
                        TO THE REACTION LIBRARY USING INPUT PARAMETER function_group_library") 
        else:
            if os.path.isdir(vars['complimentary_mol_directory']) == False:
                raise ValueError("TO USE CUSTOM REACTION LIBRARY OPTION, ONE MUST SPECIFY THE PATH \
                            TO THE REACTION LIBRARY USING INPUT PARAMETER complimentary_mol_directory") 
    else:   # Using default settings
        if vars['rxn_library_file'] != "":
            raise ValueError("You have selected a custom rxn_library_file group library but not chosen to use \
                            the CUSTOM option for Rxn_library. Please use either the provided Rxn_library options or \
                            chose the CUSTOM option for Rxn_library") 
        if vars['function_group_library'] != "":
            raise ValueError("You have selected a custom function_group_library but not chosen to use \
                            the CUSTOM option for Rxn_library. Please use either the provided Rxn_library options or \
                            chose the CUSTOM option for Rxn_library") 
        if vars['complimentary_mol_directory'] != "":
            raise ValueError("You have selected a custom complimentary_mol_directory but not chosen to use \
                            the CUSTOM option for Rxn_library. Please use either the provided Rxn_library options or \
                            chose the CUSTOM option for Rxn_library") 
    
    # Check if the Operating System is Windows, if so turn off Multiprocessing.
    if os.name == "nt" or os.name == "ce": # so it's running under windows. multiprocessing disabled
        vars['number_of_processors'] = 1
        printout = printout + "\nWARNING: Multiprocessing is disabled on windows machines.\n"

    # convert paths to abspath, in case necessary
    vars['nn1_script'] = os.path.abspath(vars['nn1_script'])
    vars['nn2_script'] = os.path.abspath(vars['nn2_script'])
    
    # make sure directories end in os.sep
    if vars['root_output_folder'][-1] != os.sep: 
        vars['root_output_folder'] = vars['root_output_folder'] + os.sep
    if vars['mgltools_directory'][-1] != os.sep: 
        vars['mgltools_directory'] = vars['mgltools_directory'] + os.sep
    
    # find other mgltools-related scripts
    if vars['prepare_ligand4.py'] == "": 
        vars['prepare_ligand4.py'] = vars['mgltools_directory'] + 'MGLToolsPckgs' + os.sep + 'AutoDockTools' + os.sep + 'Utilities24' + os.sep + 'prepare_ligand4.py'
    if vars['prepare_receptor4.py'] == "": 
        vars['prepare_receptor4.py'] = vars['mgltools_directory'] + 'MGLToolsPckgs' + os.sep + 'AutoDockTools' + os.sep + 'Utilities24' + os.sep + 'prepare_receptor4.py'
    if vars['mgl_python'] == "": 
        vars['mgl_python'] = vars['mgltools_directory'] + 'bin' + os.sep + 'pythonsh'
        
    # More Handling for Windows OS
    # convert path names with spaces if this is windows
    if os.name == "nt" or os.name == "ce": # so it's running under windows. multiprocessing disabled

        if " " in vars['filename_of_receptor']: 
            vars['filename_of_receptor'] = '"' + vars['filename_of_receptor'] + '"'
        if " " in vars['root_output_folder']: 
            vars['root_output_folder'] = '"' + vars['root_output_folder'] + '"'
        if " " in vars['mgltools_directory']: 
            vars['mgltools_directory'] = '"' + vars['mgltools_directory'] + '"'
        if " " in vars['nn1_script']: 
            vars['nn1_script'] = '"' + vars['nn1_script'] + '"'
        if " " in vars['nn2_script']: 
            vars['nn2_script'] = '"' + vars['nn2_script'] + '"'
        if " " in vars['prepare_ligand4.py']: 
            vars['prepare_ligand4.py'] = '"' + vars['prepare_ligand4.py'] + '"'
        if " " in vars['prepare_receptor4.py']: 
            vars['prepare_receptor4.py'] = '"' + vars['prepare_receptor4.py'] + '"'
        if " " in vars['mgl_python']: 
            vars['mgl_python'] = '"' + vars['mgl_python'] + '"'
    #
        
    # output the paramters used
    printout = printout + "\nPARAMETERS" + "\n"
    printout = printout + " ========== " + "\n"

    
    # Make sure scripts and executables exist
    if not os.path.exists(vars['prepare_ligand4.py']) and not os.path.exists(vars['prepare_ligand4.py'].replace('"','')):
        printout = printout + "\nERROR: Could not find prepare_ligand4.py at " + vars['prepare_ligand4.py'] + "\n"
        print(printout)
        raise NotImplementedError(printout)
    if not os.path.exists(vars['prepare_receptor4.py']) and not os.path.exists(vars['prepare_receptor4.py'].replace('"','')):
        printout = printout + "\nERROR: Could not find prepare_receptor4.py at " + vars['prepare_receptor4.py'] + "\n"
        print(printout)
        raise NotImplementedError(printout)
    if not os.path.exists(vars['mgl_python']) and not os.path.exists(vars['mgl_python'].replace('"','')):
        printout = printout + "\nERROR: Could not find pythonsh at " + vars['mgl_python'] + "\n"
        print(printout)
        raise NotImplementedError(printout)
    if not os.path.exists(vars['nn1_script']) and not os.path.exists(vars['nn1_script'].replace('"','')):
        printout = printout + "\nERROR: Could not find " + os.path.basename(vars['nn1_script']) + " at " + vars['nn1_script'] + "\n"
        print(printout)
        raise NotImplementedError(printout)
    if not os.path.exists(vars['nn2_script']) and not os.path.exists(vars['nn2_script'].replace('"','')):
        printout = printout + "\nERROR: Could not find " + os.path.basename(vars['nn2_script']) + " at " + vars['nn2_script'] + "\n"
        print(printout)
        raise NotImplementedError(printout)
    if not os.path.exists(vars['filename_of_receptor']):
        printout = printout + "\nERROR: There receptor file does not exist: \"" + vars['filename_of_receptor'] + "\"." + "\n"
        print(printout)
        raise NotImplementedError(printout)


    # CHECK THAT NN1/NN2 are using only traditional Vina Docking
    if vars['scoring_function'] == "NN1" or  vars['scoring_function'] == "NN2":
        if vars['Dock_choice'] != "VinaDocking":
            printout = "\nNeural Networks 1 and 2 (NN1/NN2) are trained on data using PDBQT files converted by MGLTools \n"
            printout = printout + "and docked using Autodock Vina 1.1.2.\n"
            printout = printout + "Using conversion or docking software besides these will not work. \n"
            printout = printout + "Please switch Dock_choice option to VinaDocking or deselect NN1/NN2 as the scoring_function.\n"
            print(printout)
            raise Exception(printout)

    # IF ALTERNATIVE CONVERSION OF PDB2PDBQT CHECK THAT NN1/NN2 are using only MGLTOOLS 



    # Check if the user wants to continue a run or start a new run.
    # Make new run directory if necessary. return the Run folder path
    # The run folder path will be where we place our generations and output files
    vars["output_directory"] = set_run_directory(vars['root_output_folder'], vars['start_a_new_run'])
  
    return vars, printout
# 

############################################
######### File Handlining Settings #########
############################################
def find_previous_runs(folder_name_path):
    """
    This will check if there are any previous runs in the output directory.
        - If there are it will return the interger of the number label of the last Run folder path.
            - ie if there are folders Run_0, Run_1, Run_2 the function will return int(2)
        - If there are no previous Run folders it returns None.
        
    Input:
    :param str folder_name_path: is the path of the root output folder. We will make a directory within
                this folder to store our output files

    Returns:
    :returns: int last_run_number: the int of the last run number or None if no previous runs.
    """
    
    path_exists = True
    i = 0
    while path_exists is True:
        folder_path = "{}{}{}".format(folder_name_path, i, os.sep)
        if os.path.exists(folder_path):
            i = i + 1
        else:
            path_exists = False

    if i == 0:
        # There are no previous runs in this directory
        last_run_number = None
        return None
    else:
        # A previous run exists. The number of the last run.
        last_run_number = i - 1 
        return last_run_number
#

def set_run_directory(root_folder_path, start_a_new_run):
    """
    Determine and make the folder for the run directory. 
        If start_a_new_run == True    Start a frest new run. 
            -If no previous runs exist in the root_folder_path then make a new folder named root_folder_path + "Run_0"
            -If there are previous runs in the root_folder_path then make a 
                new folder incremental increasing the name by 1 from the last run in the same output directory.
        If start_a_new_run == False    Find the last run folder and return that path
            -If no previous runs exist in the root_folder_path then make a new folder named root_folder_path + "Run_0"

    Input:
    :param str root_folder_path: is the path of the root output folder. We will make a directory within
                this folder to store our output files        
    :param bol start_a_new_run: True or False to determine if we continue from the last run or start a new run
        - This is set as a vars["start_a_new_run"] 
        - The default is vars["start_a_new_run"] = True
    Returns:
    :returns: str folder_path: the string of the newly created directory for puting output folders
    """

    folder_name_path = root_folder_path + "Run_"
    print(folder_name_path)
    
    last_run_number = find_previous_runs(folder_name_path)

    if last_run_number == None:
        # There are no previous simulation runs in this directory
        print("There are no previous runs in this directory.")
        print("Starting a new run named Run_0.")

        # make a folder for the new generation
        run_number = 0
        folder_path = "{}{}{}".format(folder_name_path,run_number,os.sep)
        os.makedirs(folder_path)

    else:
        if start_a_new_run == False:
            # Continue from the last simulation run 
            run_number = last_run_number
            folder_path = "{}{}{}".format(folder_name_path, last_run_number,os.sep)
        else:   #start_a_new_run == True
            # Start a new fresh simulation
            # Make a directory for the new run by increasing run number by +1 from last_run_number
            run_number = last_run_number + 1
            folder_path = "{}{}{}".format(folder_name_path, run_number,os.sep)
            os.makedirs(folder_path)

    print("The Run number is: ", run_number)
    print("The Run folder path is: ", folder_path)
    print("")
    return folder_path
# 

############################################
######## Filter Handlining Settings ########
############################################
def filter_choice_handling(vars):
    """
    This function handles selecting the user defined Ligand filters.

    Input:
    :param dict vars: Dictionary of User variables
    Returns:
    :returns: dict vars: Dictionary of User variables with the Chosen_Ligand_Filters added
    """
    if "No_Filters" in list(vars.keys()):
        if vars["No_Filters"] is True:
            Chosen_Ligand_Filters = None
        else:
            Chosen_Ligand_Filters, vars = picked_filters(vars)
    else:
        Chosen_Ligand_Filters, vars = picked_filters(vars)
    vars["Chosen_Ligand_Filters"] = Chosen_Ligand_Filters

    import autogrow.Operators.Filter.ExecuteFilters as Filter
    # get child filter class object function dictionary
    vars["Filter_Object_Dict"] = Filter.make_run_class_dict(Chosen_Ligand_Filters)

    return vars
# 

def picked_filters(vars):
    """
    This will take the user vars and return a list of the filters which a molecule must pass
    to move into the next generation.

    Input:
    :param dict vars: Dictionary of User variables
    Returns:
    :returns: list filter_list: a list of the class of filter which will be used 
                                later to check for drug likeliness for a generation.
                                If a User adds their own filter they just need to follow the same nominclature and enter
                                    that filter in the user vars["Alternative_filters"] as the name of that class and place
                                    that file in the same folder as the other filter classes.
    """
    filter_list = []
    vars_keys = list(vars.keys())

    if "Lipinski_Strict" in vars_keys:
        if vars['Lipinski_Strict'] is True:
            filter_list.append('Lipinski_Strict')
    else:
        vars['Lipinski_Strict'] = False

    if "Lipinski_Lenient" in vars_keys:
        if vars['Lipinski_Lenient'] is True:
            filter_list.append('Lipinski_Lenient')
    else:
        vars['Lipinski_Lenient'] = False

    if "Ghose" in vars_keys:
        if vars['Ghose'] is True:
            filter_list.append('Ghose')
    else:
        vars['Ghose'] = False

    if "Mozziconacci" in vars_keys:
        if vars['Mozziconacci'] is True:
            filter_list.append('Mozziconacci')
    else:
        vars['Mozziconacci'] = False
        
    if "VandeWaterbeemd" in vars_keys:
        if vars['VandeWaterbeemd'] is True:
            filter_list.append('VandeWaterbeemd')
    else:
        vars['VandeWaterbeemd'] = False
        
    if "PAINS_Filter" in vars_keys:
        if vars['PAINS_Filter'] is True:
            filter_list.append('PAINS_Filter')
    else:
        vars['PAINS_Filter'] = False
        
    if "NIH_Filter" in vars_keys:
        if vars['NIH_Filter'] is True:
            filter_list.append('NIH_Filter')
    else:
        vars['NIH_Filter'] = False
        
    if "BRENK_Filter" in vars_keys:
        if vars['BRENK_Filter'] is True:
            filter_list.append('BRENK_Filter')
    else:
        vars['BRENK_Filter'] = False
        
    if "Alternative_filter" in vars_keys:
        if vars["Alternative_filter"] is not None:
            filter_list.extend(vars["Alternative_filter"])
    else:
        vars['Alternative_filter'] = None
        
    # if there is no user specified ligand filters but they haven't set
    # filters to None ---> set filter to default of Lipinski_Lenient.
    if len(filter_list) == 0:
        vars['Lipinski_Lenient'] = True
        filter_list.append('Lipinski_Lenient')

    return filter_list, vars
# 

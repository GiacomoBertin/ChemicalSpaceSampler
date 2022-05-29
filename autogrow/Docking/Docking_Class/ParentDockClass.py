class ParentDocking(object):
    """
    Docking
    Inputs:
    :param class object: a class to initialize on
    """
    def __init__(self, vars, receptor_file):
        """
        Require to initialize any docking class.

        Inputs:
        :param dict vars: Dictionary of User variables
        :param str receptor_file: a string for the receptor PDB file
        """
        pass
    #

    def get_name(self):
        """
        Returns the current class name.    
        Returns:
        :returns: str self.__class__.__name__: the current class name.
        """
        return self.__class__.__name__
    #

    def run_dock(self, input_string):
        """
        run_dock is needs to be implimented in each class.
        Inputs:
        :param str input_string: a string for docking process
                                raise exception if missing
        """
        raise NotImplementedError("run_dock() not implemented")
    #
    def run_ligand_handling_for_docking(self, input_string):
        """
        run_ligand_handling_for_docking is needs to be implimented in each class.
        This converts the PDB to whatever file format required by the docking.
            ie. PDB to PDBQT conversion for Vina and QuickVina
        Inputs:
        :param str input_string: a string for docking process
                                raise exception if missing
        """
        raise NotImplementedError("run_ligand_handling_for_docking() not implemented")
    #

    def rank_and_save_output_smi(self):
        """
        rank_and_save_output_smi is needs to be implimented in each class.
        raise exception if missing
        """
        raise NotImplementedError("rank_and_save_output_smi() not implemented")
    #
    

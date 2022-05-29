from abc import ABC, abstractmethod
from typing import List, Dict, Union
from Utility import LigandInfo
import torch.nn as nn
from featurization import Featurization
import numpy as np
from sklearn.neighbors import NearestNeighbors
from visualization import smi2fp
from rdkit import DataStructs


class Sampler(ABC, nn.Module):
    def __init__(self, sampler_id=0):
        super(Sampler, self).__init__()
        self.sampler_id = sampler_id
        self.chemical_space: List[LigandInfo] = None

    @abstractmethod
    def step(self, i, return_dict: dict) -> List[LigandInfo]:
        """
        Perform a step of the sampler: must identify the best ligands to evaluate

        :param return_dict:  at return_dict[self.sampler_id] set the output of the function
        :param i:            number of the step
        :return:             the molecules to evaluate
        """
        pass

    @abstractmethod
    def initialize(self, chemical_space: List[LigandInfo], sampler_id):
        """
        Load the chemical space

        :param sampler_id:       an id to identify samplers in multiprocessing context
        :param chemical_space:   a list with info about ligands
        :return:                 self
        """
        pass

    @abstractmethod
    def closing_step(self, chemical_space: List[LigandInfo]):
        """
        After that the proposed smiles are evaluated we adjourn the sampler

        :param chemical_space:   a dictionary with info about ligands
        :return:
        """
        pass

    def get_history(self) -> List[LigandInfo]:
        """
        Get the sampler history

        :return:
        """
        pass


class KNNSampler(Sampler, ABC):
    def __init__(self, k, featurization: Featurization, sampler_id=0):
        super(KNNSampler, self).__init__(sampler_id)
        self.k = k
        self.proposed_ids = []
        self.best_ligand_id = None
        self.featurization = featurization
        self.best_ligands_history = []
        self.proposed_id_history = []

    def initialize(self, chemical_space: List[LigandInfo], sampler_id):
        self.featurization.initialize(chemical_space)
        self.sampler_id = sampler_id
        self.chemical_space = chemical_space
        self.best_ligand_id = np.random.randint(0, len(self.chemical_space))
        return self

    def step(self, i, return_dict: dict) -> List[LigandInfo]:
        smiles = [smi.smiles for smi in self.chemical_space if smi.compound_id not in self.proposed_ids]
        sim = self.featurization.compute_distances([smiles[self.best_ligand_id]], smiles)
        idx = np.argsort(sim)[0, :self.k]
        return_dict[self.sampler_id] = [self.chemical_space[i] for i in idx]
        self.proposed_ids += list(idx)
        self.proposed_id_history.append(idx)
        return [self.chemical_space[i] for i in idx]

    def closing_step(self, chemical_space: List[LigandInfo]):
        self.featurization.adjourn(chemical_space)
        self.chemical_space = chemical_space
        scores = np.array([smi.score if smi.score is not None else 0. for smi in self.chemical_space])
        self.best_ligand_id = scores.argmin()
        self.best_ligands_history.append(self.best_ligand_id)

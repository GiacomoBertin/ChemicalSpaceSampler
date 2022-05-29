from abc import ABC, abstractmethod
from typing import Union, List
import torch
from visualization import smi2fp
import numpy as np
from Utility import LigandInfo
from sklearn.decomposition import PCA
from rdkit import DataStructs


class Featurization(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self, chemical_space: List[LigandInfo]):
        pass

    @abstractmethod
    def adjourn(self, chemical_space: List[LigandInfo]):
        pass

    @abstractmethod
    def compute_distances(self, X, Y):
        pass

    @abstractmethod
    def __call__(self, smiles: Union[List[str], str]):
        pass


class PCAFeaturization(Featurization):
    def __init__(self, n_components=2):
        super(PCAFeaturization, self).__init__()
        self.smiles = []
        self.n_components = n_components
        self.pca = None

    def initialize(self, chemical_space: List[LigandInfo]):
        self.adjourn(chemical_space)

    def adjourn(self, chemical_space: List[LigandInfo]):
        self.smiles = [ligand.smiles for ligand in chemical_space]
        fps = np.array([smi2fp(smi) for smi in self.smiles])
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(fps.reshape(-1, 1024))

    def compute_distances(self, X, Y):
        x_feat = self(X)
        y_feat = self(Y)
        dist = torch.cdist(torch.from_numpy(x_feat), torch.from_numpy(y_feat))
        return dist.cpu().numpy()

    def __call__(self, smiles_list: List[str]):
        fps = np.array([smi2fp(smi) for smi in smiles_list])
        return self.pca.transform(fps)


class FingerprintFeaturization(Featurization):
    def __init__(self):
        super(FingerprintFeaturization, self).__init__()

    def initialize(self, chemical_space: List[LigandInfo]):
        pass

    def adjourn(self, chemical_space: List[LigandInfo]):
        pass

    def compute_distances(self, X, Y):
        x_feat = self(X)
        y_feat = self(Y)
        dist = DataStructs.BulkTanimotoSimilarity(x_feat, y_feat)
        return dist

    def __call__(self, smiles_list: List[str]):
        fps = np.array([smi2fp(smi) for smi in smiles_list])
        return fps








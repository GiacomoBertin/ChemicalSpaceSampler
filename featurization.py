from abc import ABC, abstractmethod
from typing import Union, List
import torch
from visualization import smi2fp
import numpy as np
from Utility import LigandInfo
from sklearn.decomposition import PCA
from rdkit import DataStructs
import deepchem as dc
from torch.utils.data import Dataset
from tqdm import trange
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


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
        x_feat = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(X), 2, nBits=1024)
        y_feat = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, nBits=1024) for x in Y]
        dist = 1.0 - np.array(DataStructs.BulkTanimotoSimilarity(x_feat, y_feat))
        return dist

    def __call__(self, smiles_list: Union[List[str], str]):
        if isinstance(smiles_list, list):
            fps = np.array([smi2fp(smi) for smi in smiles_list])
        else:
            fps = smi2fp(smiles_list)
        return fps


class GraphFeaturization(Featurization):
    def __init__(self,
                 use_edges: bool = False,
                 use_chirality: bool = False,
                 use_partial_charge: bool = False,
                 graph_lib: str = 'pyg'):
        super(GraphFeaturization, self).__init__()
        self.featurizer = dc.feat.MolGraphConvFeaturizer(
            use_edges=use_edges,
            use_chirality=use_chirality,
            use_partial_charge=use_partial_charge
        )
        self.graph_lib = graph_lib

    def initialize(self, chemical_space: List[LigandInfo]):
        pass

    def adjourn(self, chemical_space: List[LigandInfo]):
        pass

    def compute_distances(self, X, Y):
        return None

    def to_data(self, X):
        pass

    def transform(self, graphdata: dc.feat.graph_data.GraphData):
        if self.graph_lib == 'pyg':
            return graphdata.to_pyg_graph()
        elif self.graph_lib == 'dgl':
            return graphdata.to_dgl_graph()
        else:
            return graphdata

    def __call__(self, smiles_list: Union[List[str], str]):
        fps = self.featurizer.featurize(smiles_list)
        if len(fps) == 1:
            fps = self.transform(fps[0])
        else:
            fps = [self.transform(f) for f in fps]
        return fps


class MolDataset(Dataset):
    def __init__(self, chemical_space: List[LigandInfo], featurizer):
        super(MolDataset, self).__init__()
        self.features = []
        self.featurizer = featurizer
        self.chemical_space = chemical_space
        for i in trange(0, len(self.chemical_space)):
            self.features.append(self.featurizer(self.chemical_space[i].smiles))
        self.set_chemical_space(chemical_space)

    def set_dg(self, idx, list_dg):
        for i, score in zip(idx, list_dg):
            self.chemical_space[i].score = score

    def set_chemical_space(self, chemical_space):
        self.chemical_space = [lig for lig in chemical_space if lig.score is not None]

    def __len__(self):
        return len(self.chemical_space)

    def __getitem__(self, item):
        lig: LigandInfo = self.chemical_space[item]
        x = self.featurizer(lig.smiles)
        y = torch.tensor(lig.score)
        return dict(
            x=x,
            y=y,
            lig=lig.compound_id
        )



from abc import ABC, abstractmethod
from typing import Union, List
import torch
import torch.nn as nn
from visualization import smi2fp
import numpy as np
from Utility import LigandInfo
from sklearn.decomposition import PCA
from rdkit import DataStructs
import deepchem as dc
from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, n_components: int = 2):
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


class VAEFeaturization(Featurization):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 smile_to_tensor: Featurization,
                 criterion: nn.Module = nn.MSELoss(),
                 epochs: int = 50,
                 lr: float = 1e-3,
                 batch_size: int = 32,
                 chkpt_path: str = None,
                 save_every: int = 10,
                 fixed_space: bool = False):
        super(VAEFeaturization, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.smile_to_tensor = smile_to_tensor

        self.dataset = None
        self.criterion = criterion
        self.epochs = epochs
        self.lr = lr
        self.dataset: MolDataset = None
        self.batch_size = batch_size
        self.chkpt_path = chkpt_path
        self.save_every = save_every
        self.optimizer = torch.optim.Adam(nn.ModuleList([self.encoder, self.decoder]).parameters(),
                                          lr=self.lr)

        self.fixed_space = fixed_space

    def compute_loss(self, y_hat, y, mu, log_var):
        recon_loss = self.criterion(y_hat, y)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + KLD

    def load_chkpt(self, optimizer):
        if self.chkpt_path is not None:
            chkpt = torch.load(self.chkpt_path)
            self.encoder.load_state_dict(chkpt['encoder'])
            self.decoder.load_state_dict(chkpt['decoder'])
            starting_epoch = chkpt['epoch']
            optimizer.load_state_dict(chkpt['optimizer'])

        else:
            starting_epoch = 0

        return starting_epoch

    def save_chkpt(self, optimizer, epoch):
        if self.chkpt_path is not None:
            torch.save(self.chkpt_path, dict(
                encoder=self.encoder.state_dict(),
                decoder=self.decoder.state_dict(),
                epoch=epoch,
                optimizer=optimizer.state_dict()
            ))

    def _train(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.load_chkpt(self.optimizer)
        pbar = trange(0, self.epochs, desc='iteration')

        for epoch in pbar:
            losses = []

            for batch_idx, batch in enumerate(dataloader):
                # print(vae(batch)[0].shape)
                self.optimizer.zero_grad()
                data = batch['x']
                z, mu, log_var = self.encoder(data)
                recon_data = self.decoder(z)

                loss = self.compute_loss(recon_data, data, mu=mu, log_var=log_var)
                loss.backward()
                losses.append(loss.detach().item())

                self.optimizer.step()
                # losses.append(loss.detach().item())

            mean_loss = np.mean(losses)
            pbar.set_description(f'loss: {mean_loss:.2f} ')

            if epoch % self.save_every == 0:
                self.save_chkpt(self.optimizer, epoch)

    def initialize(self, chemical_space: List[LigandInfo]):
        self.dataset = MolDataset(chemical_space=chemical_space,
                                  featurizer=self.smile_to_tensor,
                                  use_only_scored_ligands=False)
        self._train()

    def adjourn(self, chemical_space: List[LigandInfo]):
        if not self.fixed_space:
            self.initialize(chemical_space)

    def compute_distances(self, X, Y):
        return None

    def __call__(self, smiles_list: Union[List[str], str]):
        if isinstance(smiles_list, list):
            features = torch.cat([self.smile_to_tensor(smiles) for smiles in smiles_list], dim=0)
        else:
            features = self.smile_to_tensor(smiles_list)

        x, _, _ = self.encoder(features)
        return x


class MolDataset(Dataset):
    def __init__(self,
                 chemical_space: List[LigandInfo],
                 featurizer: Featurization,
                 use_only_scored_ligands: bool = True):
        super(MolDataset, self).__init__()
        self.features = []
        self.featurizer = featurizer
        self.chemical_space = chemical_space
        for i in trange(0, len(self.chemical_space)):
            self.features.append(self.featurizer(self.chemical_space[i].smiles))

        self.use_only_scored_ligands = use_only_scored_ligands

        if self.use_only_scored_ligands:
            self.filter_chemical_space(chemical_space)

    def set_dg(self, idx, list_dg):
        for i, score in zip(idx, list_dg):
            self.chemical_space[i].score = score

    def filter_chemical_space(self, chemical_space):
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



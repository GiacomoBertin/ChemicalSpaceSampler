from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple
from Utility import LigandInfo
import torch.nn as nn
from featurization import Featurization, MolDataset
import numpy as np
from sklearn.neighbors import NearestNeighbors
from visualization import smi2fp
from rdkit import DataStructs
from torch.utils.data import Dataset
import torch
from tqdm import trange, tqdm
from torch_geometric.loader import DataLoader


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
        ligs = [smi for smi in self.chemical_space if smi.compound_id not in self.proposed_ids]
        smiles = [smi.smiles for smi in ligs]
        sim = self.featurization.compute_distances(self.chemical_space[self.best_ligand_id].smiles, smiles)
        idx = np.argsort(sim)[1:self.k+1]
        return_dict[self.sampler_id] = [ligs[i] for i in idx]
        self.proposed_ids += [ligs[i].compound_id for i in idx]
        self.proposed_id_history.append(idx)
        return [self.chemical_space[i] for i in idx]

    def closing_step(self, chemical_space: List[LigandInfo]):
        self.featurization.adjourn(chemical_space)
        self.chemical_space = chemical_space
        scores = np.array([smi.score if smi.score is not None else 0. for smi in self.chemical_space])
        self.best_ligand_id = scores.argmin()
        self.best_ligands_history.append(self.best_ligand_id)


class GCNActiveLearning(Sampler):
    def __init__(self,
                 featurization: Featurization,
                 net: nn.Module,
                 criterion = nn.L1Loss(),
                 best_n: int = 10,
                 epochs=100,
                 lr=1e-3,
                 batch_size=32,
                 chkpt_path=None,
                 save_every=50
                 ):
        super(GCNActiveLearning, self).__init__()
        self.featurization = featurization
        self.proposed_ids = []
        self.net = net
        self.criterion = criterion
        self.epochs = epochs
        self.lr = lr
        self.dataset: MolDataset = None
        self.batch_size = batch_size
        self.chkpt_path = chkpt_path
        self.save_every = save_every
        self.best_n = best_n
        self.optimizer = None
        self.predictions = None

    def load_chkpt(self, optimizer):
        if self.chkpt_path is not None:
            chkpt = torch.load(self.chkpt_path)
            self.net.load_state_dict(chkpt['net'])
            starting_epoch = chkpt['epoch']
            optimizer.load_state_dict(chkpt['optimizer'])

        else:
            starting_epoch = 0

        return starting_epoch

    def save_chkpt(self, optimizer, epoch):
        if self.chkpt_path is not None:
            torch.save(self.chkpt_path, dict(
                net=self.net.state_dict(),
                epoch=epoch,
                optimizer=optimizer.state_dict()
            ))

    def train_(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        device = next(self.net.parameters()).device

        starting_epoch = self.load_chkpt(self.optimizer)
        pbar = trange(0, self.epochs, desc='iteration')
        self.net = self.net.train().to(device)

        for epoch in pbar:

            losses = []
            prediction = {}
            for batch in dataloader:
                self.optimizer.zero_grad()

                out = self.net(batch['x'].to(device))
                loss = self.criterion(out, batch['y'][:, None].to(device))

                loss.backward()

                self.optimizer.step()
                losses.append(loss.detach().item())

                for i, k in enumerate(batch['lig']):
                    prediction[k] = out[i].item()

            mean_loss = np.mean(losses)
            pbar.set_description(f'loss: {mean_loss:.2f} ')

            self.predictions = prediction
            if epoch % self.save_every == 0:
                self.save_chkpt(self.optimizer, epoch)
                # print(f'[{epoch:>5d} / {self.epochs:>5d}] | loss: {mean_loss:1.3e}')

    @property
    def device(self):
        return next(self.net.parameters()).device

    @torch.no_grad()
    def predict_(self):
        self.net.eval()
        dataset = MolDataset(list(filter(lambda x: x.compound_id not in self.proposed_ids, self.chemical_space)),
                             self.featurization,
                             use_only_scored_ligands=False)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        scores: List[Tuple[float, LigandInfo]] = []
        device = next(self.net.parameters()).device

        for batch in tqdm(dataloader):
            out = self.net(batch['x'].to(device))

            for i, k in enumerate(batch['lig']):
                scores.append((out[i].item(), self.chemical_space[k]))

        return scores

    def initialize(self, chemical_space: List[LigandInfo], sampler_id):
        self.featurization.initialize(chemical_space)
        self.sampler_id = sampler_id
        self.chemical_space = chemical_space
        self.dataset = MolDataset(self.chemical_space, self.featurization, use_only_scored_ligands=True)
        return self

    def step(self, i, return_dict: dict) -> List[LigandInfo]:
        res = self.predict_()
        scores = [r[0] for r in res]
        idx = np.argsort(scores)[:self.best_n]
        idx = [res[i][1].compound_id for i in idx]
        return_dict[self.sampler_id] = [self.chemical_space[i] for i in idx]
        self.proposed_ids += idx
        return [self.chemical_space[i] for i in idx]

    def closing_step(self, chemical_space: List[LigandInfo]):
        self.featurization.adjourn(chemical_space)
        self.chemical_space = chemical_space
        self.dataset.filter_chemical_space(self.chemical_space)
        self.train_()




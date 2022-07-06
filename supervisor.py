import threading as thread
from autogrow.Operators.Filter.Filter_classes.FilterClasses.Ghose import *
from reactor import *
from multiprocessing import Process, Manager
import multiprocessing as mp
import time
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import json
from samplers import Sampler
from copy import deepcopy
from Utility import LigandInfo
from typing import Dict, List
from logger import Logger
import numpy as np
from score import Score
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from torch.utils.data import Dataset, DataLoader


class Watchdog:
    def __init__(self, n_max_processes):
        self.processes = [None] * n_max_processes
        self.queue = mp.Queue()
        self.n_processes_alive = 0
        self.waiting_processes = []

    def kill_job(self, i):
        print("Killing " + str(self.processes[i].name))
        self.processes[i].terminate()
        while self.processes[i].is_alive():
            time.sleep(0.1)
        self.processes[i].join(timeout=1.0)
        print("Succesfully Killed " + str(self.processes[i].name))

    def submit_process(self, target, args):
        self.waiting_processes.append(Process(target=target, args=args))
        self.check()
        print("Process succesfully queued")

    def kill_jobs(self, idx):
        for i in idx:
            self.kill_job(i)

    def join(self):
        for p in self.processes:
            p.join()

    def check_processes_alive(self):
        self.n_processes_alive = 0
        for i in range(len(self.processes)):
            if self.processes[i] is not None:
                if not self.processes[i].is_alive():
                    self.kill_job(i)
                else:
                    self.n_processes_alive += 1

    def get_free_position(self):
        self.check_processes_alive()
        for k in range(len(self.processes)):
            if (self.processes[k] is None) or (not self.processes[k].is_alive()):
                return k
        return None

    @staticmethod
    def multiprocess_execution(f, args_list: List[List]):
        processes = [mp.Process(target=f, args=args) for args in args_list]
        for p in processes:
            p.start()

        for p in processes:
            p.join()

    def check(self):

        self.check_processes_alive()

        try:
            msg = self.queue.get_nowait()
            if msg[0] == "KILL":
                self.kill_job(msg[1])
        except:
            print("No job to kill")

        for i in range(len(self.waiting_processes)):
            pos = self.get_free_position()
            if pos is not None:
                if pos < len(self.processes) and i < len(self.waiting_processes):
                    self.processes[pos] = self.waiting_processes[i]
                    self.waiting_processes.remove(self.waiting_processes[i])
                    self.processes[pos].start()
                else:
                    print("Something wrong {}, {}".format(pos, i))


class DrugDiscoverySupervisor:
    def __init__(self,
                 starting_smiles,
                 working_directory,
                 protein_pdb,
                 log_file,
                 sampler,
                 score: Score,
                 n_child=1000,
                 n_threads=1,
                 domain_selection=None,
                 n_max_parallel_process=10,
                 lag_iterations=5,
                 best_n_ligands=10,
                 run_id=0
                 ):
        """
        An algorithm for drug discovery.

        :param score:                    function score
        :param starting_smiles:          Initial SMILES
        :param working_directory:        Directory for saving results
        :param protein_pdb:              pdb name of the protein
        :param log_file:
        :param n_child:                  number of molecules to generate for each generation
        :param n_threads:                number of threads for charge calculations
        :param domain_selection:         prody selection string: for example 'protein and chain A'
        :param n_max_parallel_process:   number of processes running simultaneously
        :param lag_iterations:           number of steps before killing
        :param best_n_ligands            use this number of ligands for the next generation
        :param run_id                    run identifier

        """
        self.best_n_ligands = best_n_ligands
        self.score = score
        self.sub_samplers = None
        self.starting_smiles = starting_smiles
        self.n_child = n_child
        self.working_dir = working_directory
        self.n_threads = n_threads
        self.protein_pdb = protein_pdb
        self.log_file = log_file
        self.domain_selection = domain_selection
        self.n_parallel_processes = n_max_parallel_process
        self.lag_iterations = lag_iterations
        self.sampler = sampler

        self.ligands_info_pool = []
        self.childes_smiles = []
        self.childes_info = []

        self.watchdog = None

        self.chemical_space: List[LigandInfo] = None
        self.run_id = run_id
        self.logger = Logger(run_id=self.run_id, working_directory=self.working_dir, restart=True)

    @property
    def computed_ligands(self):
        return [k for k in self.chemical_space if k.score is not None]

    @property
    def all_smiles(self):
        return [k.smiles for k in self.chemical_space]

    @staticmethod
    def __cluster_fps(ms, cutoff=0.2):
        """
        Find the representative structures between the molecules passed as input. Use the Butina algorithm
        and Morgan fingerprint.

        :param ms:      Molecules to cluster
        :param cutoff:  elements within this range of each other are considered
                        to be neighbors
        :return:        a tuple of tuples containing information about the clusters:
                        ( (cluster1_elem1, cluster1_elem2, ...),
                          (cluster2_elem1, cluster2_elem2, ...),
                          ...
                        )
                        The first element for each cluster is its centroid.
        """

        # first generate the distance matrix:
        fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in ms]
        dists = []
        nfps = len(fps)
        print("Generate the distance matrix\n")

        for i in range(1, nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1 - x for x in sims])

        # now cluster the data:
        print("Cluster the data\n")
        cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)

        return np.array(cs)

    @staticmethod
    def __fingerprint_similarity(mol_1, mol_2):
        """
        returns the calculated similarity between two fingerprints,
        """
        return DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(mol_1),
                                                 Chem.RDKFingerprint(mol_2))

    def new_generation(self, parents: List[LigandInfo], n_child, global_step: int = 0):

        generation = []
        for parent in parents:
            smileClick = Reactor(["ClickChem", "", "", ""], [], {})
            childes = []
            # ['COc1ccc2[nH]cc(CCNC(C)OC3(C(C)OC(C)=O)CC4C(CC3(C)O)C4(C)C)c2c1', '1', 'ZINC03921399']

            for i in range(n_child):
                child_info = smileClick.run_Smile_Click(parent.smiles)
                if child_info is not None and self.validate_product(child_info[0]):
                    childes.append(child_info)
                    smileClick.update_list_of_already_made_smiles(childes)
                    ligand = LigandInfo(smiles=child_info[0],
                                        score=None,
                                        compound_id=len(self.chemical_space),
                                        run_id=-1,
                                        chemical_reaction=dict(
                                            parent_smiles=parent.smiles,
                                            parent_id=parent.compound_id,
                                            reaction=child_info[1],
                                            zinc_id=child_info[2]
                                        )
                                        )
                    self.chemical_space.append(ligand)
                    generation.append(ligand)

        self.logger.generate_step(generation, global_step)

    @staticmethod
    def validate_product(mol_, already_computed_mol_protonated=None, similarity_threshold=0.05):
        filter = Ghose()
        if isinstance(mol_, str):
            mol = AllChem.MolFromSmiles(mol_)
        else:
            mol = mol_

        mol_prot = AllChem.AddHs(mol)
        filter_res = filter.run_filter(mol_prot)
        # ohf = OneHotFeaturizer()
        if not filter_res:
            print("Molecule does not pass Goose test")
            return False
        smiles_prot = AllChem.MolToSmiles(mol_prot)
        smile = AllChem.MolToSmiles(AllChem.RemoveHs(mol))
        # try:
        #     ohf.featurize([smile.ljust(120)])
        # except:
        #     print("Molecule failed featurization test")
        #    return False

        if already_computed_mol_protonated is not None:
            for s in already_computed_mol_protonated:
                d = DrugDiscoverySupervisor.__fingerprint_similarity(mol_prot, s)
                if d > (1 - similarity_threshold):
                    print("Molecule too similar at another computed yet")
                    return False
        return True

    @property
    def samplers_ids(self):
        return [self.sub_samplers[i].sampler_id for i in range(len(self.sub_samplers))]

    def mp_samplers_step(self, global_step):
        if len(self.sub_samplers) > 1:
            with mp.Manager() as manager:
                return_dict = manager.dict({self.sub_samplers[i].sampler_id: [] for i in range(len(self.sub_samplers))})
                processes = [mp.Process(target=self.sub_samplers[i].step, args=(global_step, return_dict)) for i in range(len(self.sub_samplers))]
                for p in processes:
                    p.start()

                for p in processes:
                    p.join()

                return_dict = dict(return_dict)
        else:
            return_dict = {}
            self.sub_samplers[0].step(global_step, return_dict)

        return_list = []
        for k in self.samplers_ids:
            for i in range(len(return_dict[k])):
                return_dict[k][i].run_id = int(k)
            return_list += return_dict[k]

        return return_list

    def mp_samplers_adjourns(self, keep_separate_runs=True):
        if len(self.sub_samplers) > 1:
            if not keep_separate_runs:
                processes = [mp.Process(target=self.sub_samplers[i].closing_step,
                                        args=(self.chemical_space,))
                             for i in range(len(self.sub_samplers))]
            else:
                processes = [mp.Process(target=self.sub_samplers[i].closing_step,
                                        args=([
                                                  LigandInfo(smiles=lig.smiles,
                                                             compound_id=lig.compound_id,
                                                             run_id=lig.run_id,
                                                             chemical_reaction=lig.chemical_reaction,
                                                             score=lig.score if lig.run_id == i or lig.run_id < 0 else None)
                                                  for lig in self.chemical_space], )
                                        )
                             for i in range(len(self.sub_samplers))]

            for p in processes:
                p.start()

            for p in processes:
                p.join()
        else:
            self.sub_samplers[0].closing_step(self.chemical_space)

    @property
    def best_drugs(self):
        idx = np.argsort([k.score if k.score is not None else 0. for k in self.chemical_space])
        return idx[:self.best_n_ligands]

    def run(self, n_iterations):
        # Load the chemical space
        self.chemical_space = [LigandInfo(smiles=self.starting_smiles,
                                          compound_id=0,
                                          run_id=-1,
                                          chemical_reaction=None,
                                          score=None)
                               ]

        self.new_generation([self.chemical_space[0]], self.n_child)

        # Initialize the samplers
        self.sub_samplers = [deepcopy(self.sampler).initialize(self.chemical_space, i) for i in range(self.n_parallel_processes)]

        for global_step in range(n_iterations):
            sampled_ligands = self.mp_samplers_step(global_step)

            scores, computed_smiles = self.score(protein_pdb=self.protein_pdb,
                                                 ligands=sampled_ligands,
                                                 domain_selection=self.domain_selection)
            self.logger.compute_step(computed_smiles, scores, global_step)

            for ligand, score in zip(computed_smiles, scores):
                self.chemical_space[ligand.compound_id].score = score

            self.mp_samplers_adjourns()

            if global_step % self.lag_iterations == 0:
                self.new_generation([self.chemical_space[t] for t in self.best_drugs], self.n_child, global_step)

    def run_sampling_test(self, n_iterations: int, smiles: List[str], dg: List[float]):
        assert len(smiles) == len(dg)
        self.chemical_space = [LigandInfo(smiles=smiles[i],
                                          compound_id=i,
                                          run_id=-1,
                                          chemical_reaction=None,
                                          score=None)
                               for i in range(len(smiles))
                               ]

        # Initialize the samplers
        self.sub_samplers = [deepcopy(self.sampler).initialize(self.chemical_space, i) for i in range(self.n_parallel_processes)]

        for global_step in range(n_iterations):
            sampled_ligands = self.mp_samplers_step(global_step)

            scores = [dg[sampled_ligands[i].compound_id] for i in range(len(sampled_ligands))]
            computed_smiles = [sampled_ligands[i].smiles for i in range(len(sampled_ligands))]

            self.logger.compute_step(sampled_ligands, scores, global_step)

            for ligand, score in zip(sampled_ligands, scores):
                self.chemical_space[ligand.compound_id].score = score

            self.mp_samplers_adjourns(keep_separate_runs=False)

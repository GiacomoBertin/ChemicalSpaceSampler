from abc import ABC
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from sklearn import manifold
from Utility import *
from numpy import random
import multiprocessing
from torch import optim
from VAE.featurizer import *
from VAE.models import *
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from typing import List


class Score(ABC):
    working_dir = None

    def set_workingdir(self, working_dir):
        pass

    def valuate(self, protein_pdb, ligands, domain_selection=None):
        pass

    def __call__(self, *args, **kwargs):
        return self.valuate(*args, **kwargs)


class Autodock(Score):
    def __init__(self, output_file=None, folder='input/', keep_ions=True, open_mode='w', n_threads=1):
        self.output_file = output_file if output_file is not None else './output_vina.txt'
        self.folder = folder
        self.keep_ions = keep_ions
        self.open_mode = open_mode
        self.n_threads = n_threads
        self.ligands = []

    def set_workingdir(self, working_dir):
        self.working_dir = working_dir

    def __generate_ligands(self, th_id, from_i, to_i, ligands, ligands_name, ligands_folder, out):
        if not os.path.exists("THREAD_" + str(th_id)):
            os.mkdir("THREAD_" + str(th_id))
        os.chdir("THREAD_" + str(th_id))
        print("running thread " + str(th_id))
        for i in range(from_i, to_i):
            print("generating ligand " + ligands_name[i])
            out[i] = Ligand(ligands[i].smiles, ligands_name[i], ligands_folder[i])
        os.chdir("../")
        # os.system("rm -r THREAD_" + str(th_id))

    def valuate(self, protein_pdb, ligands: List[LigandInfo], domain_selection=None):
        lig_name = 'LIG'
        name = protein_pdb.split('/')[len(protein_pdb.split('/')) - 1].split('.')[0]
        protein = Protein(protein_pdb, name, True, self.folder)
        dG = []

        if not os.path.exists(os.path.join(self.folder, "ligands")):
            os.mkdir(os.path.join(self.folder, "ligands"))
        if not os.path.exists(os.path.join(self.folder, "complexes")):
            os.mkdir(os.path.join(self.folder, "complexes"))

        self.ligands = [None] * len(ligands)
        ligands_name = []
        ligands_folder = []
        for k in range(len(ligands)):
            if ligands[k].compound_id is not None:
                ligand_folder = os.path.join(self.folder, "ligands", str(ligands[k].compound_id))
            else:
                ligand_folder = os.path.join(self.folder, "ligands", str(k))
            ligands_name.append(lig_name)
            ligands_folder.append(ligand_folder)

        # pool = multiprocessing.Pool(processes=len(ligands)) pool.map(target=self.__generate_ligands, [(i,
        # ligands[i], ligands_name[i], ligands_folder[i], self.ligands[i],) for i in range(len(ligands))])
        # TODO Use a Pool object

        threads = [multiprocessing.Process(target=self.__generate_ligands,
                                           args=(i,
                                                 int(i * len(ligands) / self.n_threads),
                                                 int((i + 1) * len(ligands) / self.n_threads),
                                                 ligands, ligands_name, ligands_folder, self.ligands,))
                   for i in range(self.n_threads)]

        for th in threads:
            th.start()

        for th in threads:
            th.join()

        ligands_computed = []
        for k in range(len(ligands)):
            # try:
            output = open(self.output_file, 'a')
            if ligands[k].compound_id is not None:
                id = ligands[k].compound_id
            else:
                id = k

            ligand = Ligand(ligands[k].smiles, ligands_name[k], ligands_folder[k])
            ligands_computed.append(ligands[k])

            if ligands[k].compound_id is not None:
                complex_folder = os.path.join(self.folder, "complexes", str(ligands[k].compound_id))
                complex_name = protein.name + "_" + str(ligands[k].compound_id)
            else:
                complex_folder = os.path.join(self.folder, "complexes", str(k))
                complex_name = name + "_" + str(k)

            _complex = Complex(protein, ligand, complex_name,
                               working_dir=complex_folder,
                               domain_selection=domain_selection)

            output.write(str(k) + " " + ligand.smile + " " + str(_complex.dG_autodock) + "\n")
            output.close()
            dG.append(_complex.dG_autodock)
            # except e:
            #    print("error valutate autodock")
        return dG, ligands_computed
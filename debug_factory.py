import torch

from factory import *
from net_utility import test_sampler, plot_violin, evaluate_net, evaluate_VAE
import os.path as osp


def load_dataset(file_dg, file_smiles):
    dataset = {}
    with open(file_dg) as f_dg:
        for line in f_dg:
            w = line.split()
            name = w[0][w[0].index('_') + 1:].replace('_', ' ')
            dg = float(w[1])
            dataset[name] = {'dg': dg, 'smiles': None}

    smiles = []
    dg = []
    with open(file_smiles) as f_smiles:
        for line in f_smiles:
            w = line.split('\t')
            if len(w) == 3 and w[0] in dataset.keys():
                name = w[0]
                dataset[name]['smiles'] = w[-1].replace('\n', '')
                smiles.append(dataset[name]['smiles'])
                dg.append(dataset[name]['dg'])

    return dataset, smiles, dg


dataset, smiles, dg = load_dataset('./DatabaseOMSDrugs_scores.dat', './DatabaseOMSDrugs.dat')

supervisor = build_supervisor('config/config_OHNet_RUN_0.yml')
supervisor.run(100)



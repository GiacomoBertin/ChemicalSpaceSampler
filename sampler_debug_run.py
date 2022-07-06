from samplers import GCNActiveLearning, KNNSampler
from featurization import GraphFeaturization, FingerprintFeaturization
from supervisor import DrugDiscoverySupervisor
from models import GATNet
import torch.nn as nn
import torch.nn.functional as F

featurization = GraphFeaturization(
    use_edges=True,
    use_chirality=True,
    use_partial_charge=True,
    graph_lib='pyg'
)

featurization = FingerprintFeaturization()

test = 'COc1ccc2[nH]cc(CCN(C(C)=O)C(=S)NC(C)c3ccc4c(c3)CCC4)c2c1'
out = featurization(test)
print(out)

net = GATNet(
    input_channels=33,
    edge_dim=11,
    output_channels=1,
    hidden_channels=64,
    num_layers=2,
    activation=nn.ELU,
    use_batch_norm=True,
    heads=2,
    concat=False,
    dropout=0.2,
    num_final_layers=3
)

criterion = nn.L1Loss()

sampler = GCNActiveLearning(
    featurization,
    net,
    criterion,
    best_n=32,
    epochs=10,
    lr=1e-3,
    batch_size=32,
    chkpt_path=None,
    save_every=50
).cuda()

sampler = KNNSampler(k=6, featurization=featurization, sampler_id=0)

supervisor = DrugDiscoverySupervisor(
    starting_smiles=[],
    working_directory='/home/giacomo/Documents/LCP_runs/sampler_test',
    protein_pdb=None,
    log_file='./log.dat',
    sampler=sampler,
    score=None,
    n_child=1000,
    n_threads=1,
    domain_selection=None,
    n_max_parallel_process=1,
    lag_iterations=5,
    best_n_ligands=10,
    run_id=0

)


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
supervisor.run_sampling_test(n_iterations=20, smiles=smiles, dg=dg)

from reactor import Reactor
from samplers import KNNSampler
from supervisor import DrugDiscoverySupervisor
from featurization import PCAFeaturization
from score import Autodock
import os.path as osp

featurization = PCAFeaturization(n_components=2)
folder = '/home/giacomo/Documents/LCP_runs'
sampler = KNNSampler(
    k=4,
    featurization=featurization,
    sampler_id=0
)

score = Autodock(output_file=osp.join(folder, 'docking_res.txt'), folder=osp.join(folder, 'docking'), keep_ions=True, open_mode='w', n_threads=3)

supervisor = DrugDiscoverySupervisor(
    starting_smiles='CCOC(=O)C1=CC=C(N)C=C1',
    working_directory=folder,
    protein_pdb='4UYG',
    log_file=osp.join(folder, 'log.txt'),
    sampler=sampler,
    score=score,
    n_child=100,
    n_threads=4,
    domain_selection='protein',
    n_max_parallel_process=1,
    lag_iterations=5,
    best_n_ligands=10,
    run_id=0
)

supervisor.run(10000)


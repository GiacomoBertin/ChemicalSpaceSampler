import os.path as osp
from pathlib import Path
from typing import List
from Utility import LigandInfo
import redis
import json


class Logger:
    def __init__(self, run_id=0, working_directory='./', restart=True):
        self.working_directory = working_directory
        self.run_id = run_id
        self.log_file = osp.join(self.working_directory, f'log_run_{self.run_id}.log')
        self.log_dictionary = osp.join(self.working_directory, f'log_run_{self.run_id}.json')
        self.results = {}
        self.computed_ids = []

        self.computed_steps_ids = []
        self.generated_steps_ids = []
        self.r = redis.Redis(db=run_id)
        if restart:
            self.reset()

    def generate_step(self, generated_ligand: List[LigandInfo], global_step: int = 0):
        self.r.set('generate_step', global_step)

        for ligand in generated_ligand:
            self.r.set(ligand.compound_id, json.dumps(ligand._asdict()))
            self.results[ligand.compound_id] = ligand

        self.generated_steps_ids.append(len(self.results))
        self.r.set('num_ligands', len(self.results))
        self.r.set('generated_steps_ids', json.dumps(self.generated_steps_ids))

    def compute_step(self, computed_ligand: List[LigandInfo], scores: List[float], global_step: int = 0):
        self.r.set('computed_step', global_step)
        computed_id = []
        for ligand, score in zip(computed_ligand, scores):
            computed_id.append(ligand.compound_id)
            self.results[ligand.compound_id] = ligand
            self.results[ligand.compound_id].score = score
            self.r.set(ligand.compound_id, json.dumps(self.results[ligand.compound_id]._asdict()))
        self.computed_steps_ids.append(computed_id)
        self.r.set('computed_steps_ids', json.dumps(self.computed_steps_ids))

    def reset(self):
        self.results.clear()
        self.computed_ids.clear()

        self.computed_steps_ids.clear()
        self.generated_steps_ids.clear()

        self.r.flushdb()
        self.r.set('computed_step', 0)
        self.r.set('generate_step', 0)

    def log(self):
        pass

    def to_json(self):
        pass

    def save_step(self):
        pass








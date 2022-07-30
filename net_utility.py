from supervisor import DrugDiscoverySupervisor
import numpy as np
import torch
from typing import Dict, List
import plotly.graph_objects as go


def test_sampler(supervisor: DrugDiscoverySupervisor,
                 smiles,
                 dg,
                 num_iterations=100,
                 num_reruns=10,
                 file_name=None):

    min_dg = [[] for _ in range(num_iterations)]
    for run_id in range(num_reruns):
        scores_at_step, smiles_at_step = supervisor.run_sampling_test(num_iterations, smiles, dg)

        for i in range(len(scores_at_step)):
            min_dg_i = np.min(scores_at_step[:i+1])
            min_dg[i].append(min_dg_i)

        if file_name is not None:
            torch.save(dict(min_dg=min_dg), file_name)

    return min_dg


def plot_violin(min_dg_dict: Dict):
    fig = go.Figure()

    for name in min_dg_dict.keys():
        min_dg = min_dg_dict[name]
        x = np.concatenate([np.array([i for _ in range(len(min_dg[i]))]) for i in range(len(min_dg))], axis=0)
        y = np.concatenate([np.array(min_dg[i]) for i in range(len(min_dg))], axis=0)

        fig.add_trace(go.Violin(x=x,
                                y=y,
                                legendgroup=name, scalegroup=name, name=name,
                                side='negative')
                      )

    fig.update_traces(meanline_visible=True)
    fig.update_layout(violingap=0, violinmode='overlay')

    return fig






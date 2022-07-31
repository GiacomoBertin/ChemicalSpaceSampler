from supervisor import DrugDiscoverySupervisor
import numpy as np
import torch
from typing import Dict, List
import plotly.graph_objects as go
from torch.utils.data import Dataset
from featurization import Featurization, MolDataset, VAEFeaturization
from Utility import LigandInfo
import torch
from tqdm import trange, tqdm
from torch_geometric.loader import DataLoader
from copy import deepcopy
from factory import build_supervisor
import torch.nn as nn


def test_sampler(supervisor_file,
                 smiles,
                 dg,
                 num_iterations=100,
                 num_reruns=10,
                 file_name=None):
    min_dg = [[] for _ in range(num_iterations)]
    for run_id in range(num_reruns):
        supervisor_ = build_supervisor(supervisor_file)
        scores_at_step, smiles_at_step = supervisor_.run_sampling_test(num_iterations, smiles, dg)

        for i in range(len(scores_at_step)):
            min_dg_i = np.min(scores_at_step[:i + 1])
            min_dg[i].append(min_dg_i)

        if file_name is not None:
            torch.save(dict(min_dg=min_dg), file_name)

    return min_dg


def run_net(net, dataset, batch_size=64):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    device = next(net.parameters()).device

    result = []
    for batch in tqdm(dataloader):
        out = net(batch['x'].to(device))

        for i, k in enumerate(batch['lig']):
            result.append((out[i].item(), batch['y'][i].item()))

    return result


def run_VAE(encoder, decoder, dataset, batch_size=64):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    device = next(encoder.parameters()).device

    encoder.eval()
    decoder.eval()

    result = []
    for batch in tqdm(dataloader):
        z = encoder(batch['x'].to(device))
        recon_x = decoder(z)

        for i, k in enumerate(batch['lig']):
            result.append((recon_x[i].cpu(), batch['x'][i].cpu()))

    encoder.train()
    decoder.train()

    return result


def evaluate_net(model: torch.nn.Module,
                 criterion,
                 featurization,
                 smiles,
                 dg,
                 n_incrementation=100,
                 batch_size=64,
                 lr=1e-3,
                 epochs=50,
                 n_max_test=5,
                 ):
    chemical_space = [LigandInfo(smiles=smiles[i],
                                 compound_id=i,
                                 run_id=-1,
                                 chemical_reaction=None,
                                 score=dg[i]
                                 )
                      for i in range(len(smiles))]
    dataset = MolDataset(chemical_space, featurization, use_only_scored_ligands=True)
    train_length = n_incrementation
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    all_results = {}
    num = 0
    all_losses = {}
    while (train_length < len(dataset)) and (num < n_max_test):
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_length, len(dataset) - train_length])

        train_length += n_incrementation

        net = deepcopy(model)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        pbar = trange(0, epochs, desc='iteration')
        net = net.train().to(device)

        for epoch in pbar:

            losses = []
            prediction = {}
            for batch in dataloader:
                optimizer.zero_grad()

                out = net(batch['x'].to(device))
                loss = criterion(out, batch['y'][:, None].to(device))

                loss.backward()

                optimizer.step()
                losses.append(loss.detach().item())

                for i, k in enumerate(batch['lig']):
                    prediction[k] = out[i].item()

            mean_loss = np.mean(losses)
            pbar.set_description(f'loss: {mean_loss:.2f} ')

        results = run_net(net, test_dataset, batch_size)
        dat = torch.tensor(results)
        all_losses[len(train_dataset)] = criterion(dat[:, 0], dat[:, 1])
        print(all_losses)
        all_results[len(train_dataset)] = results

        num += 1
    return all_results, all_losses


def multiple_evaluations(model: torch.nn.Module,
                         criterion,
                         featurization,
                         smiles,
                         dg,
                         n_incrementation=100,
                         batch_size=64,
                         lr=1e-3,
                         epochs=50,
                         n_max_test=5,
                         num_runs=2,
                         file_path=None
                         ):
    all_eval_results = {}
    all_losses = {}
    for i in range(num_runs):
        eval_results, losses = evaluate_net(model,
                                            criterion,
                                            featurization,
                                            smiles,
                                            dg,
                                            n_incrementation=n_incrementation,
                                            batch_size=batch_size,
                                            lr=lr,
                                            epochs=epochs,
                                            n_max_test=n_max_test
                                            )

        for k in eval_results.keys():
            if k not in all_eval_results.keys():
                all_eval_results[k] = [eval_results[k]]
                all_losses[k] = [losses[k]]
            else:
                all_eval_results[k].append(eval_results[k])
                all_losses[k].append(losses[k])

    for k in all_eval_results.keys():
        all_eval_results[k] = torch.tensor(all_eval_results[k]).T
        all_losses[k] = torch.tensor(all_losses[k]).T

    if file_path is not None:
        torch.save(dict(eval_results=all_eval_results, losses=all_losses), file_path)


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


def plot_box(min_dg_dict: Dict):
    fig = go.Figure()

    for name in min_dg_dict.keys():
        min_dg = min_dg_dict[name]
        x = np.concatenate([np.array([i for _ in range(len(min_dg[i]))]) for i in range(len(min_dg))], axis=0)
        y = np.concatenate([np.array(min_dg[i]) for i in range(len(min_dg))], axis=0)

        fig.add_trace(go.Box(x=x,
                             y=y,
                             name=name)
                      )
    fig.update_layout(boxmode='group')
    fig.update_layout(
        title="Sampling Progresses",
        xaxis_title="Step",
        yaxis_title="Min Score",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="rgba(255,255,255,1)"
        )
    )

    return fig


def plot_loss_vs_size(losses: Dict):
    fig = go.Figure()

    for name in losses.keys():
        loss_ = losses[name]
        y = np.concatenate([loss_[k].numpy() for k in loss_.keys()], axis=0)
        x = np.concatenate([np.array([k for _ in range(len(loss_[k]))]) for k in loss_.keys()], axis=0)

        fig.add_trace(go.Box(x=x,
                             y=y,
                             name=name)
                      )
    fig.update_layout(boxmode='group')
    fig.update_layout(
        title="L1 Loss vs Dataset Size",
        xaxis_title="Dataset Size",
        yaxis_title="L1 Loss",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="rgba(255,255,255,1)"
        )
    )

    return fig


def compute_loss(criterion, y_hat, y, mu, log_var):
    recon_loss = criterion(y_hat, y)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + KLD


def evaluate_VAE(encoder,
                 decoder,
                 criterion,
                 featurization,
                 smiles,
                 n_incrementation=100,
                 batch_size=64,
                 lr=1e-3,
                 epochs=50,
                 n_max_test=5,
                 ):
    chemical_space = [LigandInfo(smiles=smiles[i],
                                 compound_id=i,
                                 run_id=-1,
                                 chemical_reaction=None,
                                 score=None
                                 )
                      for i in range(len(smiles))]
    dataset = MolDataset(chemical_space, featurization, use_only_scored_ligands=False)

    num = 0
    train_length = n_incrementation

    all_results = {}
    all_losses = {}

    while (train_length < len(dataset)) and (num < n_max_test):
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_length, len(dataset) - train_length])

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        pbar = trange(0, epochs, desc='iteration')
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        encoder_ = deepcopy(encoder).to(device)
        decoder_ = deepcopy(decoder).to(device)
        optimizer = torch.optim.Adam(nn.ModuleList([encoder_, decoder_]).parameters(), lr=lr)

        for epoch in pbar:
            losses = []
            for batch_idx, batch in enumerate(dataloader):
                # print(vae(batch)[0].shape)
                optimizer.zero_grad()
                data = batch['x'].to(device)
                z, mu, log_var = encoder_(data)
                recon_data = decoder_(z)

                loss = compute_loss(criterion, recon_data, data, mu=mu, log_var=log_var)
                loss.backward()
                losses.append(loss.detach().item())

                optimizer.step()

            mean_loss = np.mean(losses)
            pbar.set_description(f'loss: {mean_loss:.2f} ')

        results = run_VAE(encoder_, decoder_, test_dataset, batch_size)

        dat_recon = torch.stack([r[0] for r in results], dim=0)
        dat_real = torch.stack([r[1] for r in results], dim=0)

        all_losses[len(train_dataset)] = criterion(dat_recon, dat_real).cpu().item()
        all_results[len(train_dataset)] = results

        print(all_losses)

        train_length += n_incrementation
        num += 1

    return all_results, all_losses


def multiple_evaluations_VAE(encoder,
                             decoder,
                             criterion,
                             featurization,
                             smiles,
                             n_incrementation=100,
                             batch_size=64,
                             lr=1e-3,
                             epochs=50,
                             n_max_test=5,
                             num_runs=2,
                             file_path=None
                             ):
    all_eval_results = {}
    all_losses = {}
    for i in range(num_runs):
        eval_results, losses = evaluate_VAE(encoder,
                                            decoder,
                                            criterion,
                                            featurization,
                                            smiles,
                                            n_incrementation=n_incrementation,
                                            batch_size=batch_size,
                                            lr=lr,
                                            epochs=epochs,
                                            n_max_test=n_max_test
                                            )

        for k in losses.keys():
            if k not in all_losses.keys():
                all_eval_results[k] = [eval_results[k]]
                all_losses[k] = [losses[k]]
            else:
                all_eval_results[k].append(eval_results[k])
                all_losses[k].append(losses[k])

    for k in all_losses.keys():
        all_eval_results[k] = None  # torch.tensor(all_eval_results[k]).T
        all_losses[k] = torch.tensor(all_losses[k]).T

    if file_path is not None:
        torch.save(dict(eval_results=all_eval_results, losses=all_losses), file_path)

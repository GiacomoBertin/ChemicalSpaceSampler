from samplers import Sampler
from score import Score
from featurization import Featurization
from supervisor import DrugDiscoverySupervisor
from models import GATNet
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
import inspect
import importlib
import yaml


def get_module_and_kwargs(module):
    my_module = importlib.import_module(module)
    classes = [m[0] for m in inspect.getmembers(my_module, inspect.isclass) if (m[1].__module__ == module) and (not inspect.isabstract(getattr(my_module, m[0])))]
    args = {c: inspect.getfullargspec(getattr(my_module, c)) for c in classes}
    return args, my_module


def build_components(args_spec: inspect.FullArgSpec, kwargs):
    for k in kwargs.keys():
        if k in args_spec.annotations.keys():
            if args_spec.annotations[k] == Featurization:
                kwargs[k] = build_featurization(kwargs[k]['name'], kwargs[k]['kwargs'])
            elif args_spec.annotations[k] == nn.Module:
                kwargs[k] = build_module(kwargs[k]['name'], kwargs[k]['kwargs'])
            elif args_spec.annotations[k] == Sampler:
                kwargs[k] = build_sampler(kwargs[k]['name'], kwargs[k]['kwargs'])
            elif args_spec.annotations[k] == Score:
                kwargs[k] = build_score(kwargs[k]['name'], kwargs[k]['kwargs'])

    return kwargs


def build_module(name, kwargs: dict):
    args, module = get_module_and_kwargs('models')
    kwargs = build_components(args[name], kwargs)
    net = getattr(module, args[name])(**kwargs)
    return net


def build_featurization(name, kwargs: dict):
    args, module = get_module_and_kwargs('featurization')
    kwargs = build_components(args[name], kwargs)
    feat = getattr(module, args[name])(**kwargs)
    return feat


def build_sampler(name, kwargs):
    args, module = get_module_and_kwargs('samplers')
    kwargs = build_components(args[name], kwargs)
    sampler = getattr(module, args[name])(**kwargs)
    return sampler


def build_score(name, kwargs):
    args, module = get_module_and_kwargs('score')
    score = getattr(module, args[name])(**kwargs)
    return score


def build_supervisor(config_file):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    cfg = EasyDict()

    # Copy
    for k, v in config.items():
        cfg[k] = v

    kwargs_sup = config['supervisor']
    args, module = get_module_and_kwargs('supervisor')
    kwargs_sup = build_components(args['DrugDiscoverySupervisor'], kwargs_sup)
    return DrugDiscoverySupervisor(**kwargs_sup)

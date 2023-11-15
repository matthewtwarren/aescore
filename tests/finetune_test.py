import os

import mlflow
import numpy as np
import torch
from torch import nn, optim
import copy
import torchani

from ael import loaders, models, train, finetune, utils

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = False  # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Radial coefficients
RcR = 5.2
EtaR = torch.tensor([16.0], device=device)
RsR = torch.tensor([0.9], device=device)

# Angular coefficients (Ga)
RcA = 3.5
Zeta = torch.tensor([32], device=device)
TsA = torch.tensor([0.19634954], device=device)  # Angular shift in GA
EtaA = torch.tensor([8.0], device=device)
RsA = torch.tensor([0.9], device=device)  # Radial shift in GA

def test_freeze_single_layer(testdata, testdir):

    # Distance 0.0 produces a segmentation fault (see MDAnalysis#2656)
    data = loaders.PDBData(testdata, 0.1, testdir)

    batch_size = 2

    # Transform atomic numbers to species
    amap = loaders.anummap(data.species)
    data.atomicnums_to_idxs(amap)

    n_species = len(amap)

    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=loaders.pad_collate
    )

    # Define AEVComputer
    AEVC = torchani.AEVComputer(RcR, RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, n_species)

    affinity_model = models.AffinityModel(n_species, AEVC.aev_length) # Default model has 4 layers, no dropouts

    finetune.freeze_layer_params(affinity_model, [0])
    
    for i in affinity_model:
        
        model = affinity_model[i]

        assert not model.layers[0].weight.requires_grad # Trainable layer 0
        assert not model.layers[0].bias.requires_grad
        assert model.layers[2].weight.requires_grad # Trainable layer 1
        assert model.layers[2].bias.requires_grad
        assert model.layers[4].weight.requires_grad # Trainable layer 2
        assert model.layers[4].bias.requires_grad
        assert model.layers[6].weight.requires_grad # Trainable layer 3
        assert model.layers[6].bias.requires_grad


def test_freeze_single_layer_dropout(testdata, testdir):

    # Distance 0.0 produces a segmentation fault (see MDAnalysis#2656)
    data = loaders.PDBData(testdata, 0.1, testdir)

    batch_size = 2

    # Transform atomic numbers to species
    amap = loaders.anummap(data.species)
    data.atomicnums_to_idxs(amap)

    n_species = len(amap)

    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=loaders.pad_collate
    )

    # Define AEVComputer
    AEVC = torchani.AEVComputer(RcR, RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, n_species)

    affinity_model = models.AffinityModel(n_species, AEVC.aev_length, dropp=0.5)

    finetune.freeze_layer_params(affinity_model, [0])
    
    for i in affinity_model:
        
        model = affinity_model[i]

        assert not model.layers[0].weight.requires_grad # Trainable layer 0
        assert not model.layers[0].bias.requires_grad
        assert model.layers[3].weight.requires_grad # Trainable layer 1
        assert model.layers[3].bias.requires_grad
        assert model.layers[6].weight.requires_grad # Trainable layer 2
        assert model.layers[6].bias.requires_grad
        assert model.layers[9].weight.requires_grad # Trainable layer 3
        assert model.layers[9].bias.requires_grad


def test_freeze_multiple_layers(testdata, testdir):

    # Distance 0.0 produces a segmentation fault (see MDAnalysis#2656)
    data = loaders.PDBData(testdata, 0.1, testdir)

    batch_size = 2

    # Transform atomic numbers to species
    amap = loaders.anummap(data.species)
    data.atomicnums_to_idxs(amap)

    n_species = len(amap)

    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=loaders.pad_collate
    )

    # Define AEVComputer
    AEVC = torchani.AEVComputer(RcR, RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, n_species)

    affinity_model = models.AffinityModel(n_species, AEVC.aev_length, dropp=0.5)

    finetune.freeze_layer_params(affinity_model, [1, 3])

    for i in affinity_model:
        
        model = affinity_model[i]

        assert model.layers[0].weight.requires_grad
        assert model.layers[0].bias.requires_grad
        assert not model.layers[3].weight.requires_grad
        assert not model.layers[3].bias.requires_grad
        assert model.layers[6].weight.requires_grad
        assert model.layers[6].bias.requires_grad
        assert not model.layers[9].weight.requires_grad
        assert not model.layers[9].bias.requires_grad


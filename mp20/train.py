#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('../cace/')

import numpy as np
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import logging

import cace
from cace.representations import Cace
from cace.modules import CosineCutoff, MollifierCutoff, PolynomialCutoff
from cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered
from cace.modules.type import ElementEncoder
from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask

torch.set_default_dtype(torch.float32)
#torch.autograd.set_detect_anomaly(True)

cace.tools.setup_logger(level='INFO')

import ase

collection = cace.tasks.get_dataset_from_xyz(train_path='../mp20/mp-train-s-1-augmented-nonoise.xyz',
                                 valid_fraction=0.1,
                                 energy_key='ee',
                                 forces_key='forces',
                                 stress_key='stress',
                                              )

cutoff = 4.0
batch_size = 10

train_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='train',
                              batch_size=batch_size,
                              cutoff=cutoff)

valid_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='valid',
                              batch_size=20,
                              cutoff=cutoff)

del collection

use_device = 'cuda'
device = cace.tools.init_device(use_device)
logging.info(f"device: {use_device}")


logging.info("building CACE representation")
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
#cutoff_fn = CosineCutoff(cutoff=cutoff)
cutoff_fn = PolynomialCutoff(cutoff=cutoff, p=2)
node_encoder = ElementEncoder()

cace_representation = Cace(
    zs=[ i for i in range(1,95)],
    #node_encoder=node_encoder,
    n_atom_basis=4,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=8,
    max_l=2,
    max_nu=2,
    device=device,
    num_message_passing=0,
    timeit=False
           )

cace_representation.to(device)
logging.info(f"Representation: {cace_representation}")

atomwise = cace.modules.Atomwise(n_layers=3,
                                 output_key='CACE_energy',
                                 n_hidden=[32,16],
                                 residual=False,
                                 use_batchnorm=False,
                                 add_linear_nn = True)

forces = cace.modules.forces.Forces(energy_key='CACE_energy',
                                    forces_key='CACE_forces',
                                   stress_key='CACE_stress',
                                   calc_stress=True)
preprocessor = cace.modules.Preprocess()

logging.info("building CACE NNP")
cace_nnp = NeuralNetworkPotential(
    input_modules=[preprocessor],
    representation=cace_representation,
    output_modules=[atomwise, forces]
)

#trainable_params = sum(p.numel() for p in cace_nnp.parameters() if p.requires_grad)
#logging.info(f"Number of trainable parameters: {trainable_params}")

cace_nnp.to(device)


logging.info(f"First train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=10
)

force_loss = cace.tasks.GetLoss(
    target_name='forces',
    predict_name='CACE_forces',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)

energy_loss_2 = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)

stress_loss = cace.tasks.GetLoss(
    target_name='stress',
    predict_name='CACE_stress',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=10000
)

from cace.tools import Metrics

e_metric = Metrics(
    target_name='energy',
    predict_name='CACE_energy',
    name='e',
    per_atom=False
)

f_metric = Metrics(
    target_name='forces',
    predict_name='CACE_forces',
    name='f'
)

s_metric = Metrics(
    target_name='stress',
    predict_name='CACE_stress',
    name='s'
)

# Example usage
logging.info("creating training task")

optimizer_args = {'lr': 1e-2}  
scheduler_args = {'mode': 'min', 'factor': 0.8, 'patience': 10}

for _ in range(5):
    task = TrainingTask(
        model=cace_nnp,
        losses=[force_loss, stress_loss],
        metrics=[f_metric, s_metric],
        device=device,
        optimizer_args=optimizer_args,
        #scheduler_cls=torch.optim.lr_scheduler.StepLR,
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args=scheduler_args,
        max_grad_norm=10,
        ema=True,
        ema_start=10,
        warmup_steps=10,
    )

    logging.info("training")
    task.fit(train_loader, valid_loader, epochs=70, screen_nan=False)

logging.info("Finished!")

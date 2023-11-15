import os
from typing import Union
import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
import json
import matplotlib.pyplot as plt
from ael import argparsers, loaders, models, utils, train, predict, plot


def freeze_layer_params(model: models.AffinityModel, freeze_layers: list):
    """Freeze the weights and biases of specified trainable layers.

    Parameters
    ----------
    model : models.AffinityModel
        Model to be finetuned.
    freeze : list
        List of indices of trainable layers whose parameters should be frozen. Indexing starts at 0.

    """
    trainable_layers = []
    
    for m in model:
        atomicmodel = model[m]
        for i, layer in enumerate(atomicmodel.layers):
            if any(param.requires_grad for param in layer.parameters()):
                trainable_layers.append((i, layer))

    if len(freeze_layers) > len(trainable_layers):
        raise ValueError(f"Cannot freeze {len(freeze_layers)} layers, only {len(trainable_layers)} are trainable")
    
    elif len(freeze_layers) == len(trainable_layers):
        raise ValueError("Cannot freeze all layers, at least one must be trainable")
    
    elif max(freeze_layers) >= len(trainable_layers):
        raise ValueError(f"Cannot freeze layer {max(freeze_layers)}, only {len(trainable_layers)} are trainable")

    layer_indices_to_freeze = [trainable_layers[i][0] for i in freeze_layers if i < len(trainable_layers)]

    for i, layer in trainable_layers:
        if i in layer_indices_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    
    return model

if __name__ == "__main__":
    args = argparsers.finetuneparser()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True  # type: ignore
        cudnn.benchmark = False  # type: ignore

    mlflow.set_experiment(args.experiment)

    # Start MLFlow run (named finetune)
    with mlflow.start_run(run_name="finetune"):  # Check this works
        mlflow.log_param("device", args.device)
        mlflow.log_param("random_seed", args.seed)

        mlflow.log_param("distance", args.distance)
        mlflow.log_param("trainfile", args.trainfile)
        mlflow.log_param("validfile", args.validfile)
        mlflow.log_param("datapaths", args.datapaths)

        mlflow.log_param("batchsize", args.batchsize)
        mlflow.log_param("lr", args.lr)
        # mlflow.log_param("layers", args.layers) # Lets leave this out for now ...
        mlflow.log_param("dropout", args.dropout)

        if args.chemap is not None:
            with open(args.chemap, "r") as fin:
                cmap = json.load(fin)
        else:
            cmap = None

        if args.vscreening is None:
            traindata: Union[loaders.PDBData, loaders.VSData] = loaders.PDBData(
                args.trainfile,
                args.distance,
                args.datapaths,
                cmap,
                desc="Training set",
                removeHs=args.removeHs,
                ligmask=args.ligmask,
            )
            validdata: Union[loaders.PDBData, loaders.VSData] = loaders.PDBData(
                args.validfile,
                args.distance,
                args.datapaths,
                cmap,
                desc="Validation set",
                removeHs=args.removeHs,
                ligmask=args.ligmask,
            )
        else:
            traindata = loaders.VSData(
                args.trainfile,
                args.distance,
                args.datapaths,
                cmap,
                desc="Training set",
                removeHs=args.removeHs,
                labelspath=args.vscreening,
            )
            validdata = loaders.VSData(
                args.validfile,
                args.distance,
                args.datapaths,
                cmap,
                desc="Validation set",
                removeHs=args.removeHs,
                labelspath=args.vscreening,
            )
        if args.testfile is not None:
            if args.vscreening is None:
                testdata: Union[loaders.PDBData, loaders.VSData] = loaders.PDBData(
                    args.testfile,
                    args.distance,
                    args.datapaths,
                    cmap,
                    desc="Test set",
                    removeHs=args.removeHs,
                    ligmask=args.ligmask,
                )
            else:
                testdata = loaders.VSData(
                    args.testfile,
                    args.distance,
                    args.datapaths,
                    cmap,
                    desc="Test set",
                    removeHs=args.removeHs,
                    labelspath=args.vscreening,
                )

        amap = utils.load_amap(
            args.amap
        )  # We might want to add a check that atoms in the finetuning dataset are the same as in the training dataset

        if args.scale:
            if args.testfile is None:
                scaler = utils.labels_scaler(traindata, validdata)
            else:
                scaler = utils.labels_scaler(traindata, validdata, testdata)
        else:
            scaler = None

        # Transform atomic numbers to 0-based indices
        traindata.atomicnums_to_idxs(amap)
        validdata.atomicnums_to_idxs(amap)

        if args.testfile is not None:
            testdata.atomicnums_to_idxs(amap)

        n_species = len(amap)

        mlflow.log_param("n_species", n_species)

        trainloader = data.DataLoader(
            traindata,
            batch_size=args.batchsize,
            shuffle=True,
            collate_fn=loaders.pad_collate,
        )

        validloader = data.DataLoader(
            validdata,
            batch_size=args.batchsize,
            shuffle=True,
            collate_fn=loaders.pad_collate,
        )

        if args.testfile is not None:
            testloader = data.DataLoader(
                testdata,
                batch_size=args.batchsize,
                shuffle=False,
                collate_fn=loaders.pad_collate,
            )

        AEVC = utils.loadAEVC(args.aev)

        models = [
            utils.loadmodel(m, eval=False) for m in args.models
        ] 
        optimizers_list = []

        for idx, model in enumerate(models):
            optimizers_list.append(torch.optim.Adam(model.parameters(), lr=args.lr))
            mse = nn.MSELoss()
            model = freeze_layer_params(model, freeze_layers=args.freeze)

            train_losses, valid_losses = train.train(
                model,
                optimizers_list[idx],
                mse,
                AEVC,
                trainloader,
                validloader,
                epochs=args.epochs,
                savepath=args.outpath,
                idx=None if len(models) == 1 else idx,
            )

            # Save training and validation losses
            if len(models) == 1:
                fname_loss = os.path.join(args.outpath, "loss.dat")
            else:
                fname_loss = os.path.join(args.outpath, f"loss_{idx}.dat")

            np.savetxt(
                fname_loss,
                np.stack((train_losses, valid_losses), axis=-1),
                fmt="%.6e",
                header="tran_losses, valid_losses",
            )

            mlflow.log_artifact(fname_loss)

            if args.plot:
                for ext in [".png", ".pdf"]:
                    e = np.arange(args.epochs)

                    plt.figure()
                    plt.plot(e, train_losses, label="train")
                    plt.plot(e, valid_losses, label="validation")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.legend()

                    # Save figure and store as MLFlow artifact
                    if len(models) == 1:
                        plot.savefig("losses", path=args.outpath)
                    else:
                        plot.savefig(f"losses_{idx}", path=args.outpath)

                    plt.close()

        # Load best models
        best_models = []
        for idx in range(len(models)):
            if len(models) == 1:
                best_models.append(
                    utils.loadmodel(os.path.join(args.outpath, "best.pth"))
                )
            else:
                best_models.append(
                    utils.loadmodel(os.path.join(args.outpath, f"best_{idx}.pth"))
                )

        predict.evaluate(
            best_models,
            trainloader,
            AEVC,
            args.outpath,
            scaler=scaler,
            baseline=None,
            stage="train",
            plt=args.plot,
        )
        predict.evaluate(
            best_models,
            validloader,
            AEVC,
            args.outpath,
            scaler=scaler,
            baseline=None,
            stage="valid",
            plt=args.plot,
        )

        if args.testfile is not None:
            predict.evaluate(
                best_models,
                testloader,
                AEVC,
                args.outpath,
                scaler=scaler,
                baseline=None,
                stage="test",
                plt=args.plot,
            )

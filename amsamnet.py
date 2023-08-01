import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import argparse
import pandas as pd
from tqdm import tqdm
import os
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from model.amsam import SimpleAMSAMNet, NoAttentionAMSAMNet, TwoScalesAMSAMNet, SingleScaleAMSAMNet
from dataset.SyntheticHyperspectralDataset import SyntheticHyperspectralDataset
from dataset.hyperspectral_ds_lmdb import HyperspectralPatchLmdbDataset
from loss_functions.kon_losses import NTXentLoss

def save_best_model(epoch, model, train_loss, top_k_accuracy_train, val_loss, top_k_accuracy_val, model_path, model_dir):
    """
    Function to save the best model and metrics.
    """
    torch.save({'epoch': epoch,'model_state_dict': model.state_dict()}, model_path)

    metrics = {
        'epoch': epoch, 
        'train_loss': train_loss, 
        'top_k_accuracy_train': top_k_accuracy_train, 
        'val_loss': val_loss, 
        'top_k_accuracy_val': top_k_accuracy_val
    }
    
    with open(os.path.join(model_dir, "best_metrics.json"), "w+") as outfile: 
        json.dump(metrics, outfile, indent=4)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

def read_csv_keys(csv_file, csv_file_col_name):
    df = pd.read_csv(csv_file)
    keys = df[csv_file_col_name].tolist()
    return keys

def cosine_similarity(x, y):
    return torch.nn.functional.cosine_similarity(x, y)

def evaluate(afeatures, pfeatures, query_index, k=5):
    query = pfeatures[query_index].unsqueeze(0)
    distances = cosine_similarity(query, afeatures)
    topk_indices = torch.topk(distances, k, dim=-1).indices.squeeze()
    return topk_indices

def compute_top_k_accuracy(dataloader, fextractor, step_num, writer, topk_tag, k=1):
    afeatures, pfeatures = [], []
    with torch.no_grad():
        for anchor, positive in dataloader:
            afeatures.append(fextractor(anchor).detach().cpu())
            pfeatures.append(fextractor(positive).detach().cpu())

    afeatures = torch.cat(afeatures, dim=0)
    pfeatures = torch.cat(pfeatures, dim=0)

    total_correct_equal = 0
    for i in range(len(pfeatures)):
        topk_indices = evaluate(
            afeatures=afeatures, pfeatures=pfeatures, query_index=i, k=k
        )
        total_correct_equal += torch.sum(topk_indices == i).item()

    top_k_accuracy_equal = total_correct_equal / len(afeatures)
    
    writer.add_scalar(topk_tag, top_k_accuracy_equal, step_num)

    return top_k_accuracy_equal

def train(
    dataloader,
    model,
    criterion,
    optimizer,
    epoch,
    writer,
    k=1,
    add_tb_images=False,
    compute_top_k=True,
    dataset_obj=None,
):
    model.train()
    running_loss = 0.0
    top_k_accuracy_train = 0.0
    do_logging = True
    with tqdm(dataloader, unit="batch") as tepoch:
        for i, (anchor, positive) in enumerate(tepoch):
            tepoch.set_description(f"Training: Epoch {epoch + 1}")

            optimizer.zero_grad()

            a_output = model(anchor)
            p_output = model(positive)

            a_output = F.normalize(a_output)
            p_output = F.normalize(p_output)

            loss = criterion(a_output, p_output)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if ((epoch + 1) % config["val_frequency"] == 0) and do_logging:
                if add_tb_images:
                    pass
                    # tb_add_images(anchor, positive, epoch+1, writer, dataset_obj)

                if compute_top_k and ((epoch + 1) % config["val_frequency"] == 0):
                    top_k_accuracy_train = compute_top_k_accuracy(
                        dataloader=dataloader,
                        fextractor=model,
                        step_num=epoch+1,
                        writer=writer,
                        topk_tag="Top-k Accuracy/Train",
                        k=k
                    )

                do_logging = False

            tepoch.set_postfix(loss=loss.item())

    return running_loss / (i + 1), top_k_accuracy_train


def validate(dataloader, model, criterion, epoch, tbwriter, k=1):
    # model.eval()
    val_loss = 0.0
    top_k_accuracy_val = 0.0
    do_logging = True
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            for i, (vanchor, vpositive) in enumerate(tepoch):
                tepoch.set_description(f"Validation: Epoch {epoch + 1}")

                va_output = model(vanchor)
                vp_output = model(vpositive)

                # normalize => l2 norm
                va_output = F.normalize(va_output)
                vp_output = F.normalize(vp_output)

                vloss = criterion(va_output, vp_output)

                # Print statistics
                val_loss += vloss.item()

                if ((epoch + 1) % config["val_frequency"] == 0) and do_logging:
                    top_k_accuracy_val = compute_top_k_accuracy(
                        dataloader=dataloader,
                        fextractor=model,
                        step_num=epoch+1,
                        writer=tbwriter,
                        topk_tag="Top-k Accuracy/Validation",
                        k=k
                    )

                    do_logging = False

                tepoch.set_postfix(loss=val_loss)

    return val_loss / (i + 1), top_k_accuracy_val

if __name__ == "__main__":
    # Parse the arguments
    if 1:
        config_path = r'/vol/research/RobotFarming/Projects/amsamnet/config/config_test.json'
    else:
        config_path = None
    parser = argparse.ArgumentParser(description='AMSAMNet Training')
    parser.add_argument('-c', '--config', default=config_path,type=str,
                            help='Path to the config file')
    args = parser.parse_args()
    
    config = json.load(open(args.config))

    tag = config["tag"]
    log_dir = config["log_dir"]
    experiment_dir = config["experiment_dir"]
    experiment_name = config["experiment_name"]
    lmdb_save_dir = config["lmdb_save_dir"]
    l1_mean_file = config["l1_mean_file"]
    l1_std_file = config["l1_std_file"]
    l2_mean_file = config["l2_mean_file"]
    l2_std_file = config["l2_std_file"]
    lmdb_file_name = config["lmdb_file_name"]
    columns = config["columns"]
    csv_file_name = config["csv_file_name"]
    in_channels = config["in_channels"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    val_batch_size = config["val_batch_size"]
    val_split = config["val_split"]
    out_features = config["out_features"]
    normalize = config["normalize"]
    k = config["k"]
    learning_rate = config["learning_rate"] 
       
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device.type}')
    # is_gpu = device.type != "cpu"

    keys = read_csv_keys(os.path.join(lmdb_save_dir, csv_file_name), columns[0])

    train_keys, val_keys = train_test_split(keys, test_size=val_split, random_state=42, shuffle=False)

    # Report split sizes
    print("Training set has {} instances".format(len(train_keys)))
    print("Validation set has {} instances".format(len(val_keys)))

    # Instantiate the dataset and the dataloaders
    train_dataset = HyperspectralPatchLmdbDataset(
        train_keys,
        lmdb_save_dir,
        lmdb_file_name,
        in_channels,
        device,
        normalize=normalize,
        mean_file=l1_mean_file,
        std_file=l1_std_file,
    )

    val_dataset = HyperspectralPatchLmdbDataset(
        val_keys,
        lmdb_save_dir,
        lmdb_file_name,
        in_channels,
        device,
        normalize=normalize,
        mean_file=l1_mean_file,
        std_file=l1_std_file,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=val_batch_size, 
        shuffle=False, 
        num_workers=0, 
        drop_last=True
    ) 
    
    writer_name_tag = f'{experiment_name}_C{in_channels}_{csv_file_name.split(".")[0]}_b{batch_size}_e{num_epochs}_OF{out_features}_{tag}'
    writer = SummaryWriter(os.path.join(log_dir, writer_name_tag))

    # Create an instance of the network
    # SimpleAMSAMNet, NoAttentionAMSAMNet, TwoScalesAMSAMNet, SingleScaleAMSAMNet
    model = NoAttentionAMSAMNet(in_channels=in_channels).to(device)

    # Loss and optimizer
    criterion = NTXentLoss() #nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f'{writer_name_tag}: {csv_file_name.split(".")[0]} Training started successfully...')
    best_vloss = 1_000_000.0
    val_loss = 1_000_001.0

    model_name = f'{experiment_name}_C{in_channels}_{csv_file_name.split(".")[0]}_b{batch_size}_e{num_epochs}_OF{out_features}_{tag}'
    model_dir = os.path.join(experiment_dir, model_name)
    model_path = os.path.join(model_dir, "best_model.pth")
    ensure_dir(model_path)
    start_epoch = 0
    

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded model checkpoint from {model_path}")
        print(f'starting from epoch {start_epoch}')

    for epoch in range(start_epoch, num_epochs):

        train_loss, top_k_accuracy_train = train(
            train_dataloader,
            model,
            criterion,
            optimizer,
            epoch,
            writer,
            k=k,
            add_tb_images=True,
            compute_top_k=True,
            dataset_obj=train_dataset,
        )
        
        # print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        
        if (epoch + 1) % config["val_frequency"] == 0:
            val_loss, top_k_accuracy_val = validate(
                dataloader=val_dataloader, 
                model=model, 
                criterion=criterion, 
                epoch=epoch, 
                tbwriter=writer, 
                k=k
            )

            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

        # if (epoch + 1) % config["val_frequency"] == 0:
            print(f" Top-k Accuracy (Train): {top_k_accuracy_train:.4f}, Top-k Accuracy (Val): {top_k_accuracy_val:.4f}")
            writer.add_scalar("Loss/Train", train_loss, epoch + 1)
            writer.add_scalar("Loss/Validation", val_loss, epoch + 1)

            # Save the model
            if val_loss < best_vloss:
                best_vloss = val_loss
                # Start a new process to save the best model and metrics
                save_best_model(epoch + 1, model, train_loss, top_k_accuracy_train, val_loss, top_k_accuracy_val, model_path, model_dir)

    # Close Tensorboard writer
    writer.close()

    print(
        f'{writer_name_tag}: {csv_file_name.split(".")[0]} Training completed successfully...'
    )
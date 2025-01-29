import torch
import copy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def save_histogram_data(bin_errors_dict, bin_ranges, filename):
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(bin_errors_dict)
    
    # Add the bin ranges as a new column
    df.insert(0, 'Bin Range', bin_ranges)
    
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)

import matplotlib.pyplot as plt
import numpy as np

def display_histogram(histogram_data_list, bin_ranges, labels, title='Prediction error for each range of target ghost lengths', save_path=None, y_max=None):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Calculate the width of each bar
    num_histograms = len(histogram_data_list)
    total_width = 0.8
    bar_width = total_width / num_histograms
    indices = np.arange(len(bin_ranges))

    # Loop over the list of histogram data
    for i, histogram_data in enumerate(histogram_data_list):
        ax.bar(indices + i*bar_width, histogram_data, bar_width, label=labels[i])
    if y_max is not None:
        ax.set_ylim(0, y_max)
    # Set y-axis to be logarithmic
    # ax.set_yscale('log')

    # Add labels and title
    ax.set_xlabel('Bin Range')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title(title)

    # Add legend
    ax.legend()

    # Add x-tick labels
    ax.set_xticks(indices + total_width / 2)
    ax.set_xticklabels(bin_ranges, rotation=45)

    # Save the plot if a path is provided, otherwise show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    # Close the figure to free memory
    plt.close(fig)


def calculate_histogram(dataloader, model, n_bins, device, max_value, list_rescalings=None, error_type="MAE", save_results_path=None):
    bin_errors = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    bin_ranges = []
    model.eval()
    targets_all = []
    for data in dataloader:
        data = data.to(device)  
        x, edge_index, edge_attr, parenthood, batch, y = extract_tensors(data)
        targets_all.extend(y.cpu().detach().numpy().tolist())
    targets_all = np.array(targets_all)
    bins = np.linspace(0, max_value, n_bins + 1)  # bins start at 0
    for bin_start, bin_end in zip(bins[:-1], bins[1:]):
        bin_ranges.append(f"{bin_start:.2f} - {bin_end:.2f}")
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device) 
            x, edge_index, edge_attr, parenthood, batch, y = extract_tensors(data)
            out = model(x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        parenthood=parenthood,
                        batch=batch).squeeze(1)
            predictions = out.cpu().detach().numpy()
            targets = y.cpu().detach().numpy()
            # save predictions and targets
            if save_results_path:
                predictions_df = pd.DataFrame({
                    'predictions': predictions,
                    'targets': targets
                })
                predictions_df.to_csv(save_results_path, index=False)
            if error_type == "MSE":
                errors = (predictions - targets) ** 2
            if error_type == "MAE":
                errors = abs(predictions-targets)
            # We need to rescale the errors, so that we get the real errors and not rescaled errors
            if list_rescalings is not None:
                tensor_rescaling = torch.tensor(list_rescalings)
            else:
                tensor_rescaling = torch.ones(len(errors))
            batch = batch.cpu()
            tensor_rescaling = tensor_rescaling[batch]
            errors = torch.tensor(errors)
            errors = torch.einsum("a, a -> a", errors, tensor_rescaling)
            if error_type == "MSE":
                errors = torch.einsum("a, a -> a", errors, tensor_rescaling)
            bin_indices = np.digitize(targets, bins) - 1
            for bin_idx in range(n_bins):
                bin_errors[bin_idx] += errors[bin_indices == bin_idx].sum()
                bin_counts[bin_idx] += (bin_indices == bin_idx).sum()
        bin_errors /= np.maximum(bin_counts, 1)  # calculate mean squared error, avoid divide by 0
    return bin_errors, bin_ranges

def extract_tensors(pyg_tree):
    """convenient way to get the relevant information"""
    x = pyg_tree["node"].x
    edge_index = pyg_tree["node", "sends_gene_to", "node"].edge_index
    edge_attr = pyg_tree["node", "sends_gene_to", "node"].edge_attr
    parenthood = pyg_tree["node", "is_parent_of", "node"].edge_index
    y = pyg_tree["node"].y
    if hasattr(pyg_tree["node"], "batch"):
        batch = pyg_tree["node"].batch
    else:
        batch = None
    return x, edge_index, edge_attr, parenthood, batch, y

def training_loop(model, n_epochs, train_dataloader, test_dataloader, optimizer, early_stopping=False,  patience=10, n_bins=10, lr_scheduler=None, loss_type="MSE"):
    loss_fn = torch.nn.MSELoss() if loss_type == "MSE" else torch.nn.L1Loss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    pbar = tqdm(range(n_epochs))
    test_losses = []
    train_losses = []
    
    # Initialize variables for Early Stopping
    best_loss = float('inf')
    best_epoch = 0
    counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    model.train()
    for epoch in pbar:
        for data in train_dataloader:

            data = data.to(device)  # move data to device

            optimizer.zero_grad()

            x, edge_index, edge_attr, parenthood, batch, y = extract_tensors(data)

            out = model(x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        parenthood=parenthood,
                        batch=batch).squeeze(1)
            loss = loss_fn(out, y)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_description(f"Epoch: {epoch}, Loss: {loss.item()}")
        if lr_scheduler:
            lr_scheduler.step()
        if epoch%100 == 0:
            model.eval()
            with torch.no_grad():
                for data in test_dataloader:  # added loop over test data
                    data = data.to(device) # transfer data to device
                    x, edge_index, edge_attr, parenthood, batch, y = extract_tensors(data)
                    out = model(x=x,
                                edge_index=edge_index,
                                edge_attr=edge_attr,
                                parenthood=parenthood,
                                batch=batch).squeeze(1)
                    loss = loss_fn(out, y)
                    loss_value = loss.cpu().detach().numpy()
                    print(f"test loss {loss_value}")
                    test_losses.append(float(loss_value))

                    # Check for Early Stopping
                    if loss_value < best_loss and early_stopping:
                        best_loss = loss_value
                        best_epoch = epoch
                        counter = 0
                        # Save the model weights
                        best_model_wts = copy.deepcopy(model.state_dict())
                    elif early_stopping:
                        counter += 1
                        if counter >= patience:
                            print(f'Early stopping at epoch {epoch}, best loss was {best_loss} at epoch {best_epoch}')
                            model.load_state_dict(best_model_wts)
                            pbar.close()
                            return train_losses, test_losses
            model.train()
    if early_stopping:
        model.load_state_dict(best_model_wts)
    pbar.close()
    return train_losses, test_losses
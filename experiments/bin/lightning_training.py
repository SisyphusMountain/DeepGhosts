from bin.linear import LinearModel
from bin.mlp import MLP
from bin.transformer_vanilla import VanillaTransformer
from bin.transformer_gcn import TransformerGCN
from bin.transformer_parenthood import TransformerParenthood
from lightning.fabric import Fabric
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.utils import to_undirected
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wandb.integration.lightning.fabric import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
import copy
import pickle
from typing import Tuple, List
from collections import defaultdict
from tqdm import tqdm
import seaborn as sns
from bin.linear import LinearModel
from bin.mlp import MLP

from scipy.ndimage import gaussian_filter1d
import wandb
import heapq

# TODO:
# 4. Setup comparison of model performance, for example superimposing plots in wandb Not possible, need to create wandb reports to compare
# 5. Binning plot throughout training. Done (box plots)
# 6. Add learning rate scheduler Added, not tested yet
# 7. Try to train the models with MAE instead of L2 loss to see the results
# 8. Perform ablation studies, on the presence of batch normalization and other regularization techniques
# 9. Test also weight decay Implemented, need to test more thouroughly
# 10. Perform hyperparameter optimization
# 11. Put README files everywhere to explain what everything is doing in the GitHub repository
# 12. Add everything on github to make sure it's not lost. Done for now, log regularly from now on.
# 13. Change the wandb local logging folder (now it's "outputs") Done
# 14. Check the normalization and code for coded neural networks
# 15. Run the training code on the more complex neural networks
# What else to log?
torch.set_float32_matmul_precision("high")


@hydra.main(version_base="1.3", config_path="configs")
def main(cfg: DictConfig) -> None:
    global max_epochs
    max_epochs = cfg.training.epochs
    # Initialize W&B logger
    logger = WandbLogger(
        name=cfg.wandb.run_name,
        save_dir=cfg.wandb.save_dir,
        log_model=cfg.wandb.log_model,
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Load dataset
    with open(cfg.data.path, "rb") as f:
        full_dataset = pickle.load(f)

    # Prepare datasets
    train_data, test_data, val_data = prepare_datasets(full_dataset, cfg.data.split_ratio)

    # Normalize datasets if configured
    if cfg.data.normalize:
        node_mean, node_std, edge_mean, edge_std = compute_normalization_stats(train_data)
        node_stats = (node_mean, node_std)
        edge_stats = (edge_mean, edge_std)
        train_data = normalize_dataset(train_data, node_stats, edge_stats)
        test_data = normalize_dataset(test_data, node_stats, edge_stats)
        val_data = normalize_dataset(val_data, node_stats, edge_stats)
    
    # Instantiate callbacks
    callbacks = []
    early_stop_cb = None
    if cfg.training.early_stopping:
        early_stop_cb = EarlyStoppingCallback(
            patience=cfg.training.patience, 
            min_delta=0.0, 
            verbose=True,
            best_model_path=os.path.join(cfg.wandb.save_dir, "best_model.pt"),
        )
        callbacks.append(early_stop_cb)
    top_k_checkpoint_cb = TopKModelCheckpoint(
        save_dir=cfg.wandb.save_dir,  # Save top-K models in W&B folder
        top_k=5  # Keep only top 5 models
    )
    callbacks.append(top_k_checkpoint_cb)
    # Create LightningDataset (for PyG)
    datamodule = LightningDataset(
        train_dataset=train_data,
        test_dataset=test_data,
        val_dataset=val_data,
        batch_size=cfg.training.batch_size
    )
    # 1) Instantiate your new prediction logger callback
    pred_logger_cb = PredictionLoggerCallback(
        output_dir=cfg.wandb.save_dir,  # Save CSVs to the same directory as W&B logs
        log_to_wandb=True,
        wandb_logger=logger,
        plot_every_n_validation_epoch=cfg.logging.log_plot_every_n_validation_epoch,  # Generate plot every epoch. Adjust as needed.
        window_size=cfg.logging.log_window_size,  # Window size for moving quantiles
        step=cfg.logging.log_step,  # Step size for moving quantiles
        sigma=cfg.logging.log_sigma,  # Sigma for Gaussian filter
        num_bins=cfg.logging.log_num_bins,  # Number of bins for binned plots
        violin_width=cfg.logging.log_violin_width,  # Width of violin plots
        cfg=cfg,
        test_loader=None,
    )
    callbacks.append(pred_logger_cb)

    # Initialize Fabric with your callbacks
    fabric = Fabric(loggers=[logger], callbacks=callbacks)
    fabric.seed_everything(cfg.seed)



    fabric.launch()
    # Compute the mean of the target values from the training dataset
    train_targets = []
    for sample in train_data:
        train_targets.append(sample["node"].y)
    train_targets = torch.cat(train_targets)
    target_mean = train_targets.mean().item()
    # Initialize model
    if cfg.model == "linear":
        model = LinearModel(
            in_features=cfg.data.node.in_features,
            out_features=cfg.data.node.out_features
        )
    elif cfg.model == "mlp":
        model = MLP(in_features=cfg.data.node.in_features,
                    out_features=cfg.data.node.out_features,
                    hidden_layers=cfg.mlp.hidden_layers,
                    dropout=cfg.mlp.dropout)
    elif cfg.model == "vanilla_transformer":
        model = VanillaTransformer(node_in_features=cfg.data.node.in_features,
                                   d_model=cfg.transformer.d_model,
                                   n_heads=cfg.transformer.n_heads,
                                   mlp_expansion_factor=cfg.transformer.mlp_expansion_factor,
                                   n_blocks=cfg.transformer.n_blocks)
    elif cfg.model == "transformer_gcn":
        model = TransformerGCN(node_in_features=cfg.data.node.in_features,
                d_model=cfg.transformer.d_model,
                n_heads=cfg.transformer.n_heads,
                mlp_expansion_factor=cfg.transformer.mlp_expansion_factor,
                n_blocks=cfg.transformer.n_blocks,
                aggr=cfg.transformer.aggr,)
    elif cfg.model == "transformer_parenthood":
        model = TransformerParenthood(node_in_features=cfg.data.node.in_features,
                                      d_model=cfg.transformer.d_model,
                                      n_heads=cfg.transformer.n_heads,
                                      mlp_expansion_factor=cfg.transformer.mlp_expansion_factor,
                                      n_blocks=cfg.transformer.n_blocks,
                                      dropout=0.0)

    # Set the bias of the last linear layer to the target mean
    if cfg.model in ["vanilla_transformer", "transformer_gcn"]:
        # For transformer models, directly access the unembedding's linear layer
        if hasattr(model.unembedding, "linear") and model.unembedding.linear.bias is not None:
            model.unembedding.linear.bias.data.fill_(target_mean)
        else:
            fabric.print("Warning: Unembedding layer's linear layer has no bias term.")
    params_decay = []
    params_no_decay = []

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for name, param in module.named_parameters(recurse=False):
                if "bias" in name:
                    params_no_decay.append(param)
                else:
                    params_decay.append(param)
        elif isinstance(module, nn.LayerNorm):
            params_no_decay.extend(module.parameters(recurse=False))
        else:
            # Collect parameters from other modules (not Linear or LayerNorm)
            other_params = list(module.parameters(recurse=False))
            params_no_decay.extend(other_params)

    # Remove duplicates by converting to sets and back to lists
    params_decay = list(set(params_decay))
    params_no_decay = list(set(params_no_decay))
    # Ensure all parameters are accounted for
    all_model_params = set(model.parameters())
    combined_params = set(params_decay + params_no_decay)
    assert all_model_params == combined_params, "Some parameters are missing in the groups"

    # Create parameter groups for the optimizer
    param_groups = [
        {'params': params_decay, 'weight_decay': cfg.training.weight_decay},
        {'params': params_no_decay, 'weight_decay': 0.0}
    ]

    # Initialize optimizer with parameter groups
    optimizer = optim.Adam(
        param_groups,
        lr=cfg.training.lr,
        betas=(cfg.training.beta_1, cfg.training.beta_2),
        eps=cfg.training.eps,
    )
    # Initialize learning rate scheduler (if configured)
    if cfg.training.scheduler == "linear":
        warmup_steps = cfg.training.warmup_epochs * len(datamodule.train_dataloader())
        total_steps = cfg.training.epochs * len(datamodule.train_dataloader())
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / warmup_steps  # Linear warmup
            else:
                return max((total_steps - current_step) / (total_steps - warmup_steps), 0.0)  # Linear decay
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # Setup model + optimizer for distributed
    model, optimizer = fabric.setup(model, optimizer)

    # Setup dataloaders for distributed
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader)
    pred_logger_cb.test_loader = test_loader
    # Train
    train_losses, val_losses = train(
        fabric, model, optimizer, train_loader, val_loader, cfg, 
        scheduler=scheduler, 
        early_stop_cb=early_stop_cb
    )


def register_activation_hooks(model: nn.Module):
    """
    Registers forward hooks on every module that has trainable parameters,
    keyed by the module's 'name' (e.g., 'layer1.0.conv', etc.).
    
    Returns a dictionary 'activation_stats' that will store a list of RMS 
    activation values for each named module, collected after every forward pass.
    """
    from collections import defaultdict
    
    activation_stats = defaultdict(list)

    def make_hook(module_name):
        """
        Create a closure that knows the current module's name,
        so we avoid Python's late-binding issues in loops.
        """
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                # Root-mean-square over the entire output, including batch dimension
                rms_val = output.detach().pow(2).mean().sqrt().item()
                activation_stats[module_name].append(rms_val)
        return hook_fn

    # Loop over named_modules, just like your parameter-norm code:
    for name, module in model.named_modules():
        # Hook only modules that have trainable parameters
        if any(p.requires_grad for p in module.parameters()):
            # Initialize list in our dictionary
            activation_stats[name] = []
            # Register the forward hook
            module.register_forward_hook(make_hook(name))

    return activation_stats

def compute_normalization_stats(train_dataset: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute normalization statistics (mean and std) for node features and edge attributes."""
    node_x_mean = torch.zeros(train_dataset[0]["node"].x.shape[1])
    node_x_std = torch.zeros_like(node_x_mean)
    edge_attr_mean = torch.zeros(train_dataset[0]["node", "sends_gene_to", "node"].edge_attr.shape[1])
    edge_attr_std = torch.zeros_like(edge_attr_mean)

    # Compute means
    for sample in train_dataset:
        node_x_mean += sample["node"].x.mean(dim=0)
        edge_attr_mean += sample["node", "sends_gene_to", "node"].edge_attr.mean(dim=0)
    node_x_mean /= len(train_dataset)
    edge_attr_mean /= len(train_dataset)

    # Compute standard deviations
    for sample in train_dataset:
        node_x_std += ((sample["node"].x - node_x_mean) ** 2).mean(dim=0)
        edge_attr_std += ((sample["node", "sends_gene_to", "node"].edge_attr - edge_attr_mean) ** 2).mean(dim=0)
    node_x_std = torch.sqrt(node_x_std / len(train_dataset))
    edge_attr_std = torch.sqrt(edge_attr_std / len(train_dataset))

    # Add epsilon to avoid division by zero
    epsilon = 1e-8
    return node_x_mean, node_x_std + epsilon, edge_attr_mean, edge_attr_std + epsilon


def normalize_dataset(dataset: list, node_stats: Tuple[torch.Tensor, torch.Tensor], edge_stats: Tuple[torch.Tensor, torch.Tensor]) -> list:
    """Normalize dataset using precomputed statistics."""
    node_mean, node_std = node_stats
    _, edge_std = edge_stats # Don't remove edge attr mean from the edge attributes because negative edge weights are not supported by gcnconv 
    normalized = []
    for sample in dataset:
        new_sample = sample.clone()
        new_sample["node"].x = (new_sample["node"].x - node_mean) / node_std
        new_sample["node", "sends_gene_to", "node"].edge_attr = (
            (new_sample["node", "sends_gene_to", "node"].edge_attr) / edge_std
        )
        normalized.append(new_sample)
    return normalized


def compute_gradient_norms(model: nn.Module) -> dict:
    """
    Compute the L2 norm of gradients for each layer in the model.
    """
    grad_norms = {}
    for name, module in model.named_modules():
        if any(p.requires_grad for p in module.parameters()):  # Only consider layers with trainable parameters
            total_norm = torch.tensor(0.0)
            for param in module.parameters():
                if param.grad is not None:
                    param_norm = param.grad.detach().norm(2).cpu()  # L2 norm of gradients
                    total_norm += param_norm ** 2
            if total_norm > 0:
                total_norm = total_norm ** 0.5  # Square root to get L2 norm
                grad_norms[name] = total_norm
    return grad_norms


def extract_tensors(batch):
    return (
        batch["node"].x,
        batch["node", "sends_gene_to", "node"].edge_index,
        batch["node", "sends_gene_to", "node"].edge_attr,
        batch["node", "is_parent_of", "node"].edge_index,  # Assuming parenthood is in this edge type
        batch["node"].batch,
        batch["node"].y
    )


def train(
    fabric: Fabric,
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader,
    val_dataloader,
    cfg: DictConfig,
    scheduler: optim.lr_scheduler._LRScheduler = None,
    early_stop_cb = None,
) -> Tuple[List[float], List[float]]:
    """Example training loop that uses Fabric callbacks for early stopping."""
    loss_fn = nn.MSELoss() if cfg.training.loss_type == "MSE" else nn.L1Loss()

    train_losses = []
    val_losses = []
    if cfg.training.log_activations:
        activation_stats = register_activation_hooks(model)
    for epoch in tqdm(range(cfg.training.epochs), desc="Training"):
        if cfg.training.log_activations:
            for k in activation_stats.keys():
                activation_stats[k].clear()
        model.train()
        epoch_loss = 0.0
        grad_norms = defaultdict(float)
        counts = defaultdict(int)

        # ------------------
        # 1) TRAINING
        # ------------------
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            x, edge_index, edge_attr, parenthood, batch_idx_t, y = extract_tensors(batch)
            out = model(x, edge_index, edge_attr, parenthood, batch_idx_t).squeeze(-1)
            loss = loss_fn(out, y)

            fabric.backward(loss)

            # Track gradient norms
            if epoch % cfg.logging.measure_parameters_every_n_epochs == 0:
                current_grad_norms = compute_gradient_norms(model.module if hasattr(model, "module") else model)
                for name, norm in current_grad_norms.items():
                    grad_norms[name] += norm
                    counts[name] += 1

            optimizer.step()
            epoch_loss += loss.detach().cpu()


        # Update scheduler if any
        if scheduler:
            scheduler.step()
        # Log learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        fabric.log("train/lr", current_lr, step=epoch)
        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        fabric.log("train/loss", avg_train_loss, step=epoch)

        # Log gradient norms and parameter norms
        if epoch % cfg.logging.measure_parameters_every_n_epochs == 0:
            if cfg.training.log_activations:
                for module_name, rms_values in activation_stats.items():
                    if rms_values:  # non-empty
                        avg_rms = sum(rms_values) / len(rms_values)
                        fabric.log(f"activation_rms/{module_name}", avg_rms, step=epoch)
            
            for name in grad_norms:
                avg_grad_norm = grad_norms[name] / counts[name]
                fabric.log(f"grad_norm/{name}", avg_grad_norm, step=epoch)
            param_norms = compute_parameter_norms(model.module if hasattr(model, "module") else model)
            for name, norm in param_norms.items():
                fabric.log(f"param_norm/{name}", norm, step=epoch)
        # ------------------
        # 2) VALIDATION
        # ------------------
        if epoch % cfg.training.validate_every_n_epoch == cfg.training.validate_every_n_epoch - 1:
            val_loss = validate(fabric, model, val_dataloader, loss_fn)
            val_losses.append(val_loss)
            tqdm.write(f"Epoch {epoch+1}/{cfg.training.epochs} | "
                       f"Train Loss: {avg_train_loss:.4f} | "
                       f"Val Loss: {val_loss:.4f}")
            fabric.log("val/loss", val_loss, step=epoch)

            # IMPORTANT: pass val_loader so the callback can collect predictions
            fabric.call(
                "on_validation_epoch_end", 
                fabric=fabric, 
                model=model,
                val_loss=val_loss.detach().cpu(), 
                epoch=epoch, 
                val_loader=val_dataloader
            )

            if early_stop_cb and early_stop_cb.should_stop:
                fabric.print("Early stopping triggered, ending training.")
                fabric.log
                break

    # Restore best weights if early stopping was used
    if early_stop_cb:
        early_stop_cb.restore_best_weights(fabric, model)
    fabric.call("on_train_end", fabric, model)
    return train_losses, val_losses


def validate(fabric: Fabric, model: nn.Module, dataloader, loss_fn) -> float:
    """Validation loop."""
    model.eval()
    with torch.no_grad():
        total_loss = torch.tensor(0.0)
        for batch in dataloader:
            x, edge_index, edge_attr, parenthood, batch_idx, y = extract_tensors(batch)
            out = model(x, edge_index, edge_attr, parenthood, batch_idx).squeeze(-1)
            loss = loss_fn(out, y)
            total_loss += loss.cpu().detach()
    return total_loss / len(dataloader)


def test(fabric: Fabric, model: nn.Module, dataloader) -> float:
    """Test loop."""
    loss_fn = nn.MSELoss()  # Or get from config
    model.eval()
    total_loss = torch.tensor(0.0)
    with torch.no_grad():
        for batch in dataloader:
            x, edge_index, edge_attr, parenthood, batch_idx, y = extract_tensors(batch)
            out = model(x, edge_index, edge_attr, parenthood, batch_idx).squeeze(-1)
            loss = loss_fn(out, y)
            total_loss += loss.cpu().detach()
    avg_loss = total_loss / len(dataloader)
    fabric.print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss


def prepare_datasets(full_dataset, split_ratio):
    """Split dataset into train, test, and validation sets."""
    # Preprocess dataset (reverse edges and make undirected)
    for graph in full_dataset:
        graph["node", "sends_gene_to", "node"].edge_index = graph["node", "sends_gene_to", "node"].edge_index[[1, 0]].contiguous()
        graph["node", "is_parent_of", "node"].edge_index = to_undirected(
            graph["node", "is_parent_of", "node"].edge_index
        ).contiguous()

    n_total = len(full_dataset)
    train_end = int(n_total * split_ratio)
    test_end = train_end + int(n_total * (1 - split_ratio) / 2)
    train_data = full_dataset[:train_end]
    test_data = full_dataset[train_end:test_end]
    val_data = full_dataset[test_end:]
    return train_data, test_data, val_data


def get_unwrapped_state_dict(model: torch.nn.Module):
    """Helper to get the state_dict from an unwrapped model if Fabric wraps it."""
    return model.module.state_dict() if hasattr(model, 'module') else model.state_dict()


class EarlyStoppingCallback:
    """
    Stops training if validation loss does not improve after a given patience.
    Also keeps track of (and saves) the best model checkpoint to disk.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        verbose: bool = False,
        best_model_path: str = "./best_model.pt"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0

        self.best_model_wts = None
        self.best_epoch = 0
        self.should_stop = False
        self.best_model_path = best_model_path

    def on_validation_epoch_end(self, fabric, model, val_loss: float, epoch: int, **kwargs):
        """
        Called at the end of each validation epoch. Checks if `val_loss` is better
        than the previous best. If so, saves the new best model to disk.
        """
        if (self.best_loss - val_loss) > self.min_delta:
            # We have a new best model
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.best_model_wts = copy.deepcopy(get_unwrapped_state_dict(model))
            
            # Only save on the main process (rank 0) if distributed
            if fabric.global_rank == 0:
                torch.save(self.best_model_wts, self.best_model_path)
                fabric.print(f"[EarlyStopping] New best model saved at {self.best_model_path} "
                             f"(epoch={epoch+1}, val_loss={val_loss:.4f})")

        else:
            self.counter += 1
            if self.verbose:
                fabric.print(f"[EarlyStopping] No improvement. "
                             f"Patience {self.counter}/{self.patience}")

            # Check if we exceeded patience
            if self.counter >= self.patience:
                if self.verbose:
                    fabric.print(f"[EarlyStopping] Triggered at epoch {epoch+1}.")
                self.should_stop = True

    def restore_best_weights(self, fabric, model):
        """Call after training loop ends to restore the best model weights in memory."""
        if self.best_model_wts is not None:
            current_state = model.state_dict()
            current_state.update(self.best_model_wts)
            model.load_state_dict(current_state)
            fabric.print(
                f"[EarlyStopping] Restored best model from epoch {self.best_epoch+1} "
                f"with val_loss={self.best_loss:.4f}"
            )


class PredictionLoggerCallback:
    def __init__(
        self,
        output_dir: str = "./results",
        log_to_wandb: bool = True,
        wandb_logger=None,
        plot_every_n_validation_epoch: int = 1,
        window_size: int = 50,
        step: int = 10,
        sigma: float = 2,
        num_bins: int = 10,
        violin_width = 1.5,
        cfg=None,
        test_loader=None,
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_to_wandb = log_to_wandb
        self.logger = wandb_logger
        self.plot_every_n_validation_epoch = plot_every_n_validation_epoch
        self.current_validation_epoch = 0
        self.window_size = window_size
        self.step = step
        self.sigma = sigma
        self.num_bins = num_bins
        self.violin_width = violin_width
        self.cfg = cfg
        self.test_loader = test_loader
    def on_train_end(self, fabric, model, **kwargs):
        """Called at the end of training to test the best model on the test set."""
        if self.test_loader is None:
            fabric.print("Warning: No test loader provided. Skipping final test.")
            return

        # Ensure the model is in evaluation mode
        model.eval()
        
        # Compute test loss
        test_loss = test(fabric, model, self.test_loader)
        fabric.print(f"\nFinal Test Loss: {test_loss:.4f}")
        fabric.log("test/loss", test_loss)

        # Collect test predictions and log plots
        fabric.print("[PredictionLogger] Logging test set predictions...")
        predictions, targets = self._collect_predictions(fabric, model, self.test_loader)
        df_test = pd.DataFrame({"targets": targets, "predictions": predictions})
        df_test.to_csv(os.path.join(self.output_dir, "test_predictions.csv"), index=False)
        
        if self.log_to_wandb:
            self.logger.log_table(key="test/predictions", dataframe=df_test)
        
        self._log_all_plots(df_test, epoch=0, stage="test")  # Log test plots

    def on_validation_epoch_end(self, fabric, model, val_loss, epoch, val_loader=None, **kwargs):
        if self.current_validation_epoch % self.plot_every_n_validation_epoch != self.plot_every_n_validation_epoch - 1 and epoch != max_epochs - 1:
            self.current_validation_epoch += 1
            return
        
        if val_loader is None:
            self.current_validation_epoch += 1
            return
        self.current_validation_epoch += 1
        fabric.print(f"[PredictionLogger] Logging prediction plots for epoch {epoch}...")
        predictions, targets = self._collect_predictions(fabric, model, val_loader)
        df = pd.DataFrame({"targets": targets, "predictions": predictions})
        df.to_csv(os.path.join(self.output_dir, f"val_predictions_epoch_{epoch}.csv"), index=False)
        
        if self.log_to_wandb:
            self.logger.log_table(key=f"val/predictions", dataframe=df)
        
        self._log_all_plots(df, epoch)

    def _log_all_plots(self, df: pd.DataFrame, epoch: int, stage="val"):
        y_true, y_pred = df["targets"].values, df["predictions"].values
        absolute_errors = np.abs(y_pred - y_true)

        # Moving window spread function
        def moving_window_spread(x, y, window_size=50, step=10, sigma=2):
            sorted_indices = np.argsort(x)
            x_sorted, y_sorted = x[sorted_indices], y[sorted_indices]
            means, stds, x_centers = [], [], []
            for i in range(0, len(x_sorted) - window_size + 1, step):
                x_window = x_sorted[i : i + window_size]
                y_window = y_sorted[i : i + window_size]
                means.append(np.mean(y_window))
                stds.append(np.std(y_window))
                x_centers.append(np.mean(x_window))
            return (
                np.array(x_centers),
                gaussian_filter1d(means, sigma=sigma),
                gaussian_filter1d(stds, sigma=sigma),
            )

        x_centers, mean_preds, std_preds = moving_window_spread(y_true, y_pred, self.window_size, self.step, self.sigma)

        # Smoothed Predictions Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, alpha=0.2, s=10, label="Predictions")
        ax.plot(x_centers, mean_preds, color="red", label="Mean Prediction", lw=2)
        ax.fill_between(x_centers, mean_preds - std_preds, mean_preds + std_preds, 
                        color="red", alpha=0.3, label="±1 Std Dev")
        ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2, label="y = x")
        #ax.set_xscale("symlog")
        #ax.set_yscale("symlog")
        ax.set_ylim(min(y_true), max(y_true))
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"Smoothed Predictions with Error Bands ({stage.capitalize()} Set)")
        ax.legend()
        pred_path = os.path.join(self.output_dir, f"{stage}_predictions_epoch_{epoch}.png")
        fig.savefig(pred_path)
        plt.close(fig)

        # Absolute Error Trend Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, absolute_errors, alpha=0.2, s=10, label="Absolute Error")
        ax.set_xlabel("True Values")
        ax.set_ylabel("|Predicted - True| (Absolute Error)")
        ax.set_title(f"Absolute Error Trend vs. True Values ({stage.capitalize()} Set)")
        ax.set_ylim(0, self.cfg.logging.y_lim)
        error_path = os.path.join(self.output_dir, f"{stage}_errors_epoch_{epoch}.png")
        fig.savefig(error_path)
        plt.close(fig)

        # Boxplot of Prediction Errors
        df["error"] = df["predictions"] - df["targets"]
        bin_edges = np.linspace(y_true.min(), y_true.max(), self.num_bins + 1)
        df["binned_targets"] = pd.cut(df["targets"], bins=bin_edges, include_lowest=True)

        fig, ax = plt.subplots(figsize=(6, 6))
        sns.boxplot(x="binned_targets", y="error", data=df, ax=ax, showfliers=False)
        ax.axhline(0, color="red", linestyle="--", lw=2)
        ax.set_xlabel("Target Value Bins")
        ax.set_ylabel("Prediction Error (Pred - True)")
        ax.tick_params(axis='x', rotation=45)
        ax.set_title(f"Binned Boxplot of Prediction Errors ({stage.capitalize()} Set)")
        ax.set_ylim(-self.cfg.logging.y_lim, self.cfg.logging.y_lim)
        boxplot_path = os.path.join(self.output_dir, f"{stage}_boxplot_epoch_{epoch}.png")
        fig.savefig(boxplot_path)
        plt.close(fig)

        # Violin Plot of Prediction Errors
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.violinplot(x="binned_targets", y="error", data=df, ax=ax, inner="quartile", width=1.5)
        ax.axhline(0, color="red", linestyle="--", lw=2)
        ax.set_xlabel("Target Value Bins")
        ax.set_ylabel("Prediction Error (Pred - True)")
        ax.tick_params(axis='x', rotation=45)
        ax.set_title(f"Violin Plot of Prediction Errors ({stage.capitalize()} Set)")
        ax.set_ylim(-self.cfg.logging.y_lim, self.cfg.logging.y_lim)
        violin_path = os.path.join(self.output_dir, f"{stage}_violin_epoch_{epoch}.png")
        fig.savefig(violin_path)

        plt.tight_layout()
        png_path = os.path.join(self.output_dir, f"{stage}_all_plots_epoch_{epoch}.png")
        plt.savefig(png_path)
        plt.close(fig)
        
        # Log to WandB
        if self.log_to_wandb:
            self.logger.experiment.log({
                f"{stage}/predictions_plot": wandb.Image(pred_path),
                f"{stage}/errors_plot": wandb.Image(error_path),
                f"{stage}/boxplot": wandb.Image(boxplot_path),
                f"{stage}/violin_plot": wandb.Image(violin_path),
            })


    def _collect_predictions(self, fabric, model, dataloader):
        model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for batch in dataloader:
                x, edge_index, edge_attr, parenthood, batch_idx, y = extract_tensors(batch)
                out = model(x, edge_index, edge_attr, parenthood, batch_idx).squeeze(-1)
                preds.append(out.cpu())
                tgts.append(y.cpu())
        return torch.cat(preds, dim=0).numpy(), torch.cat(tgts, dim=0).numpy()


class TopKModelCheckpoint:
    """
    Saves the top-K models with the best validation loss. Automatically removes
    older models when the limit is exceeded.
    """

    def __init__(self, save_dir: str = "./checkpoints", top_k: int = 5):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)  # Ensure directory exists
        self.top_k = top_k

        # Min heap to track top-K models (sorted by val_loss)
        self.best_models = []  # Stores tuples: (val_loss, epoch, path)

    def on_validation_epoch_end(self, fabric, model, val_loss: float, epoch: int, **kwargs):
        """
        If the current model is among the top-K best (by val_loss), it is saved.
        If there are more than K models, the worst (highest val_loss) is removed.
        """
        if len(self.best_models) < self.top_k or val_loss < self.best_models[-1][0]:  
            # Save new top-K model
            checkpoint_path = os.path.join(self.save_dir, f"best_model_epoch_{epoch}.pt")
            torch.save(get_unwrapped_state_dict(model), checkpoint_path)

            # Add to heap
            heapq.heappush(self.best_models, (val_loss, epoch, checkpoint_path))

            # If we now have more than top_k models, remove the worst one
            if len(self.best_models) > self.top_k:
                worst_model = heapq.heappop(self.best_models)
                os.remove(worst_model[2])  # Delete the worst model file

            fabric.print(f"[TopKModelCheckpoint] Saved model at {checkpoint_path} (val_loss={val_loss:.4f})")

    def restore_best_model(self, fabric, model):
        """
        Restores the best model (lowest validation loss) from disk.
        """
        if len(self.best_models) > 0:
            best_checkpoint = sorted(self.best_models, key=lambda x: x[0])[0][2]  # Best model (lowest val_loss)
            best_model_wts = torch.load(best_checkpoint)
            model.load_state_dict(best_model_wts)
            fabric.print(f"[TopKModelCheckpoint] Restored best model from {best_checkpoint}")

def compute_parameter_norms(model: nn.Module) -> dict:
    """
    Compute the L2 norm of parameters for each layer in the model.
    
    Returns:
        A dictionary mapping layer names to their parameter norms.
    """
    param_norms = {}
    for name, module in model.named_modules():
        if any(p.requires_grad for p in module.parameters()):  # Only consider layers with trainable parameters
            total_norm = torch.tensor(0.0)
            for param in module.parameters():
                if param is not None:
                    param_norm = param.detach().norm(2).cpu()  # L2 norm of parameters
                    total_norm += param_norm ** 2
            if total_norm > 0:
                total_norm = total_norm ** 0.5
                param_norms[name] = total_norm
    return param_norms

if __name__ == "__main__":
    main()

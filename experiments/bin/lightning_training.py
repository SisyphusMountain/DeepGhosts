from lightning.fabric import Fabric
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.utils import to_undirected
import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from wandb.integration.lightning.fabric import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
import copy
import pickle
from typing import Tuple, List
from collections import defaultdict
from tqdm import tqdm
import wandb

# TODO:
# 5. Binning plot throughout training.
# 6. Add learning rate scheduler
# 7. Try to train the models with MAE instead of L2 loss to see the results
# 8. Perform ablation studies, on the presence of batch normalization and other regularization techniques
# 9. Test also weight decay
# 10. Perform hyperparameter optimization
# 11. Put README files everywhere to explain what everything is doing in the GitHub repository
# 12. Add everything on github to make sure it's not lost
# What else to log?
torch.set_float32_matmul_precision("high")


@hydra.main(version_base="1.3", config_path="configs", config_name="debug.yaml")
def main(cfg: DictConfig) -> None:
    # Initialize W&B logger
    logger = WandbLogger(
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # Load dataset
    with open("/media/enzo/Stockage/Output_general/dataset_3/dataset.pkl", "rb") as f:
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
            verbose=True
        )
        callbacks.append(early_stop_cb)

    # 1) Instantiate your new prediction logger callback
    pred_logger_cb = PredictionLoggerCallback(
        output_dir="./results",
        plot_lowess=True,
        log_to_wandb=True,
        frac=0.02,
        wandb_logger=logger,
        plot_every_n_epoch=cfg.logging.log_plot_every_n_epoch  # Generate plot every epoch. Adjust as needed.
    )
    callbacks.append(pred_logger_cb)

    # Initialize Fabric with your callbacks
    fabric = Fabric(loggers=[logger], callbacks=callbacks)
    fabric.seed_everything(cfg.seed)

    # Create LightningDataset (for PyG)
    datamodule = LightningDataset(
        train_dataset=train_data,
        test_dataset=test_data,
        val_dataset=val_data,
        batch_size=cfg.training.batch_size
    )

    fabric.launch()

    # Initialize model
    model = LinearModel(
        in_features=cfg.data.node.in_features,
        out_features=cfg.data.node.out_features
    )

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Initialize learning rate scheduler (if configured)
    if cfg.training.scheduler == "linear_warmup":
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
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # Train
    train_losses, val_losses = train(
        fabric, model, optimizer, train_loader, val_loader, cfg, 
        scheduler=scheduler, 
        early_stop_cb=early_stop_cb
    )


class LinearModel(nn.Module): 
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features)
    def forward(self, x: torch.Tensor, edge_index=None, edge_attr=None, parenthood=None, batch=None) -> torch.Tensor:
        return self.layer(x)


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
    edge_mean, edge_std = edge_stats
    normalized = []
    for sample in dataset:
        new_sample = sample.clone()
        new_sample["node"].x = (new_sample["node"].x - node_mean) / node_std
        new_sample["node", "sends_gene_to", "node"].edge_attr = (
            (new_sample["node", "sends_gene_to", "node"].edge_attr - edge_mean) / edge_std
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

    for epoch in tqdm(range(cfg.training.epochs), desc="Training"):
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
            if hasattr(model, "module"):
                y=1
            current_grad_norms = compute_gradient_norms(model.module if hasattr(model, "module") else model)
            for name, norm in current_grad_norms.items():
                grad_norms[name] += norm
                counts[name] += 1

            optimizer.step()
            epoch_loss += loss.detach().cpu()
        param_norms = compute_parameter_norms(model.module if hasattr(model, "module") else model)
        for name, norm in param_norms.items():
            fabric.log(f"param_norm/{name}", norm, step=epoch)
        # Update scheduler if any
        if scheduler:
            scheduler.step()

        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        fabric.log("train/loss", avg_train_loss, step=epoch)

        # Log gradient norms
        for name in grad_norms:
            avg_grad_norm = grad_norms[name] / counts[name]
            fabric.log(f"grad_norm/{name}", avg_grad_norm, step=epoch)

        # ------------------
        # 2) VALIDATION
        # ------------------
        if epoch % cfg.training.validate_every_n_epoch == 0:
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
                break

    # Restore best weights if early stopping was used
    if early_stop_cb:
        early_stop_cb.restore_best_weights(fabric, model)

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
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: bool = False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0

        self.best_model_wts = None
        self.best_epoch = 0
        self.should_stop = False

    def on_validation_epoch_end(self, fabric, model, val_loss: float, epoch: int, **kwargs):
        if (self.best_loss - val_loss) > self.min_delta:
            # Improved
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.best_model_wts = copy.deepcopy(get_unwrapped_state_dict(model))
        else:
            self.counter += 1
            if self.verbose:
                fabric.print(f"[EarlyStopping] No improvement. Patience {self.counter}/{self.patience}")
            # Check if we exceeded patience
            if self.counter >= self.patience:
                if self.verbose:
                    fabric.print(f"[EarlyStopping] Triggered at epoch {epoch+1}.")
                self.should_stop = True

    def restore_best_weights(self, fabric, model):
        """Call after training loop ends to restore the best model weights."""
        if self.best_model_wts is not None:
            current_state = model.state_dict()
            current_state.update(self.best_model_wts)
            model.load_state_dict(current_state)
            fabric.print(
                f"[EarlyStopping] Restored best model from epoch {self.best_epoch+1} "
                f"with val_loss={self.best_loss:.4f}"
            )


class PredictionLoggerCallback:
    """
    Logs predictions vs. targets at the end of each validation epoch, 
    saves them to a CSV, and optionally plots them with LOWESS + MAE.
    """
    def __init__(
        self, 
        output_dir: str = "./results", 
        plot_lowess: bool = True, 
        log_to_wandb: bool = True,
        frac: float = 0.02,
        wandb_logger=None,
        plot_every_n_epoch: int = 1
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.plot_lowess = plot_lowess
        self.log_to_wandb = log_to_wandb
        self.frac = frac
        self.logger = wandb_logger
        self.plot_every_n_epoch = plot_every_n_epoch

    def on_validation_epoch_end(self, fabric, model, val_loss, epoch, val_loader=None, **kwargs):
        """
        Called at the end of each validation epoch.
        
        fabric: The Fabric object.
        model: The model we just validated.
        val_loss: The validation loss for this epoch.
        epoch: The current epoch number (0-based).
        val_loader: The validation dataloader to collect predictions from.
        """
        # Only log every N epochs
        if epoch % self.plot_every_n_epoch != 0:
            return

        if val_loader is None:
            # If there's no loader, can't collect predictions
            return

        # 1) Collect predictions
        predictions, targets = self._collect_predictions(fabric, model, val_loader)

        # 2) Save to CSV
        csv_path = os.path.join(self.output_dir, f"val_predictions_epoch_{epoch}.csv")
        df = pd.DataFrame({"targets": targets, "predictions": predictions})
        df.to_csv(csv_path, index=False)

        # Also log as W&B table if desired
        if self.log_to_wandb and fabric.global_rank == 0:

            self.logger.log_table(
                key=f"val/predictions",
                dataframe=df,
            )

        # 3) Plot and log the LOWESS + MAE plot
        if self.plot_lowess:
            fig = self._plot_lowess_mae(df, epoch)
            if self.log_to_wandb and fabric.global_rank == 0:
                # log figure
                self.logger.experiment.log({f"val/lowess_plot": fig})
            plt.close(fig)

    def _collect_predictions(self, fabric, model, dataloader):
        model.eval()
        preds = []
        tgts = []
        with torch.no_grad():
            for batch in dataloader:
                x, edge_index, edge_attr, parenthood, batch_idx, y = extract_tensors(batch)
                out = model(x, edge_index, edge_attr, parenthood, batch_idx).squeeze(-1)
                preds.append(out.cpu())
                tgts.append(y.cpu())
        preds = torch.cat(preds, dim=0).numpy()
        tgts = torch.cat(tgts, dim=0).numpy()
        return preds, tgts

    def _plot_lowess_mae(self, df: pd.DataFrame, epoch: int):
        # Sort by target
        df_sorted = df.sort_values('targets')
        mae = np.mean(np.abs(df_sorted['predictions'] - df_sorted['targets']))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            df_sorted['targets'], 
            df_sorted['predictions'],
            alpha=0.3, 
            s=10, 
            label='Data'
        )
        lowess_results = sm.nonparametric.lowess(
            df_sorted['predictions'],
            df_sorted['targets'],
            frac=self.frac,
            it=3
        )
        ax.plot(
            lowess_results[:, 0],
            lowess_results[:, 1],
            color='red',
            lw=2.0,
            label=f'LOWESS (MAE={mae:.2f})'
        )
        x_min, x_max = df_sorted['targets'].min(), df_sorted['targets'].max()
        ax.plot([x_min, x_max], [x_min, x_max], 'k--', alpha=0.6, label='Ideal')

        ax.set_title(f"Validation Predictions (Epoch {epoch})", fontsize=14)
        ax.set_xlabel('Targets')
        ax.set_ylabel('Predictions')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig


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

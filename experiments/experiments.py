import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
import pickle as pkl
from bin.utils import (training_loop,
                       save_histogram_data,
                       calculate_histogram,)
from bin.nn import (LinearModel,
                    MLP,
                    VanillaTransformer,
                    TransformerGCN,
                    TransformerParenthood)


# Ensure reproducibility
torch_geometric.seed.seed_everything(seed=1729)
# Load the dataset, and split it into training and testing sets
# use the debugger to test whether it works
split_ratio = 0.8


def get_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return current_step / warmup_steps
        else:
            # Linear decay
            return max((total_steps - current_step) / (total_steps - warmup_steps), 0.0)
    return lr_lambda


def evaluate_on_dataset(dataset_path, split_ratio=0.8):
    with open(dataset_path, "rb") as f:
        dataset = pkl.load(f)

    num_samples = len(dataset)
    split_idx = int(num_samples * split_ratio)
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]

    train_dataloader = DataLoader(train_dataset, batch_size=30, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=30, shuffle=False)


    model_dict = {"linear": LinearModel(in_features=6, out_features=1),
                "mlp": MLP(in_features=6,
                            out_features=1,
                            hidden_layers=[16, 16]),
                "vanilla_transformer": VanillaTransformer(node_in_features=6,
                                                            d_model=64,
                                                            n_heads=2,
                                                            mlp_expansion_factor=2,
                                                            n_blocks=3,
                                                            dropout=0.1,
                                                            ),
                "transformer_gcn": TransformerGCN(node_in_features=6,
                                                        d_model=64,
                                                        n_heads=2,
                                                        mlp_expansion_factor=2,
                                                        n_blocks=3,
                                                        dropout=0.1,
                                                        ),
                "transformer_parenthood": TransformerParenthood(node_in_features=6,
                                                                d_model=64,
                                                                n_heads=2,
                                                                mlp_expansion_factor=2,
                                                                n_blocks=3,
                                                                dropout=0.1,
                                                                ),}
    learning_rates = {"mlp": 1e-3,
                    "vanilla_transformer": 5e-4,
                    "transformer_gcn": 5e-4,
                    "transformer_parenthood": 5e-4,} # learning rates found by hand tuning
    bin_errors_dict = {}


    model = model_dict["linear"].to("cuda")
    print(f"Training linear")
    optimizer = optim.LBFGS(model.parameters(), lr=0.1)  # using LBFGS instead of Adam
    full_dataset_x_birth = torch.cat([data["node"].x for data in dataset], dim=0).to("cuda")
    full_dataset_y_birth = torch.cat([data["node"].y for data in dataset], dim=0).unsqueeze(-1).to("cuda")
    for epoch in range(50):
        def closure():
            optimizer.zero_grad()
            outputs = model(full_dataset_x_birth)
            loss = F.mse_loss(outputs, full_dataset_y_birth)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
    bin_errors, bin_range = calculate_histogram(test_dataloader, model, 20, "cuda", 50)
    bin_errors_dict["linear"] = bin_errors
    save_histogram_data(bin_errors, bin_range, f"./histogram_data_{dataset_name}_linear.cwv")
    del model
    del optimizer
    torch.cuda.empty_cache()

    for model_name, model in model_dict.items():
        if not model_name == "linear":
            print(f"Training {model_name}")
            optimizer = optim.Adam(model.parameters(), lr=learning_rates[model_name])
            lr_scheduler = LambdaLR(optimizer, lr_lambda=get_lr_lambda(warmup_steps, total_steps))
            training_loop(model,
                        total_steps,
                        train_dataloader,
                        test_dataloader,
                        optimizer,
                        lr_scheduler=lr_scheduler,)
            # the following line averages errors over bins, and also saves errors in a csv file
            bin_errors, bin_range = calculate_histogram(test_dataloader, model, 20, "cuda", 50, save_results_path=f"./errors_{dataset_name}_{model_name}.csv")
            bin_errors_dict[model_name] = bin_errors
            del model
            del optimizer
            torch.cuda.empty_cache()

for i in range(1, 7):
    dataset_name = f"s{i}_normalized"
    dataset_path = f"/media/enzo/Stockage/Output_2/output_s{i}/dataset.pkl"
    # dataset_path = f"/media/enzo/Stockage/Output_general/dataset_1_temp/dataset.pkl"
    with open(dataset_path, "rb") as f:
        dataset = pkl.load(f)

    num_samples = len(dataset)
    split_idx = int(num_samples * split_ratio)
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]

    train_dataloader = DataLoader(train_dataset, batch_size=30, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=30, shuffle=False)


    model_dict = {"linear": LinearModel(in_features=6, out_features=1),
                "mlp": MLP(in_features=6,
                            out_features=1,
                            hidden_layers=[16, 16]),
                "vanilla_transformer": VanillaTransformer(node_in_features=6,
                                                            d_model=64,
                                                            n_heads=2,
                                                            mlp_expansion_factor=2,
                                                            n_blocks=3,
                                                            dropout=0.1,
                                                            ),
                "transformer_gcn": TransformerGCN(node_in_features=6,
                                                        d_model=64,
                                                        n_heads=2,
                                                        mlp_expansion_factor=2,
                                                        n_blocks=3,
                                                        dropout=0.1,
                                                        ),
                "transformer_parenthood": TransformerParenthood(node_in_features=6,
                                                                d_model=64,
                                                                n_heads=2,
                                                                mlp_expansion_factor=2,
                                                                n_blocks=3,
                                                                dropout=0.1,
                                                                ),}
    learning_rates = {"mlp": 1e-3,
                    "vanilla_transformer": 5e-4,
                    "transformer_gcn": 5e-4,
                    "transformer_parenthood": 5e-4,} # learning rates found by hand tuning
    bin_errors_dict = {}


    model = model_dict["linear"].to("cuda")
    print(f"Training linear")
    optimizer = optim.LBFGS(model.parameters(), lr=0.1)  # using LBFGS instead of Adam
    full_dataset_x_birth = torch.cat([data["node"].x for data in dataset], dim=0).to("cuda")
    full_dataset_y_birth = torch.cat([data["node"].y for data in dataset], dim=0).unsqueeze(-1).to("cuda")
    for epoch in range(50):
        def closure():
            optimizer.zero_grad()
            outputs = model(full_dataset_x_birth)
            loss = F.mse_loss(outputs, full_dataset_y_birth)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
    bin_errors, bin_range = calculate_histogram(test_dataloader, model, 20, "cuda", 50)
    bin_errors_dict["linear"] = bin_errors
    save_histogram_data(bin_errors, bin_range, f"./histogram_data_{dataset_name}_linear.cwv")
    del model
    del optimizer
    torch.cuda.empty_cache()

    for model_name, model in model_dict.items():
        if not model_name == "linear":
            print(f"Training {model_name}")
            optimizer = optim.Adam(model.parameters(), lr=learning_rates[model_name])
            lr_scheduler = LambdaLR(optimizer, lr_lambda=get_lr_lambda(warmup_steps, total_steps))
            training_loop(model,
                        total_steps,
                        train_dataloader,
                        test_dataloader,
                        optimizer,
                        lr_scheduler=lr_scheduler,)
            # the following line averages errors over bins, and also saves errors in a csv file
            bin_errors, bin_range = calculate_histogram(test_dataloader, model, 20, "cuda", 50, save_results_path=f"./errors_{dataset_name}_{model_name}.csv")
            bin_errors_dict[model_name] = bin_errors
            del model
            del optimizer
            torch.cuda.empty_cache()


    histogram_data_list = [bin_errors_dict[model_name] for model_name in model_dict.keys()]
    bin_ranges = bin_range
    labels = list(model_dict.keys())
    save_path = f"./histogram_{dataset_name}.png"
    # display_histogram(histogram_data_list, bin_ranges, labels, title='Prediction error for each range of target ghost lengths', save_path=save_path)
    # import pdb; pdb.set_trace()
    # save_histogram_data(bin_errors_dict, bin_range, f"./histogram_data_{dataset_name}_{model_name}.csv")
    # import pdb; pdb.set_trace()


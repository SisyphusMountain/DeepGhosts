import os
import yaml
import random

# Input and output paths
input_files = [f"config_s{i}.yaml" for i in range(1, 7)]
extinction_rates = [0.3, 0.6, 0.9]
n_sp_trees = 50

# Create output directory if it doesn't exist
output_dir = "generated_configs"
os.makedirs(output_dir, exist_ok=True)

# Function to load a YAML file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Function to save a YAML file
def save_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

seed_index = 106
# Generate new YAML files
for i, input_file in enumerate(input_files, start=1):
    config = load_yaml(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    for j, extinction_rate in enumerate(extinction_rates, start=1):
        new_config = config.copy()
        new_config['n_sp_trees'] = n_sp_trees
        new_config['death_rate_low'] = extinction_rate
        new_config['death_rate_high'] = extinction_rate
        new_config['seed'] = seed_index  # Different random seed
        seed_index += 1
        new_config['output_folder'] = f"../Output/output_s{i}_{j}/"  # Modify output folder
        output_file = os.path.join(output_dir, f"{base_name}_{j}.yaml")
        save_yaml(new_config, output_file)

        print(f"Generated: {output_file}")

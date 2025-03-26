for config in linear mlp gcn parenthood vanilla; do
    /home/enzo/miniforge3/envs/DeepGhosts/bin/python /home/enzo/Documents/git/WP1/DeepGhosts/experiments/bin/lightning_training.py --config-name=$config
done
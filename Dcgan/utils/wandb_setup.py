#!usr/bin/env python3
# dcgan/utils/wandb_setup.py
""" visualize experiments """
import wandb

def setup_wandb(project_name, config):
    wandb.init(project=project_name, config=config)

def log_metrics(metrics):
    wandb.log(metrics)

def save_model(model, name):
    model.save(f'models/{name}.h5')
    wandb.save(f'models/{name}.h5')

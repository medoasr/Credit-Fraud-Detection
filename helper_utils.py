import matplotlib.pyplot as plt
import pandas as pd
import yaml
import  seaborn as sns
import joblib
import os
import  torch
def load_Config(Config_path):
    with open(Config_path,'r') as file:
        config=yaml.safe_load(file)
    return config

def save_checkpoint(model, epoch,  title=''):

    checkpoint_path = f'{title}{epoch}.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")

def load_model(file_path):
    try:
        model = joblib.load(file_path)
        return model
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def plot_models_comparsion(model_comparison,path='comparison.png'):
    model_comparison=pd.DataFrame(model_comparison).T
    plt.figure(figsize=(20,20))
    sns.set_theme(font_scale=1.2)
    sns.heatmap(model_comparison,annot=True,cmap='viridis',cbar=True,annot_kws={"size":13},fmt='.2f')
    plt.title('Models Performance',fontsize=20,pad=20)
    plt.xticks(rotation=0, ha='center')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


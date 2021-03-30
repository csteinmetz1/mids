import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_results(
        search_results : list, 
        param_name : str = "peak_distance", 
        output_dir : str = "data/search"
    ):

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(figsize=(6,4))

    acc = []
    param = []

    # extract the param and accuracy list
    for search_result in search_results:
        if "Top-1" in search_result["metrics"]:
            acc.append(search_result["metrics"]["Top-1"] * 100)
        else:
            acc.append(0.0)
        
        param.append(search_result["params"][param_name])

    plt.plot(np.arange(len(param)), acc, marker='o')
    plt.grid(color='lightgray')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylim([0,100])
    plt.xticks(np.arange(len(param)), param)
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f'{param_name}.png'))
    plt.savefig(os.path.join(output_dir, f'{param_name}.pdf'))
    plt.close('all')

if __name__ == '__main__':

    search_result_files = glob.glob(os.path.join("data", "search", "*.pkl"))

    for search_result_file in search_result_files:
        param_name = os.path.basename(search_result_file).replace(".pkl", "")

        with open(search_result_file, 'rb') as f:
            search_results = pickle.load(f)

        plot_results(search_results, param_name=param_name)

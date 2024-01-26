import matplotlib.pyplot as plt
import numpy as np
from plot_functions import moving_average
from stable_baselines3.common.results_plotter import load_results, ts2xy


#TODO: #should plot the mean of all curves as well


def plot_log_results(log_dirs, n_steps, save_model_path, title="Multi Average training reward"):
    """
    plot the results for multiple learning curves of the same training settings.

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    
    """
    fig = plt.figure(title)

    for log_folder in log_dirs:
        i = log_dirs.index(log_folder)
        x, y = ts2xy(load_results(log_folder), "timesteps")
        print(x.shape)
        y = moving_average(y, window=50) 
        # Truncate x
        x = x[len(x) - len(y) :]
        print(x[-1])
        plt.plot(x, y, linewidth=0.9, label=f"agent {i}")

    plt.yscale("linear")
    plt.xlabel("Number of training timesteps")
    plt.ylabel("Reward")
    plt.grid()
    plt.legend()
    #plt.title(title + " Smoothed")
    plt.savefig(f'/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_reward_logs/multi_learning_curve_{save_model_path}.pdf')

log_dirs =['log_dirs/log_dir_dynamic1', 'log_dirs/log_dir_dynamic2', 'log_dirs/log_dir_dynamic3', 'log_dirs/log_dir_dynamic4', 'log_dirs/log_dir_dynamic5', 'log_dirs/log_dir_dynamic6', 'log_dirs/log_dir_dynamic7', 'log_dirs/log_dir_dynamic8', 'log_dirs/log_dir_dynamic9', 'log_dirs/log_dir_dynamic10']

save_model_path='10_times'

plot_log_results(log_dirs, 2048, save_model_path)
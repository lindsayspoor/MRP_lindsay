import matplotlib.pyplot as plt
import numpy as np
from plot_functions import moving_average, calculate_rolling
from stable_baselines3.common.results_plotter import load_results, ts2xy
import seaborn as sns



def plot_log_results(log_dirs, save_model_path, title="Multi Average training reward"):
    """
    plot the results for multiple learning curves of the same training settings.

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    
    """

    reward_schemes=['lr=0.001','lr=0.0001','lr=0.01']
    fig = plt.figure(title, figsize=(6,4))

    for log_folder in log_dirs:
        i = log_dirs.index(log_folder)
        x, y = ts2xy(load_results(log_folder), "episodes")
        x=x[:12500]
        #y=y/np.max(y) #normalize for equal comparison between rewards
        y=y[:12500]
        y, y_error = calculate_rolling(y, window=100)
        #x=np.arange(0,len(y))
        #y = moving_average(y, window=100) #geen  oving average maar averagen over n_steps
        #y = calculate_rolling(y, window=1000)
        # Truncate x
        #x = x[len(x) - len(y) :]


        plt.plot(x, y, linewidth=0.9, label=f"{reward_schemes[i]}", alpha=1)
        plt.fill_between(x, y-y_error, y+y_error, alpha = 0.4)



    #a, b = ts2xy(load_results('log_dirs/log_dir_dynamic_random_2'), "episodes")
    #b, b_error = calculate_rolling(b, window=400)
    #a=np.arange(0,len(b))

    #plt.plot(a, b, linewidth=0.9, label=f"random", alpha=1, color = 'black')
    #plt.fill_between(a, b-b_error, b+b_error, alpha = 0.2, color='black')

    plt.yscale("linear")
    plt.xlabel("Number of training episodes")
    #plt.xlabel("Number of training timesteps")
    plt.ylabel("Reward")
    plt.grid()
    plt.legend(loc='lower right',prop={'size': 10})
    #plt.title(title + " Smoothed")
    plt.savefig(f'/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_reward_logs/ablation_study_learning_curve_{save_model_path}.pdf')

#log_dirs =['log_dirs/log_dir_dynamic1', 'log_dirs/log_dir_dynamic2', 'log_dirs/log_dir_dynamic3', 'log_dirs/log_dir_dynamic4', 'log_dirs/log_dir_dynamic5', 'log_dirs/log_dir_dynamic6', 'log_dirs/log_dir_dynamic7', 'log_dirs/log_dir_dynamic8', 'log_dirs/log_dir_dynamic9', 'log_dirs/log_dir_dynamic10']
#log_dirs = ['log_dirs/log_dir_dynamic']
log_dirs = ['log_dirs/log_dir_reward1','log_dirs/log_dir_lr2','log_dirs/log_dir_lr3']

save_model_path='static_lr'

plot_log_results(log_dirs, save_model_path)
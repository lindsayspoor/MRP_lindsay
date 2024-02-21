# this file contains all the necessary plotting functions
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
import pandas as pd
import seaborn as sns

def plot_benchmark_MWPM(success_rates_all, success_rates_all_MWPM,N_evaluates, error_rates_eval, board_size,path_plot,agent_value_N, agent_value_error_rate,evaluate_fixed):
    plt.figure(figsize=(6,4))
    #for j in range(success_rates.shape[0]):
    if evaluate_fixed:
        plt.plot(N_evaluates, success_rates_all_MWPM[-1,:], label=f'd={board_size} MWPM decoder', linestyle='-.', linewidth=0.5, color='black')
        plt.scatter(N_evaluates, success_rates_all[-1,:], label=f"d={board_size} PPO agent, N={agent_value_N}", marker="^", s=30)
        plt.plot(N_evaluates, success_rates_all[-1,:], linestyle='-.', linewidth=0.5)
        plt.xlabel(r'N')
    else:
        plt.plot(error_rates_eval, success_rates_all_MWPM[-1,:], label=f'd={board_size} MWPM', linestyle='-.',linewidth=0.9, color='black')
        plt.scatter(error_rates_eval, success_rates_all[-1,:], label=f"d={board_size} PPO agent", marker="^", s=40, color = 'blue')
        plt.plot(error_rates_eval, success_rates_all[-1,:], linestyle='-',linewidth=0.9, color='blue')
        plt.xlabel(r'$p$')
        plt.xlim((0,error_rates_eval[-1]+0.005))

    #plt.title(r'Toric Code - PPO vs MWPM')
    plt.ylabel(r'$p_s$')
    plt.legend()
    plt.grid()
    plt.savefig(path_plot)



def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window

    return np.convolve(values, weights, "valid")


def calculate_rolling(values, window):

    means = []
    errorbars = []
    for i in range(0, values.shape[0]):
        mean = np.mean(values[i:(i+window)])
        errors = np.std(values[i:(i+window)])
        means.append(mean)
        errorbars.append(errors)


    return np.array(means), np.array(errorbars)
    




def plot_log_results(log_folder,  save_model_path, title="Average training reward"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")

    y = moving_average(y, window=50) #geen  oving average maar averagen over n_steps
    #y = calculate_rolling(y, window=1000)
    # Truncate x
    x = x[len(x) - len(y) :]

    np.savetxt(f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/log_results/cnn_log_results_{save_model_path}.csv",(x,y) )

    fig = plt.figure(title)
    plt.plot(x, y, color = 'blue', linewidth=0.9)
    plt.yscale("linear")
    plt.xlabel("Number of training timesteps")
    plt.ylabel("Reward")
    plt.grid()
    #plt.title(title + " Smoothed")
    plt.savefig(f'/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_reward_logs/learning_curve_{save_model_path}.pdf')


def plot_log_results_2(log_dirs, save_model_path, title="Multi Average training reward"):

    reward_schemes=['64x64','64x64x64','20x20x20']
    fig = plt.figure(title, figsize=(6,4))

    for log_folder in log_dirs:
        i = log_dirs.index(log_folder)
        x, y = ts2xy(load_results(log_folder), "episodes")
        x=x[:12500]

        y=y[:12500]
        y, y_error = calculate_rolling(y, window=100)



        plt.plot(x, y, linewidth=0.9, label=f"{reward_schemes[i]}", alpha=1)
        plt.fill_between(x, y-y_error, y+y_error, alpha = 0.4)




    plt.yscale("linear")
    plt.xlabel("Number of training episodes")
    #plt.xlabel("Number of training timesteps")
    plt.ylabel("Reward")
    plt.grid()
    plt.legend(loc='lower right',prop={'size': 10})
    plt.show()
    plt.savefig(f'/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_reward_logs/ablation_study_learning_curve_{save_model_path}.pdf')

def plot_log_results_files(file1, file2, file3, timesteps, save_model_path, title="Multi Average training reward"):

    fig = plt.figure(title, figsize=(7,6))

    plt.plot(timesteps, file1, label=r"lr=0.0001", linewidth=0.95)
    plt.plot(timesteps, file2, label=r"lr=0.001", linewidth=0.95)
    plt.plot(timesteps, file3, label=r"lr=0.01", linewidth=0.95)


    plt.yscale("linear")
    plt.xlabel("Number of training iterations", fontsize=16)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    #plt.xlabel("Number of training timesteps")
    plt.ylabel("Reward", fontsize=16)
    plt.grid()
    plt.legend(loc='lower right',prop={'size': 15})
    #plt.title(title + " Smoothed")

    plt.savefig(f'/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_reward_logs/ablation_study_learning_curve_{save_model_path}.pdf')


def plot_log_results_1(file1,timesteps, save_model_path, title="Multi Average training reward"):
    """
    plot the results for multiple learning curves of the same training settings.

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    
    """


    fig = plt.figure(title, figsize=(7.5,6))

    plt.plot(timesteps, file1, linewidth=0.95, color='blue')


    plt.yscale("linear")
    plt.xlabel("Number of training timesteps", fontsize=16)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    #plt.xlabel("Number of training timesteps")
    plt.ylabel("Reward", fontsize=16)
    plt.grid()
    #plt.legend(loc='lower right',prop={'size': 15})
    #plt.title(title + " Smoothed")

    plt.savefig(f'/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_reward_logs/learning_curve_{save_model_path}.pdf')


def render_evaluation(obs0_k,evaluation_settings, actions_k, initial_flips_k):
        size=evaluation_settings['board_size']
        qubit_pos   = [[x,y] for x in range(2*size) for y in range((x+1)%2, 2*size, 2)]
        plaquet_pos = [[x,y] for x in range(1,2*size,2) for y in range(1,2*size,2)]


        fig, (ax3,ax1,ax2) = plt.subplots(1,3, figsize=(15,5))
        a=1/(2*size)

        for i, p in enumerate(plaquet_pos):
            if obs0_k.flatten()[i]==1:

                fc='darkorange'
                plaq = plt.Polygon([[a*p[0], a*(p[1]-1)], [a*(p[0]+1), a*(p[1])], [a*p[0], a*(p[1]+1)], [a*(p[0]-1), a*p[1]] ], fc=fc)
                ax1.add_patch(plaq)

        for i, p in enumerate(plaquet_pos):
            if obs0_k.flatten()[i]==1:

                fc='darkorange'
                plaq = plt.Polygon([[a*p[0], a*(p[1]-1)], [a*(p[0]+1), a*(p[1])], [a*p[0], a*(p[1]+1)], [a*(p[0]-1), a*p[1]] ], fc=fc)
                ax2.add_patch(plaq)

        for i, p in enumerate(plaquet_pos):
            if obs0_k.flatten()[i]==1:

                fc='darkorange'
                plaq = plt.Polygon([[a*p[0], a*(p[1]-1)], [a*(p[0]+1), a*(p[1])], [a*p[0], a*(p[1]+1)], [a*(p[0]-1), a*p[1]] ], fc=fc)
                ax3.add_patch(plaq)

        # Draw lattice
        for x in range(size):
            for y in range(size):
                pos=(2*a*x, 2*a*y)
                width=a*2
                lattice = plt.Rectangle( pos, width, width, fc='none', ec='black' )
                ax1.add_patch(lattice)

        for x in range(size):
            for y in range(size):
                pos=(2*a*x, 2*a*y)
                width=a*2
                lattice = plt.Rectangle( pos, width, width, fc='none', ec='black' )
                ax2.add_patch(lattice)

        for x in range(size):
            for y in range(size):
                pos=(2*a*x, 2*a*y)
                width=a*2
                lattice = plt.Rectangle( pos, width, width, fc='none', ec='black' )
                ax3.add_patch(lattice)

        for i, p in enumerate(qubit_pos):
            pos=(a*p[0], a*p[1])
            fc1='darkgrey'
            if i in list(actions_k[:,0]):
                fc1 = 'darkblue'


            circle1 = plt.Circle( pos , radius=a*0.25, ec='k', fc=fc1)
            ax1.add_patch(circle1)
            ax1.annotate(str(i), pos, fontsize=8, ha="center")
        
        for i, p in enumerate(qubit_pos):
            pos=(a*p[0], a*p[1])
            fc2='darkgrey'
            if i in list(actions_k[:,1]):
                fc2 = 'red'


            circle2 = plt.Circle( pos , radius=a*0.25, ec='k', fc=fc2)
            ax2.add_patch(circle2)
            ax2.annotate(str(i), pos, fontsize=8, ha="center")

        for i, p in enumerate(qubit_pos):
            pos=(a*p[0], a*p[1])
            fc3='darkgrey'
            if p in list(initial_flips_k)[0]:
                fc3 = 'magenta'


            circle3 = plt.Circle( pos , radius=a*0.25, ec='k', fc=fc3)
            ax3.add_patch(circle3)
            ax3.annotate(str(i), pos, fontsize=8, ha="center")

        ax1.set_xlim([-.1,1.1])
        ax1.set_ylim([-.1,1.1])
        ax1.set_aspect(1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title("actions agent")
        ax2.set_xlim([-.1,1.1])
        ax2.set_ylim([-.1,1.1])
        ax2.set_aspect(1)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("actions MWPM")
        ax1.axis('off')
        ax2.axis('off')
        ax3.set_xlim([-.1,1.1])
        ax3.set_ylim([-.1,1.1])
        ax3.set_aspect(1)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title("initial qubit flips")
        ax3.axis('off')
        plt.show()


def plot_benchmark_MWPM_2(path_plot,success_rates_all_3, success_rates_all_MWPM_3, success_rates_all_5, success_rates_all_MWPM_5, success_rates_all_7, success_rates_all_MWPM_7,success_rates_all_MWPM_15,error_rates_eval):
    

    plt.figure(figsize=(7,6))
    #for j in range(success_rates.shape[0]):
    plt.grid()
    plt.plot(error_rates_eval, success_rates_all_MWPM_3, label=f'd=3 MWPM', linestyle='-.',linewidth=1.1, color='blue')
    plt.scatter(error_rates_eval, success_rates_all_3, label=f"d=3 PPO agent", marker="^", s=45, color = 'blue', edgecolors='black',zorder=10)
    plt.plot(error_rates_eval, success_rates_all_MWPM_5, label=f'd=5 MWPM', linestyle=':',linewidth=1.5, color='darkorange')
    plt.scatter(error_rates_eval, success_rates_all_5, label=f"d=5 PPO agent", marker="o", s=45, color = 'darkorange', edgecolors='black',zorder=11)
    plt.plot(error_rates_eval, success_rates_all_MWPM_7, label=f'd=7 MWPM', linestyle='--',linewidth=1.1, color='green')
    plt.scatter(error_rates_eval, success_rates_all_7, label=f"d=7 PPO agent", marker="s", s=35, color = 'green', edgecolors='black', zorder=12)
    plt.plot(error_rates_eval, success_rates_all_MWPM_15, label=f'd=15 MWPM', linestyle='--',linewidth=0.7, color='purple')
    #plt.plot(error_rates_eval, success_rates_all_MWPM_30, label=f'd=30 MWPM', linestyle='--',linewidth=0.7, color='black')
    plt.xlabel(r'$p_{error}$', fontsize=16)
    plt.xlim((0.005,error_rates_eval[-1]+0.005))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.title(r'Toric Code - PPO vs MWPM')
    plt.ylabel(r'$p_s$', fontsize=16)
    plt.legend(prop={'size': 15})
    plt.savefig(path_plot)
    plt.show()

def plot_different_error_rates(path_plot,success_rates_all_5, success_rates_all_MWPM_5, success_rates_err1, success_rates_err2,success_rates_err3,success_rates_err4,success_rates_err5,success_rates_err6,success_rates_curr,error_rates_eval):#  success_rates_err4, success_rates_err5, success_rates_err6,error_rates_eval):
    

    plt.figure(figsize=(7,5))
    #for j in range(success_rates.shape[0]):
    plt.grid()
    plt.plot(error_rates_eval, success_rates_all_MWPM_5, label=f'd=5 MWPM', linestyle='-.',linewidth=1.1, color='black')
    #plt.plot(error_rates_eval, success_rates_err1, linestyle='-',linewidth=1.1)
    #plt.scatter(error_rates_eval, success_rates_err1, label=r'd=5 PPO, $p_{error}=0.01$', marker='s', s=15)
    #plt.plot(error_rates_eval, success_rates_err2, linestyle='-',linewidth=1.1)
    #plt.scatter(error_rates_eval, success_rates_err2, label=r'd=5 PPO, $p_{error}=0.038$', marker='s', s=15)
    #plt.plot(error_rates_eval, success_rates_err3, linestyle='-',linewidth=1.1)
    #plt.scatter(error_rates_eval, success_rates_err3, label=r'd=5 PPO, $p_{error}=0.066$', marker='s', s=15)
    #plt.plot(error_rates_eval, success_rates_err4,  linestyle='-',linewidth=1.1)
    #plt.scatter(error_rates_eval, success_rates_err4, label=r'd=5 PPO, $p_{error}=0.094$', marker='s', s=15)
    #plt.plot(error_rates_eval, success_rates_err5, linestyle='-',linewidth=1.1)
    #plt.scatter(error_rates_eval, success_rates_err5, label=r'd=5 PPO, $p_{error}=0.129$', marker='s', s=15)
    #plt.plot(error_rates_eval, success_rates_err6, linestyle='-',linewidth=1.1)
    #plt.scatter(error_rates_eval, success_rates_err6, label=r'd=5 PPO, $p_{error}=0.15$', marker='s', s=15)
    plt.plot(error_rates_eval, success_rates_all_5, linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, success_rates_all_5, label=r'd=5 PPO, $p_{error}=0.1$', marker='s', s=15)
    plt.plot(error_rates_eval, success_rates_curr, linestyle='-',linewidth=1.3, color = 'fuchsia')
    plt.scatter(error_rates_eval, success_rates_curr, label=f"d=5 PPO, curriculum learning", marker="o", s=40, color = 'fuchsia', edgecolors='black',zorder=6)
    plt.xlabel(r'$p_{error}$')
    plt.xlim((0,error_rates_eval[-1]+0.005))

    #plt.title(r'Toric Code - PPO vs MWPM')
    plt.ylabel(r'$p_s$')
    plt.legend()
    plt.savefig(path_plot)
    plt.show()

def plot_different_timesteps(path_plot,success_rates_all_MWPM_5, success_rates_err_64_64,success_rates_err_64_64_64,success_rates_err_20_20_20,error_rates_eval):#  success_rates_err4, success_rates_err5, success_rates_err6,error_rates_eval):
    

    plt.figure(figsize=(7,6))
    #for j in range(success_rates.shape[0]):
    plt.grid()
    plt.plot(error_rates_eval, success_rates_all_MWPM_5, label=f'd=5 MWPM', linestyle='-.',linewidth=1.1, color='black')
    plt.plot(error_rates_eval, success_rates_err_64_64, linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, success_rates_err_64_64, label=r'd=5 PPO, $64x64$', marker='s', s=15)
    plt.plot(error_rates_eval, success_rates_err_64_64_64, linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, success_rates_err_64_64_64, label=r'd=5 PPO, $64x64x64$', marker='s', s=15)
    plt.plot(error_rates_eval, success_rates_err_20_20_20, linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, success_rates_err_20_20_20, label=r'd=5 PPO, $20x20x20$', marker='s', s=15)
    #plt.plot(error_rates_eval, success_rates_all_5, linestyle='-',linewidth=1.3, color = 'fuchsia')
    #plt.scatter(error_rates_eval, success_rates_all_5, label=f"d=5 PPO, curriculum learning", marker="o", s=40, color = 'fuchsia', edgecolors='black',zorder=6)
    plt.xlabel(r'$p_{error}$', fontsize=16)
    plt.xlim((0.005,error_rates_eval[-1]+0.005))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.title(r'Toric Code - PPO vs MWPM')
    plt.ylabel(r'$p_s$', fontsize = 16)
    plt.legend(prop={'size': 15})
    plt.savefig(path_plot)
    plt.show()

def plot_different_lr(path_plot, success_rates_all_MWPM_5, success_rates_lr0001,success_rates_lr001,success_rates_lr01,error_rates_eval):#  success_rates_err4, success_rates_err5, success_rates_err6,error_rates_eval):
    

    plt.figure(figsize=(7,6))
    #for j in range(success_rates.shape[0]):
    plt.grid()
    plt.plot(error_rates_eval, success_rates_all_MWPM_5, label=f'd=5 MWPM', linestyle='-.',linewidth=1.1, color='black')
    plt.plot(error_rates_eval, success_rates_lr0001, linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, success_rates_lr0001, label=r'lr=0.0001', marker='s', s=15)
    plt.plot(error_rates_eval, success_rates_lr001, linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, success_rates_lr001, label=r'lr.0.001', marker='s', s=15)
    plt.plot(error_rates_eval, success_rates_lr01, linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, success_rates_lr01, label=r'lr=0.01', marker='s', s=15)
    #plt.plot(error_rates_eval, success_rates_all_5, linestyle='-',linewidth=1.3, color = 'fuchsia')
    #plt.scatter(error_rates_eval, success_rates_all_5, label=f"d=5 PPO, curriculum learning", marker="o", s=40, color = 'fuchsia', edgecolors='black',zorder=6)
    plt.xlabel(r'$p_{error}$',fontsize=16)
    plt.xlim((0.005,error_rates_eval[-1]+0.005))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.title(r'Toric Code - PPO vs MWPM')
    plt.ylabel(r'$p_s$', fontsize=16)
    plt.legend(prop={'size': 15})
    plt.savefig(path_plot)
    plt.show()


def plot_mean_moves(path_plot, error_rates_eval, mean_moves1,mean_moves2,mean_moves3,mean_moves4,mean_moves5,mean_moves6,mean_moves_curr):

    plt.figure(figsize=(7,6))

    plt.grid()
    #plt.scatter(error_rates_eval, mean_moves1, label=r'd=5 PPO, $p_{error}=0.1$', marker='s', s=15)
    #plt.plot(error_rates_eval, mean_moves1, linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, mean_moves2, label=r'd=5 PPO, lr=0.0001', marker='s', s=15)
    plt.plot(error_rates_eval, mean_moves2, linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, mean_moves3, label=r'd=5 PPO, lr=0.001', marker='s', s=15)
    plt.plot(error_rates_eval, mean_moves3,  linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, mean_moves4, label=r'd=5 PPO, lr=0.01', marker='s', s=15)
    plt.plot(error_rates_eval, mean_moves4, linestyle='-',linewidth=1.1)
    #plt.scatter(error_rates_eval, mean_moves5, label=r'd=5 PPO, $p_{error}=0.129$', marker='s', s=15)
    #plt.plot(error_rates_eval, mean_moves5, linestyle='-',linewidth=1.1)
    #plt.scatter(error_rates_eval, mean_moves6, label=r'd=5 PPO, $p_{error}=0.15$', marker='s', s=15)
    #plt.plot(error_rates_eval, mean_moves6, linestyle='-',linewidth=1.1)
    #plt.plot(error_rates_eval, mean_moves_curr, linestyle='-',linewidth=1.3, color = 'fuchsia')
    #plt.scatter(error_rates_eval, mean_moves_curr, label=f"d=5 PPO, curriculum learning", marker="o", s=40, color = 'fuchsia', edgecolors='black',zorder=6)
    plt.xlabel(r'$p_{error}$', fontsize=16)
    plt.xlim((0.005,error_rates_eval[-1]+0.005))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.title(r'Toric Code - PPO vs MWPM')
    plt.ylabel("Mean number of moves per game", fontsize=16)
    plt.legend(prop={'size': 15})
    plt.savefig(path_plot)
    plt.show()
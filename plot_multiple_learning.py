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

    reward_schemes=['64x64','64x64x64','20x20x20']
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
    plt.show()
    plt.savefig(f'/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_reward_logs/ablation_study_learning_curve_{save_model_path}.pdf')

def plot_log_results_files(file1, file2, file3, timesteps, save_model_path, title="Multi Average training reward"):
    """
    plot the results for multiple learning curves of the same training settings.

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    
    """


    fig = plt.figure(title, figsize=(7,6))

    plt.plot(timesteps, file1, label=r"lr=0.0001", linewidth=0.95)
    plt.plot(timesteps, file2, label=r"lr=0.001", linewidth=0.95)
    plt.plot(timesteps, file3, label=r"lr=0.01", linewidth=0.95)


    



    #a, b = ts2xy(load_results('log_dirs/log_dir_dynamic_random_2'), "episodes")
    #b, b_error = calculate_rolling(b, window=400)
    #a=np.arange(0,len(b))

    #plt.plot(a, b, linewidth=0.9, label=f"random", alpha=1, color = 'black')
    #plt.fill_between(a, b-b_error, b+b_error, alpha = 0.2, color='black')

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




    #a, b = ts2xy(load_results('log_dirs/log_dir_dynamic_random_2'), "episodes")
    #b, b_error = calculate_rolling(b, window=400)
    #a=np.arange(0,len(b))

    #plt.plot(a, b, linewidth=0.9, label=f"random", alpha=1, color = 'black')
    #plt.fill_between(a, b-b_error, b+b_error, alpha = 0.2, color='black')

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


#log_dirs =['log_dirs/log_dir_dynamic1', 'log_dirs/log_dir_dynamic2', 'log_dirs/log_dir_dynamic3', 'log_dirs/log_dir_dynamic4', 'log_dirs/log_dir_dynamic5', 'log_dirs/log_dir_dynamic6', 'log_dirs/log_dir_dynamic7', 'log_dirs/log_dir_dynamic8', 'log_dirs/log_dir_dynamic9', 'log_dirs/log_dir_dynamic10']
#log_dirs = ['log_dirs/log_dir_dynamic']
#log_dirs = ['log_dirs/log_dir_lr1','log_dirs/log_dir_lr2','log_dirs/log_dir_lr3']
log_dirs=['log_dirs/log_dir_64_64','log_dirs/log_dir_64_64_64','log_dirs/log_dir_20_20_20']

static_20_20_20 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/log_results/cnn_log_results_board_size=5error_model=0error_rate=0.1l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=3ent_coef=0.05clip_range=0.1.csv")
static_64_64_64 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/log_results/cnn_log_results_board_size=5error_model=0error_rate=0.1l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=2ent_coef=0.05clip_range=0.1.csv")
static_64_64 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/log_results/cnn_log_results_board_size=5error_model=0error_rate=0.1l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1.csv")

lr_0001 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/log_results/cnn_log_results_board_size=5error_model=0error_rate=0.1l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.0001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1.csv")
lr_001 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/log_results/cnn_log_results_board_size=5error_model=0error_rate=0.1l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=6ent_coef=0.05clip_range=0.1.csv")
lr_01 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/log_results/.csv")

ent_0 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/log_results/cnn_log_results_board_size=5error_model=0error_rate=0.1l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0clip_range=0.1.csv")
ent_01 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/log_results/cnn_log_results_board_size=5error_model=0error_rate=0.1l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=8ent_coef=0.01clip_range=0.1.csv")
ent_05 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/log_results/cnn_log_results_board_size=5error_model=0error_rate=0.1l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=9ent_coef=0.05clip_range=0.1.csv")


board_3 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/log_results/cnn_log_results_board_size=3error_model=0error_rate=0.1l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.01clip_range=0.1.csv")
board_5 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/log_results/cnn_log_results_board_size=5error_model=0error_rate=0.1l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.01clip_range=0.1.csv")
board_7 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/log_results/cnn_log_results_board_size=7error_model=0error_rate=0.1l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.01clip_range=0.1.csv")


reward1=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/log_results/cnn_log_results_board_size=5error_model=0error_rate=0.1l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.01clip_range=0.1.csv")
reward2=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/log_results/cnn_log_results_board_size=5error_model=0error_rate=0.1l_reward=10s_reward=100c_reward=-10i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.01clip_range=0.1.csv")
reward3=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/log_results/cnn_log_results_board_size=5error_model=0error_rate=0.1l_reward=1s_reward=2c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.01clip_range=0.1.csv")
reward4=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/log_results/cnn_log_results_board_size=5error_model=0error_rate=0.1l_reward=1s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.01clip_range=0.1.csv")

#print(board_3[1].shape)
print(reward1.shape)
print(reward2.shape)
print(reward3.shape)
print(reward4.shape)

number_timesteps = np.arange(0,3000)


save_model_path='lr'

#plot_log_results_1(board_7[1],number_timesteps,save_model_path)
#plot_log_results(log_dirs, save_model_path)
#plot_log_results_files(ent_0[1,:10000],ent_01[1,:10000],ent_05[1,:10000],number_timesteps,save_model_path)
#plot_log_results_files(static_64_64[1,:6000],static_64_64_64[1,:6000],static_20_20_20[1,:6000],number_timesteps,save_model_path)
plot_log_results_files(lr_0001[1,:3000],lr_001[1,:3000],lr_01[1,:3000],number_timesteps,save_model_path)
#plot_log_results_files(reward1[1,:10000]/np.abs(np.max(reward1[1])),reward2[1,:10000]/np.abs(np.max(reward2[1])),reward3[1,:10000]/np.abs(np.max(reward3[1])),reward4[1,:10000]/np.abs(np.max(reward4[1])),number_timesteps,save_model_path)
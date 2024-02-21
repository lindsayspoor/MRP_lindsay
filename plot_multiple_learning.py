import matplotlib.pyplot as plt
import numpy as np
from plot_functions import moving_average, calculate_rolling
from stable_baselines3.common.results_plotter import load_results, ts2xy
import seaborn as sns
from plot_functions import plot_log_results, plot_log_results_1, plot_log_results_2, plot_log_results_files



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
#plot_log_results_2(log_dirs, save_model_path)
#plot_log_results_files(ent_0[1,:10000],ent_01[1,:10000],ent_05[1,:10000],number_timesteps,save_model_path)
#plot_log_results_files(static_64_64[1,:6000],static_64_64_64[1,:6000],static_20_20_20[1,:6000],number_timesteps,save_model_path)
plot_log_results_files(lr_0001[1,:3000],lr_001[1,:3000],lr_01[1,:3000],number_timesteps,save_model_path)
#plot_log_results_files(reward1[1,:10000]/np.abs(np.max(reward1[1])),reward2[1,:10000]/np.abs(np.max(reward2[1])),reward3[1,:10000]/np.abs(np.max(reward3[1])),reward4[1,:10000]/np.abs(np.max(reward4[1])),number_timesteps,save_model_path)
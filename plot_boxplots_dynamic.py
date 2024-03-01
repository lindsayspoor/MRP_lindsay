import numpy as np
from plot_functions_old import plot_multiple_boxes_dynamic

folder = "/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/dynamic_ppo/moves_dynamic_agent"

'''
moves_random1 = np.loadtxt(f"{folder}/moves_dynamic_ppo_board_size=3error_rate=0.01lr=0.001total_timesteps=1n_steps=2048mask_actions=Truefixed=Truec_reward=1e_reward=10l_reward=1N=1iteration_step=3ent_coef=0.05clip_range=0.1new_N=1_1.csv")
moves_agent1 = np.loadtxt(f"{folder}/moves_dynamic_ppo_board_size=3error_rate=0.01lr=0.001total_timesteps=1000000n_steps=2048mask_actions=Truefixed=Truec_reward=1e_reward=10l_reward=1N=1iteration_step=3ent_coef=0.05clip_range=0.1new_N=1_1.csv")
moves_random2 = np.loadtxt(f"{folder}/moves_dynamic_ppo_board_size=3error_rate=0.01lr=0.001total_timesteps=1n_steps=2048mask_actions=Truefixed=Truec_reward=1e_reward=10l_reward=1N=2iteration_step=3ent_coef=0.05clip_range=0.1new_N=2_2.csv")
moves_agent2 = np.loadtxt(f"{folder}/moves_dynamic_ppo_board_size=3error_rate=0.01lr=0.001total_timesteps=1000000n_steps=2048mask_actions=Truefixed=Truec_reward=1e_reward=10l_reward=1N=2iteration_step=3ent_coef=0.05clip_range=0.1new_N=2_2.csv")
moves_random3 = np.loadtxt(f"{folder}/moves_dynamic_ppo_board_size=3error_rate=0.01lr=0.001total_timesteps=1n_steps=2048mask_actions=Truefixed=Truec_reward=1e_reward=1l_reward=1N=1iteration_step=2ent_coef=0.05clip_range=0.1new_N=1_1.csv")
moves_agent3 = np.loadtxt(f"{folder}/moves_dynamic_ppo_board_size=3error_rate=0.01lr=0.001total_timesteps=1000000n_steps=2048mask_actions=Truefixed=Truec_reward=1e_reward=10l_reward=1N=1iteration_step=2ent_coef=0.05clip_range=0.1new_N=1_1.csv")
'''

moves_random = np.loadtxt(f"{folder}/moves_dynamic_ppo_board_size=3error_rate=0.01lr=0.001total_timesteps=1n_steps=2048mask_actions=Truefixed=Truec_reward=1e_reward=1l_reward=1N=1iteration_step=2ent_coef=0.05clip_range=0.1new_N=1_1.csv")
moves_agent1=np.loadtxt(f"{folder}/moves_dynamic_ppo_board_size=3error_rate=0.01lr=0.0001total_timesteps=3000000n_steps=2048mask_actions=Truefixed=Truec_reward=1e_reward=10l_reward=1N=1iteration_step=2ent_coef=0.05clip_range=0.1new_N=1_1.csv")
moves_agent2=np.loadtxt(f"{folder}/moves_dynamic_ppo_board_size=3error_rate=0.01lr=0.001total_timesteps=3000000n_steps=2048mask_actions=Truefixed=Truec_reward=1e_reward=10l_reward=1N=1iteration_step=2ent_coef=0.05clip_range=0.1new_N=1_1.csv")
moves_agent3=np.loadtxt(f"{folder}/moves_dynamic_ppo_board_size=3error_rate=0.01lr=annealingtotal_timesteps=3000000n_steps=2048mask_actions=Truefixed=Truec_reward=1e_reward=10l_reward=1N=1iteration_step=2ent_coef=0.05clip_range=0.1new_N=1_1.csv")

#path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_dymanic_boxes_othersettings.pdf"

path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_dymanic_lr.pdf"

plot_multiple_boxes_dynamic(path_plot, moves_random, moves_agent1, moves_agent2, moves_agent3)
import numpy as np
from plot_functions import plot_fixed_vs_curr, plot_benchmark_MWPM_2


error_rates_eval=list(np.linspace(0.01,0.15,10))

folder = '/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/static_ppo'


path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_paper_reward_scheme.pdf"


success_rates_agent5 = np.loadtxt(f"{folder}/success_rates_agent/success_rates_ppo_board_size=5error_rate=0.15l_reward=-1s_reward=-1c_reward=-1i_reward=-2lr=0.001total_timesteps=1000000mask_actions=Truecorrelated=Falsefixed=FalseN=1ent_coef=0.01clip_range=0.1max_moves=200_0.1.csv")
success_rates_MWPM5 = np.loadtxt(f"{folder}/success_rates_MWPM/success_rates_ppo_board_size=5error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-2lr=0.001total_timesteps=1000000mask_actions=Truecorrelated=Falsefixed=FalseN=1ent_coef=0.01clip_range=0.1max_moves=200_0.1.csv")
success_rates_agent5_curr = np.loadtxt(f"{folder}/success_rates_agent/success_rates_ppo_board_size=5error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-2lr=0.001total_timesteps=1000000mask_actions=Truecorrelated=Falsefixed=FalseN=1ent_coef=0.01clip_range=0.1_0.15.csv")

plot_fixed_vs_curr(path_plot,success_rates_agent5, success_rates_MWPM5,success_rates_agent5_curr,error_rates_eval)
'''


success_rates_agent3 = np.loadtxt(f"{folder}/success_rates_agent/success_rates_ppo_board_size=3error_rate=0.15l_reward=-1s_reward=-1c_reward=-1i_reward=-2lr=0.001total_timesteps=1000000mask_actions=Truecorrelated=Truefixed=FalseN=1ent_coef=0.01clip_range=0.1max_moves=200_0.1.csv")
success_rates_MWPM3 = np.loadtxt(f"{folder}/success_rates_MWPM/success_rates_ppo_board_size=3error_rate=0.15l_reward=-1s_reward=-1c_reward=-1i_reward=-2lr=0.001total_timesteps=1000000mask_actions=Truecorrelated=Truefixed=FalseN=1ent_coef=0.01clip_range=0.1max_moves=200_0.1.csv")
success_rates_agent5 = np.loadtxt(f"{folder}/success_rates_agent/success_rates_ppo_board_size=5error_rate=0.15l_reward=-1s_reward=-1c_reward=-1i_reward=-2lr=0.001total_timesteps=1000000mask_actions=Truecorrelated=Truefixed=FalseN=1ent_coef=0.01clip_range=0.1max_moves=200_0.1.csv")
success_rates_MWPM5 = np.loadtxt(f"{folder}/success_rates_MWPM/success_rates_ppo_board_size=5error_rate=0.15l_reward=-1s_reward=-1c_reward=-1i_reward=-2lr=0.001total_timesteps=1000000mask_actions=Truecorrelated=Truefixed=FalseN=1ent_coef=0.01clip_range=0.1max_moves=200_0.1.csv")
success_rates_agent7 = np.loadtxt(f"{folder}/success_rates_agent/success_rates_ppo_board_size=7error_rate=0.15l_reward=-1s_reward=-1c_reward=-1i_reward=-2lr=0.001total_timesteps=1000000mask_actions=Truecorrelated=Truefixed=FalseN=1ent_coef=0.01clip_range=0.1max_moves=200_0.1.csv")
success_rates_MWPM7 = np.loadtxt(f"{folder}/success_rates_MWPM/success_rates_ppo_board_size=7error_rate=0.15l_reward=-1s_reward=-1c_reward=-1i_reward=-2lr=0.001total_timesteps=1000000mask_actions=Truecorrelated=Truefixed=FalseN=1ent_coef=0.01clip_range=0.1max_moves=200_0.1.csv")
success_rates_MWPM15 = np.loadtxt(f"{folder}/success_rates_MWPM/success_rates_ppo_board_size=15error_rate=0.15l_reward=-1s_reward=-1c_reward=-1i_reward=-2lr=0.001total_timesteps=5000mask_actions=Truecorrelated=Truefixed=FalseN=1ent_coef=0.01clip_range=0.1max_moves=200_0.1.csv")
'''
#path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_vs_MWPM_3_vs_5_vs_7_correlated.pdf"

#plot_benchmark_MWPM_2(path_plot,success_rates_agent3, success_rates_MWPM3, success_rates_agent5, success_rates_MWPM5, success_rates_agent7, success_rates_MWPM7,success_rates_MWPM15,error_rates_eval)
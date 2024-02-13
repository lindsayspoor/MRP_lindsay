import numpy as np
import matplotlib.pyplot as plt

def plot_benchmark_MWPM(success_rates_all_3, success_rates_all_MWPM_3, success_rates_all_5, success_rates_all_MWPM_5, success_rates_all_7, success_rates_all_MWPM_7,error_rates_eval):
    

    plt.figure(figsize=(6,4))
    #for j in range(success_rates.shape[0]):
    plt.grid()
    plt.plot(error_rates_eval, success_rates_all_MWPM_3, label=f'd=3 MWPM', linestyle='-.',linewidth=1.1, color='blue')
    plt.scatter(error_rates_eval, success_rates_all_3, label=f"d=3 PPO agent", marker="^", s=45, color = 'blue', edgecolors='black')
    plt.plot(error_rates_eval, success_rates_all_MWPM_5, label=f'd=5 MWPM', linestyle=':',linewidth=1.5, color='darkorange')
    plt.scatter(error_rates_eval, success_rates_all_5, label=f"d=5 PPO agent", marker="o", s=45, color = 'darkorange', edgecolors='black')
    plt.plot(error_rates_eval, success_rates_all_MWPM_7, label=f'd=7 MWPM', linestyle='--',linewidth=1.1, color='green')
    plt.scatter(error_rates_eval, success_rates_all_7, label=f"d=7 PPO agent", marker="s", s=35, color = 'green', edgecolors='black')
    plt.xlabel(r'$p_{error}$')
    plt.xlim((0,error_rates_eval[-1]+0.005))

    #plt.title(r'Toric Code - PPO vs MWPM')
    plt.ylabel(r'$p_s$')
    plt.legend()
    plt.savefig(path_plot)
    plt.show()

error_rates_eval=list(np.linspace(0.01,0.15,10))


success_rates_all_3 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=3error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.15.csv")
success_rates_all_MWPM_3 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_MWPM/success_rates_ppo_board_size=3error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.15.csv")
success_rates_all_5 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=5000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.15.csv")
success_rates_all_MWPM_5 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_MWPM/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=5000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.15.csv")
success_rates_all_7 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=7error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=5000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.15.csv")
success_rates_all_MWPM_7 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_MWPM/success_rates_ppo_board_size=7error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=5000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.15.csv")

path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_vs_MWPM_3_vs_5.pdf"

plot_benchmark_MWPM(success_rates_all_3, success_rates_all_MWPM_3, success_rates_all_5, success_rates_all_MWPM_5,success_rates_all_7, success_rates_all_MWPM_7, error_rates_eval)
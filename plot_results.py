import numpy as np
import matplotlib.pyplot as plt

def plot_benchmark_MWPM(success_rates_all_3, success_rates_all_MWPM_3, success_rates_all_5, success_rates_all_MWPM_5, success_rates_all_7, success_rates_all_MWPM_7,success_rates_all_MWPM_15,error_rates_eval):
    

    plt.figure(figsize=(6,4))
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
    plt.xlabel(r'$p_{error}$')
    plt.xlim((0,error_rates_eval[-1]+0.005))

    #plt.title(r'Toric Code - PPO vs MWPM')
    plt.ylabel(r'$p_s$')
    plt.legend()
    plt.savefig(path_plot)
    plt.show()

def plot_different_error_rates(success_rates_all_5, success_rates_all_MWPM_5, success_rates_err1, success_rates_err2,success_rates_err3,success_rates_err4,success_rates_err5,success_rates_err6,error_rates_eval):#  success_rates_err4, success_rates_err5, success_rates_err6,error_rates_eval):
    

    plt.figure(figsize=(7,5))
    #for j in range(success_rates.shape[0]):
    plt.grid()
    plt.plot(error_rates_eval, success_rates_all_MWPM_5, label=f'd=5 MWPM', linestyle='-.',linewidth=1.1, color='black')
    plt.plot(error_rates_eval, success_rates_err1, linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, success_rates_err1, label=r'd=5 PPO, $p_{error}=0.01$', marker='s', s=15)
    plt.plot(error_rates_eval, success_rates_err2, linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, success_rates_err2, label=r'd=5 PPO, $p_{error}=0.038$', marker='s', s=15)
    plt.plot(error_rates_eval, success_rates_err3, linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, success_rates_err3, label=r'd=5 PPO, $p_{error}=0.066$', marker='s', s=15)
    plt.plot(error_rates_eval, success_rates_err4,  linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, success_rates_err4, label=r'd=5 PPO, $p_{error}=0.094$', marker='s', s=15)
    plt.plot(error_rates_eval, success_rates_err5, linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, success_rates_err5, label=r'd=5 PPO, $p_{error}=0.129$', marker='s', s=15)
    plt.plot(error_rates_eval, success_rates_err6, linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, success_rates_err6, label=r'd=5 PPO, $p_{error}=0.15$', marker='s', s=15)
    plt.plot(error_rates_eval, success_rates_all_5, linestyle='-',linewidth=1.3, color = 'fuchsia')
    plt.scatter(error_rates_eval, success_rates_all_5, label=f"d=5 PPO, curriculum learning", marker="o", s=40, color = 'fuchsia', edgecolors='black',zorder=6)
    plt.xlabel(r'$p_{error}$')
    plt.xlim((0,error_rates_eval[-1]+0.005))

    #plt.title(r'Toric Code - PPO vs MWPM')
    plt.ylabel(r'$p_s$')
    plt.legend()
    plt.savefig(path_plot)
    plt.show()

def plot_different_timesteps(success_rates_all_5, success_rates_all_MWPM_5, success_rates_err6,success_rates_err6_5mil,success_rates_err6_5mil_2layers,success_rates_err6_5mil_2layers20,error_rates_eval):#  success_rates_err4, success_rates_err5, success_rates_err6,error_rates_eval):
    

    plt.figure(figsize=(7,5))
    #for j in range(success_rates.shape[0]):
    plt.grid()
    plt.plot(error_rates_eval, success_rates_all_MWPM_5, label=f'd=5 MWPM', linestyle='-.',linewidth=1.1, color='black')
    plt.plot(error_rates_eval, success_rates_err6, linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, success_rates_err6, label=r'd=5 PPO, $p_{error}=0.15, 1e6 (64x64)$', marker='s', s=15)
    plt.plot(error_rates_eval, success_rates_err6_5mil, linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, success_rates_err6_5mil, label=r'd=5 PPO, $p_{error}=0.15, 5e6 (64x64)$', marker='s', s=15)
    plt.plot(error_rates_eval, success_rates_err6_5mil_2layers, linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, success_rates_err6_5mil_2layers, label=r'd=5 PPO, $p_{error}=0.15, 5e6 (64x64x64)$', marker='s', s=15)
    plt.plot(error_rates_eval, success_rates_err6_5mil_2layers20, linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, success_rates_err6_5mil_2layers20, label=r'd=5 PPO, $p_{error}=0.15, 5e6 (20x20x20)$', marker='s', s=15)
    #plt.plot(error_rates_eval, success_rates_all_5, linestyle='-',linewidth=1.3, color = 'fuchsia')
    #plt.scatter(error_rates_eval, success_rates_all_5, label=f"d=5 PPO, curriculum learning", marker="o", s=40, color = 'fuchsia', edgecolors='black',zorder=6)
    plt.xlabel(r'$p_{error}$')
    plt.xlim((0,error_rates_eval[-1]+0.005))

    #plt.title(r'Toric Code - PPO vs MWPM')
    plt.ylabel(r'$p_s$')
    plt.legend()
    plt.savefig(path_plot)
    plt.show()


def plot_mean_moves(error_rates_eval, mean_moves1,mean_moves2,mean_moves3,mean_moves4,mean_moves5,mean_moves6,mean_moves_curr):

    plt.figure(figsize=(7,5))

    plt.grid()
    plt.scatter(error_rates_eval, mean_moves1, label=r'd=5 PPO, $p_{error}=0.01$', marker='s', s=15)
    plt.plot(error_rates_eval, mean_moves1, linestyle='-',linewidth=1.1)
    #plt.scatter(error_rates_eval, mean_moves2, label=r'd=5 PPO, $p_{error}=0.038$', marker='s', s=15)
    #plt.plot(error_rates_eval, mean_moves2, linestyle='-',linewidth=1.1)
    #plt.scatter(error_rates_eval, mean_moves3, label=r'd=5 PPO, $p_{error}=0.066$', marker='s', s=15)
    #plt.plot(error_rates_eval, mean_moves3,  linestyle='-',linewidth=1.1)
    #plt.scatter(error_rates_eval, mean_moves4, label=r'd=5 PPO, $p_{error}=0.094$', marker='s', s=15)
    #plt.plot(error_rates_eval, mean_moves4, linestyle='-',linewidth=1.1)
    #plt.scatter(error_rates_eval, mean_moves5, label=r'd=5 PPO, $p_{error}=0.129$', marker='s', s=15)
    #plt.plot(error_rates_eval, mean_moves5, linestyle='-',linewidth=1.1)
    plt.scatter(error_rates_eval, mean_moves6, label=r'd=5 PPO, $p_{error}=0.15$', marker='s', s=15)
    plt.plot(error_rates_eval, mean_moves6, linestyle='-',linewidth=1.1)
    plt.plot(error_rates_eval, mean_moves_curr, linestyle='-',linewidth=1.3, color = 'fuchsia')
    plt.scatter(error_rates_eval, mean_moves_curr, label=f"d=5 PPO, curriculum learning", marker="o", s=40, color = 'fuchsia', edgecolors='black',zorder=6)
    plt.xlabel(r'$p_{error}$')
    plt.xlim((0,error_rates_eval[-1]+0.005))

    #plt.title(r'Toric Code - PPO vs MWPM')
    plt.ylabel("Mean number of moves per game")
    plt.legend()
    plt.savefig(path_plot)
    plt.show()

error_rates_eval=list(np.linspace(0.01,0.15,10))


success_rates_all_3 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=3error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.1.csv")
success_rates_all_MWPM_3 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_MWPM/new_success_rates_ppo_board_size=3error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.1.csv")
success_rates_all_5 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.1.csv")
success_rates_all_MWPM_5 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_MWPM/new_success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.1.csv")
success_rates_all_7 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=7error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.1.csv")
success_rates_all_MWPM_7 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_MWPM/new_success_rates_ppo_board_size=7error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.1.csv")
success_rates_all_MWPM_15 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_MWPM/new_success_rates_ppo_board_size=15error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.15.csv")
#success_rates_all_MWPM_30 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_MWPM/new_success_rates_ppo_board_size=30error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.15.csv")

success_rates_err1=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.01.csv")
success_rates_err2=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.038.csv")
success_rates_err3=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.06599999999999999.csv")
success_rates_err4=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.09399999999999999.csv")
success_rates_err5=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.12199999999999998.csv")
success_rates_err6=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.15.csv")

success_rates_err6_5mil=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=5000000mask_actions=Truelambda=1fixed=FalseN=5ent_coef=0.05clip_range=0.1_0.15.csv")
success_rates_err6_5mil_3layers=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=5000000mask_actions=Truelambda=1fixed=FalseN=6ent_coef=0.05clip_range=0.1_0.15.csv")
success_rates_err6_5mil_3layers20 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=5000000mask_actions=Truelambda=1fixed=FalseN=8ent_coef=0.05clip_range=0.1_0.15.csv")




path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_vs_MWPM_3_vs_5_new_test.pdf"
#path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_fixed_diff_timesteps.pdf"
#path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_fixed_curr_mean_moves_5.pdf"

mean_moves1=[1.4429,3.0808,5.7376,9.0,13.277,18.0686,21.8143,25.5623,31.7626,33.9271]
mean_moves2=[]
mean_moves3=[]
mean_moves4=[]
mean_moves5=[]
mean_moves6=[1.3517,2.4313,4.0058,5.8997,8.2986,10.0588,13.7289,15.9117,18.3723,21.1268]
mean_moves_curr=[1.3648,2.2,3.4484,5.5688,7.9054,10.942,12.9308,15.2024,17.1617,18.8845]

#plot_mean_moves(error_rates_eval, mean_moves1,mean_moves2,mean_moves3,mean_moves4,mean_moves5,mean_moves6,mean_moves_curr)
plot_benchmark_MWPM(success_rates_all_3, success_rates_all_MWPM_3, success_rates_all_5, success_rates_all_MWPM_5,success_rates_all_7, success_rates_all_MWPM_7, success_rates_all_MWPM_15,error_rates_eval)

#plot_different_error_rates(success_rates_all_5, success_rates_all_MWPM_5, success_rates_err1, success_rates_err2,success_rates_err3,success_rates_err4,success_rates_err5,success_rates_err6,error_rates_eval)#  success_rates_err4, success_rates_err5, success_rates_err6,error_rates_eval)
#plot_different_timesteps(success_rates_all_5, success_rates_all_MWPM_5, success_rates_err6,success_rates_err6_5mil,success_rates_err6_5mil_3layers,success_rates_err6_5mil_3layers20,error_rates_eval) 
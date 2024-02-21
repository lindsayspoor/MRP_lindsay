import numpy as np
import matplotlib.pyplot as plt

def plot_benchmark_MWPM(success_rates_all_3, success_rates_all_MWPM_3, success_rates_all_5, success_rates_all_MWPM_5, success_rates_all_7, success_rates_all_MWPM_7,success_rates_all_MWPM_15,error_rates_eval):
    

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

def plot_different_error_rates(success_rates_all_5, success_rates_all_MWPM_5, success_rates_err1, success_rates_err2,success_rates_err3,success_rates_err4,success_rates_err5,success_rates_err6,success_rates_curr,error_rates_eval):#  success_rates_err4, success_rates_err5, success_rates_err6,error_rates_eval):
    

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

def plot_different_timesteps(success_rates_all_MWPM_5, success_rates_err_64_64,success_rates_err_64_64_64,success_rates_err_20_20_20,error_rates_eval):#  success_rates_err4, success_rates_err5, success_rates_err6,error_rates_eval):
    

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

def plot_different_lr(success_rates_all_MWPM_5, success_rates_lr0001,success_rates_lr001,success_rates_lr01,error_rates_eval):#  success_rates_err4, success_rates_err5, success_rates_err6,error_rates_eval):
    

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


def plot_mean_moves(error_rates_eval, mean_moves1,mean_moves2,mean_moves3,mean_moves4,mean_moves5,mean_moves6,mean_moves_curr):

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

error_rates_eval=list(np.linspace(0.01,0.15,10))


success_rates_all_3 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=3error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.01clip_range=0.1_0.1.csv")
success_rates_all_MWPM_3 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_MWPM/new_success_rates_ppo_board_size=3error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.01clip_range=0.1_0.1.csv")
success_rates_all_5 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.01clip_range=0.1_0.1.csv")
success_rates_all_MWPM_5 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_MWPM/new_success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.01clip_range=0.1_0.1.csv")
success_rates_all_7 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=7error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.01clip_range=0.1_0.1.csv")
success_rates_all_MWPM_7 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_MWPM/new_success_rates_ppo_board_size=7error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.01clip_range=0.1_0.1.csv")
success_rates_all_MWPM_15 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_MWPM/new_success_rates_ppo_board_size=15error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.15.csv")
#success_rates_all_MWPM_30 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_MWPM/new_success_rates_ppo_board_size=30error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.15.csv")

success_rates_err1=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.01.csv")
success_rates_err2=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.038.csv")
success_rates_err3=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.06599999999999999.csv")
success_rates_err4=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.09399999999999999.csv")
success_rates_err5=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.12199999999999998.csv")
success_rates_err6=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.15.csv")
success_rates_curr = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=1000000mask_actions=Truelambda=1fixed=FalseN=2ent_coef=0.05clip_range=0.1_0.15.csv")

success_rates_err6_5mil=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=5000000mask_actions=Truelambda=1fixed=FalseN=5ent_coef=0.05clip_range=0.1_0.15.csv")
success_rates_err6_5mil_3layers=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=5000000mask_actions=Truelambda=1fixed=FalseN=6ent_coef=0.05clip_range=0.1_0.15.csv")
success_rates_err6_5mil_3layers20 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=5000000mask_actions=Truelambda=1fixed=FalseN=8ent_coef=0.05clip_range=0.1_0.15.csv")

success_rates_err_64_64=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.1.csv")
success_rates_err_64_64_64=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=2ent_coef=0.05clip_range=0.1_0.1.csv")
success_rates_err_20_20_20=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=3ent_coef=0.05clip_range=0.1_0.1.csv")

success_rates_lr0001 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.0001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.05clip_range=0.1_0.1.csv")
success_rates_lr001 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=6ent_coef=0.05clip_range=0.1_0.1.csv")
success_rates_lr01 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.01total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=7ent_coef=0.05clip_range=0.1_0.1.csv")



success_rates_ent05 = np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=9ent_coef=0.05clip_range=0.1_0.1.csv")
success_rates_ent01=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.01clip_range=0.1_0.1.csv")
success_rates_ent0=np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0clip_range=0.1_0.1.csv")

success_rates_r1 =np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=5s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.01clip_range=0.1_0.1.csv")
success_rates_r2 =np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=10s_reward=100c_reward=-10i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.01clip_range=0.1_0.1.csv")
success_rates_r3 =np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=1s_reward=2c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.01clip_range=0.1_0.1.csv")
success_rates_r4 =np.loadtxt("/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/success_rates_agent/success_rates_ppo_board_size=5error_model=0error_rate=0.15l_reward=1s_reward=10c_reward=-1i_reward=-1lr=0.001total_timesteps=300000mask_actions=Truelambda=1fixed=FalseN=1ent_coef=0.01clip_range=0.1_0.1.csv")


#path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_vs_MWPM_3_vs_5_vs_7.pdf"
#path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_fixed_diff_timesteps.pdf"
#path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_fixed_curr_mean_moves_5.pdf"
#path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_fixed_curr.pdf"

#path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_diff_architectures.pdf"
#path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_diff_lr.pdf"
#path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_mean_moves_lr.pdf"
#path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_ent_coef.pdf"

path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_lr.pdf"


mean_moves_curr=[1.1077,1.8743,3.2344,5.1988,7.361,9.5388,12.8946,14.2192,16.5931,18.2564]
mean_moves4=[76.1032,140.5995,172.9777,186.8081,193.653,197.2942,198.9061,199.4632,199.6422,199.8807] #0.01
mean_moves3=[1.4668,2.4101,3.3651,4.2976,5.1622,5.9723,6.5337,7.0481,7.5215,7.8038] #0.0001
mean_moves2=[1.1188,1.9202,3.3821, 4.9203,7.1054,8.7053,11.4492,14.182,15.934,17.7075] #0.001
mean_moves5=[]
mean_moves6=[]
mean_moves1=[1.1663,1.7467,3.3847,5.0532,7.132,9.7838,12.4166,14.3253,16.7758,18.7474]

#plot_mean_moves(error_rates_eval, mean_moves1,mean_moves2,mean_moves3,mean_moves4,mean_moves5,mean_moves6,mean_moves_curr)
#plot_benchmark_MWPM(success_rates_all_3, success_rates_all_MWPM_3, success_rates_all_5, success_rates_all_MWPM_5,success_rates_all_7, success_rates_all_MWPM_7, success_rates_all_MWPM_15,error_rates_eval)

#plot_different_error_rates(success_rates_all_5, success_rates_all_MWPM_5, success_rates_err1, success_rates_err2,success_rates_err3,success_rates_err4,success_rates_err5,success_rates_err6,success_rates_curr,error_rates_eval)#  success_rates_err4, success_rates_err5, success_rates_err6,error_rates_eval)
#plot_different_timesteps(success_rates_all_MWPM_5, success_rates_err_64_64,success_rates_err_64_64_64,success_rates_err_20_20_20,error_rates_eval)
plot_different_lr(success_rates_all_MWPM_5, success_rates_lr0001,success_rates_lr001,success_rates_lr01,error_rates_eval)
#plot_different_lr(success_rates_all_MWPM_5, success_rates_r1,success_rates_r2,success_rates_r3,success_rates_r4,error_rates_eval)
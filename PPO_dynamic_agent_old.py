import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from toric_game_dynamic_env import ToricGameDynamicEnv, ToricGameDynamicEnvFixedErrs
from stable_baselines3.ppo.policies import MlpPolicy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
import os
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from custom_callback import SaveOnBestTrainingRewardCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
from plot_functions_old import plot_benchmark_MWPM, plot_log_results, render_evaluation, plot_single_box_dynamic

os.getcwd()



class PPO_agent:
    def __init__(self, initialisation_settings, log):#path):

        self.initialisation_settings=initialisation_settings
        # Create log dir
        self.log=log
        if self.log:
            self.log_dir = "log_dirs/log_dir_dynamic6"
            os.makedirs(self.log_dir, exist_ok=True)



        #INITIALISE MODEL FOR INITIALISATION
        self.initialise_model()

    def initialise_model(self):
        #INITIALISE ENVIRONMENT INITIALISATION
        print("initialising the environment and model...")
        if self.initialisation_settings['fixed']:
            self.env = ToricGameDynamicEnvFixedErrs(self.initialisation_settings)
        else:
            self.env = ToricGameDynamicEnv(self.initialisation_settings)


        # Logs will be saved in log_dir/monitor.csv
        if self.log:
            self.env = Monitor(self.env, self.log_dir)
            # Create the callback: check every ... steps
            self.callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=self.log_dir)
            
        #INITIALISE MODEL FOR INITIALISATION
        if self.initialisation_settings['mask_actions']:
            ppo = MaskablePPO
            policy = MaskableActorCriticPolicy
        else:
            ppo= PPO
            policy = MlpPolicy
        lr = self.initialisation_settings['lr']
        if lr == "annealing":
            lr = learning_rate_annealing
        self.model = ppo(policy, self.env, ent_coef=self.initialisation_settings['ent_coef'], clip_range = self.initialisation_settings['clip_range'],learning_rate=lr, verbose=0, n_steps=self.initialisation_settings['n_steps'], policy_kwargs={"net_arch":dict(pi=[64,64], vf=[64,64])})

        print("initialisation done")
        print(self.model.policy)

    def change_environment_settings(self, settings):
        print("changing environment settings...")
        if settings['fixed']:
            self.env = ToricGameDynamicEnvFixedErrs(settings)
        else:
            self.env = ToricGameDynamicEnv(settings)

        # Logs will be saved in log_dir/monitor.csv
        if self.log:
            self.env = Monitor(self.env, self.log_dir, override_existing=False)
            #self.env = Monitor(self.env, self.log_dir)
            # Create the callback: check every 1000 steps
            self.callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=self.log_dir)
        
        self.model.set_env(self.env)

        print("changing settings done")

    def train_model(self, save_model_path):
        print("training the model...")
        if self.log:
            self.model.learn(total_timesteps=self.initialisation_settings['total_timesteps'], progress_bar=True, callback=self.callback)
            plot_log_results(self.log_dir ,save_model_path)
        else:
            self.model.learn(total_timesteps=self.initialisation_settings['total_timesteps'], progress_bar=True)
    
        self.model.save(f"trained_models/dynamic_ppo_{save_model_path}")
        print("training done")

    def load_model(self, load_model_path):
        print("loading the model...")

        if self.initialisation_settings['mask_actions']:
            self.model=MaskablePPO.load(f"trained_models/dynamic_ppo_{load_model_path}")
        else:
            self.model=PPO.load(f"trained_models/dynamic_ppo_{load_model_path}")
        print("loading done")
    



def evaluate_model(agent, evaluation_settings, render, number_evaluations, max_moves):
    print("evaluating the model...")
    moves=0

    actions=np.zeros((number_evaluations,max_moves,1)) #1st column for agent (3rd dimension)
    actions[:,:,:]=np.nan
    reward_agent=[]
    moves_agent = []


    for k in tqdm(range(number_evaluations)):
        rewards=0
        moves=0
        obs, info = agent.env.reset()

        if render:
            agent.env.render()
        for i in range(max_moves):
            #if i == (max_moves-1):
                #print("max moves/max reward reached")
            if evaluation_settings['mask_actions']:
                action_masks=get_action_masks(agent.env)

                action, _state = agent.model.predict(obs, action_masks=action_masks)

            else:
                action, _state = agent.model.predict(obs)
            #print(f"{action=}")
            obs, reward, done, truncated, info = agent.env.step(action)#, without_illegal_actions=True)

            actions[k,i,0]=action
            moves+=1
            rewards+=reward
            if render:
                print(info['message'])
                agent.env.render()
            if done or truncated:
                break

        reward_agent.append(rewards)
        moves_agent.append(moves)


    mean_reward=np.mean(reward_agent)
    print(f"mean reward per evaluation is {mean_reward}")
                
        
    print(f"mean number of moves per evaluation is {moves/number_evaluations}")
    

    print("evaluation done")

    return mean_reward, reward_agent, moves_agent,actions



def evaluate_fixed_errors(agent, evaluation_settings, render, number_evaluations, max_moves,save_files):
    


    mean_reward_agent, reward_agent, moves_agent,actions = evaluate_model(agent,evaluation_settings, render, number_evaluations, max_moves)

    evaluation_path =''
    for key, value in evaluation_settings.items():
        evaluation_path+=f"{key}={value}"

    if save_files:
        folder = "/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/dynamic_ppo"
        if fixed:
            np.savetxt(f"{folder}/rewards_dynamic_agent/rewards_dynamic_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", reward_agent)
            np.savetxt(f"{folder}/moves_dynamic_agent/moves_dynamic_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", moves_agent)

        else:
            np.savetxt(f"{folder}/rewards_dynamic_agent/rewards_dynamic_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", reward_agent)
            np.savetxt(f"{folder}/moves_dynamic_agent/moves_dynamic_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", moves_agent)



    return mean_reward_agent, reward_agent, moves_agent,actions


def evaluate_error_rates(agent,evaluation_settings, render, number_evaluations, max_moves,  save_files, fixed):
    


    mean_reward_agent, reward_agent, moves_agent,actions = evaluate_model(agent,evaluation_settings, render, number_evaluations, max_moves)

    evaluation_path =''
    for key, value in evaluation_settings.items():
        evaluation_path+=f"{key}={value}"


    if save_files:
        folder = "/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Files_results/dynamic_ppo"
        if fixed:
            np.savetxt(f"{folder}/rewards_dynamic_agent/rewards_dynamic_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", reward_agent)
            np.savetxt(f"{folder}/moves_dynamic_agent/moves_dynamic_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", moves_agent)

        else:
            np.savetxt(f"{folder}/rewards_dynamic_agent/rewards_dynamic_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", reward_agent)
            np.savetxt(f"{folder}/moves_dynamic_agent/moves_dynamic_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", moves_agent)



    return mean_reward_agent,reward_agent,moves_agent, actions



def learning_rate_annealing(value):
    begin=0.001
    end=0.0001
    return begin*value+end



#SETTINGS FOR RUNNING THIS SCRIPT

train=False
curriculum=False #if set to True the agent will train on N_curriculum or error_rate_curriculum examples, using the training experience from 
benchmark_MWPM=False
save_files=True
render=False
number_evaluations=10000
max_moves=300
evaluate=True


board_size=3
error_rate=0.01
ent_coef=0.05
clip_range=0.1
total_timesteps=3000000
n_steps=2048
mask_actions=True #if set to True action masking is enabled, the illegal actions are masked out by the model. If set to False the agent gets a reward 'illegal_action_reward' when choosing an illegal action.
log = True #if set to True the learning curve during training is registered and saved.
#learning_rate=0.001
learning_rate = "annealing"
continue_reward=1
empty_reward=10
logical_error_reward=1

N=1 #the number of fixed initinal flips N the agent model is trained on or loaded when fixed is set to True
new_N=1
iteration_step=2

fixed=True #if set to True the agent is trained on training examples with a fixed amount of N initial errors. If set to False the agent is trained on training examples given an error rate error_rate for each qubit to have a chance to be flipped.
evaluate_fixed=True #if set to True the trained model is evaluated on examples with a fixed amount of N initial errors. If set to False the trained model is evaluated on examples in which each qubit is flipped with a chance of error_rate.





#SET SETTINGS TO INITIALISE AGENT ON
initialisation_settings = {'board_size': board_size,
            'error_rate': error_rate,
            'lr':learning_rate,
            'total_timesteps': total_timesteps,
            'n_steps':n_steps,
            'mask_actions': mask_actions,
            'fixed':fixed,
            'c_reward':continue_reward,
            'e_reward':empty_reward,
            'l_reward':logical_error_reward,
            'N':N,#,
            'iteration_step': iteration_step,
            'ent_coef':ent_coef,
            'clip_range':clip_range,
            'new_N':new_N
            }

#SET SETTINGS TO LOAD TRAINED AGENT ON
loaded_model_settings = {'board_size': board_size,
            'error_rate': error_rate,
            'lr':learning_rate,
            'total_timesteps': total_timesteps,
            'n_steps':n_steps,
            'mask_actions': mask_actions,
            'fixed':fixed,
            'c_reward':continue_reward,
            'e_reward':empty_reward,
            'l_reward':logical_error_reward,
            'N':N,
            'iteration_step': iteration_step,
            'ent_coef':ent_coef,
            'clip_range':clip_range,
            'new_N':new_N
            }

evaluation_settings = {'board_size': board_size,
            'error_rate': error_rate,
            'lr':learning_rate,
            'total_timesteps': total_timesteps,
            'n_steps':n_steps,
            'mask_actions': mask_actions,
            'fixed':fixed,
            'c_reward':continue_reward,
            'e_reward':empty_reward,
            'l_reward':logical_error_reward,
            'N':N,
            'iteration_step': iteration_step,
            'ent_coef':ent_coef,
            'clip_range':clip_range,
            'new_N':new_N
            }




save_model_path =''
for key, value in initialisation_settings.items():
    save_model_path+=f"{key}={value}"


load_model_path =''
for key, value in loaded_model_settings.items():
    load_model_path+=f"{key}={value}"




#initialise PPO Agent
AgentPPO = PPO_agent(initialisation_settings, log)


if train:
    AgentPPO.train_model(save_model_path=save_model_path)
else:
    print(f"{loaded_model_settings['N']=}")
    print(f"{loaded_model_settings['error_rate']=}")
    AgentPPO.load_model(load_model_path=load_model_path)



        

if evaluate:

    if evaluate_fixed:
        mean_reward, rewards_agent, moves_agent,actions = evaluate_fixed_errors(AgentPPO, evaluation_settings,  render, number_evaluations, max_moves,  save_files)
    else:
        mean_reward, rewards_agent, moves_agent,actions = evaluate_error_rates(AgentPPO, evaluation_settings,  render, number_evaluations, max_moves,  save_files, fixed)





evaluation_path =''
for key, value in evaluation_settings.items():
    evaluation_path+=f"{key}={value}"


if fixed:
    path = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/boxplot_dynamic_ppo_{evaluation_path}_{loaded_model_settings['N']}.pdf"
else:
    path = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/boxplot_dynamic_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.pdf"


plot_single_box_dynamic(path, rewards_agent)



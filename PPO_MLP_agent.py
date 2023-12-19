import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from toric_game_env import ToricGameEnv, ToricGameEnvFixedErrs
from config import ErrorModel
from stable_baselines3.ppo.policies import MlpPolicy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
import os
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from callback_class import SaveOnBestTrainingRewardCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.policies import obs_as_tensor
import networkx as nx
from tqdm import tqdm
from plot_functions import plot_benchmark_MWPM, plot_log_results, render_evaluation
from MWPM_decoder import decode_MWPM_method

os.getcwd()






class PPO_agent:
    def __init__(self, initialisation_settings, log):#path):

        self.initialisation_settings=initialisation_settings
        # Create log dir
        self.log=log
        if self.log:
            self.log_dir = "log_dir"
            os.makedirs(self.log_dir, exist_ok=True)



        #INITIALISE MODEL FOR INITIALISATION
        self.initialise_model()

    def initialise_model(self):
        #INITIALISE ENVIRONMENT INITIALISATION
        print("initialising the environment and model...")
        if self.initialisation_settings['fixed']:
            self.env = ToricGameEnvFixedErrs(self.initialisation_settings)
        else:
            self.env = ToricGameEnv(self.initialisation_settings)


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
        
        self.model = ppo(policy, self.env, ent_coef=self.initialisation_settings['ent_coef'], clip_range = self.initialisation_settings['clip_range'],learning_rate=self.initialisation_settings['lr'], verbose=0, policy_kwargs={"net_arch":dict(pi=[64,64], vf=[64,64])})

        print("initialisation done")
        print(self.model.policy)

    def change_environment_settings(self, settings):
        print("changing environment settings...")
        if settings['fixed']:
            self.env = ToricGameEnvFixedErrs(settings)
        else:
            self.env = ToricGameEnv(settings)

        # Logs will be saved in log_dir/monitor.csv
        if self.log:
            self.env = Monitor(self.env, self.log_dir)
            # Create the callback: check every 1000 steps
            self.callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=self.log_dir)
        
        self.model.set_env(self.env)

        print("changing settings done")

    def train_model(self, save_model_path):
        print("training the model...")
        if self.log:
            self.model.learn(total_timesteps=self.initialisation_settings['total_timesteps'], progress_bar=True, callback=self.callback)
            plot_log_results(self.log_dir, save_model_path)
        else:
            self.model.learn(total_timesteps=self.initialisation_settings['total_timesteps'], progress_bar=True)
    
        self.model.save(f"trained_models/ppo_{save_model_path}")
        print("training done")

    def load_model(self, load_model_path):
        print("loading the model...")
        #self.model=PPO.load(f"trained_models/ppo_{load_model_path}")
        if self.initialisation_settings['mask_actions']:
            self.model=MaskablePPO.load(f"trained_models/ppo_{load_model_path}")
        else:
            self.model=PPO.load(f"trained_models/ppo_{load_model_path}")
        print("loading done")
    




    def evaluate_model(self, evaluation_settings, render, number_evaluations, max_moves, check_fails):

        print("evaluating the model...")

        moves=0
        logical_errors=0
        success=0
        success_MWPM=0
        logical_errors_MWPM=0

        observations=np.zeros((number_evaluations, evaluation_settings['board_size']*evaluation_settings['board_size']))
        results=np.zeros((number_evaluations,2)) #1st column for agent, 2nd column for MWPM decoder
        actions=np.zeros((number_evaluations,max_moves,2)) #1st column for agent, 2nd column for MWPM decoder (3rd dimension)
        actions[:,:,:]=np.nan
        

        for k in tqdm(range(number_evaluations)):
            #print("new evaluation")
            obs, info = self.env.reset()
            initial_flips = AgentPPO.env.initial_qubits_flips

            obs0=obs.copy()
            observations[k,:]=obs
            obs0_k=obs0.reshape((evaluation_settings['board_size'],evaluation_settings['board_size']))

            MWPM_check, MWPM_actions = decode_MWPM_method(self.env.state.qubit_pos,obs0_k, initial_flips, evaluation_settings)

            actions[k,:MWPM_actions.shape[0],1] = MWPM_actions[:,0]

            if MWPM_check==True:
                success_MWPM+=1
                results[k,1]=1 #1 for success
            if MWPM_check==False:
                logical_errors_MWPM+=1
                results[k,1]=0 #0 for fail
            #self.env.render()
            #if render:
                #self.env.render()
            for i in range(max_moves):
                if evaluation_settings['mask_actions']:
                    action_masks=get_action_masks(self.env)

                    action, _state = self.model.predict(obs, action_masks=action_masks, deterministic=True)

                else:
                    action, _state = self.model.predict(obs)
                obs, reward, done, truncated, info = self.env.step(action)#, without_illegal_actions=True)
                actions[k,i,0]=action
                moves+=1
                if render:
                    self.env.render()
                if done:
                    if info['message']=='logical_error':
                        if check_fails:
                            if results[k,0]==0 and results[k,1]==1:
                                
                                print(info['message'])
                                #print(f"{action_masks=}")
                                prob_dist = ((self.model.policy.get_distribution(self.model.policy.obs_to_tensor(obs0)[0]).distribution).probs).detach().numpy()
                                #print(f"{prob_dist=}")
                                #print(f"{np.max(prob_dist)=}")
                                #print(f"{np.argwhere(prob_dist==np.max(prob_dist))=}")
                                render_evaluation(obs0_k,evaluation_settings, actions[k,:,:], initial_flips)
                        logical_errors+=1
                        results[k,0]=0 #0 for fail
                    if info['message'] == 'success':
                        success+=1
                        results[k,0]=1 #1 for success
                        #self.env.render()
                        #self.env.reset()
                    break



                    
            
        print(f"mean number of moves per evaluation is {moves/number_evaluations}")
        
        if (success+logical_errors)==0:
            success_rate = 0
        else:
            success_rate= success / (success+logical_errors)

        if (success_MWPM+logical_errors_MWPM)==0:
            success_rate_MWPM = 0
        else:
            success_rate_MWPM= success_MWPM / (success_MWPM+logical_errors_MWPM)

        print("evaluation done")

        return success_rate, success_rate_MWPM, observations, results, actions



    def evaluate_fixed_errors(self, evaluation_settings, N_evaluates, render, number_evaluations, max_moves, check_fails, save_files):
        
        success_rates=[]
        success_rates_MWPM=[]
        observations_all=[]

        for N_evaluate in N_evaluates:
            print(f"{N_evaluate=}")
            evaluation_settings['fixed'] = evaluate_fixed
            evaluation_settings['N']=N_evaluate
            evaluation_settings['success_reward']=evaluation_settings['N']
            self.change_environment_settings(evaluation_settings)
            success_rate, success_rate_MWPM, observations, results, actions = self.evaluate_model(evaluation_settings, render, number_evaluations, max_moves, check_fails)
            success_rates.append(success_rate)
            success_rates_MWPM.append(success_rate_MWPM)
            observations_all.append(observations)
            print(f"{success_rate=}")
            print(f"{success_rate_MWPM=}")



        success_rates=np.array(success_rates)
        success_rates_MWPM=np.array(success_rates_MWPM)
        observations_all=np.array(observations_all)
        print(f"{observations_all.shape=}")


        evaluation_path =''
        for key, value in evaluation_settings.items():
            evaluation_path+=f"{key}={value}"

        if save_files:
            if fixed:
                np.savetxt(f"Files_results/success_rates_agent/success_rates_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", success_rates)
                np.savetxt(f"Files_results/success_rates_MWPM/success_rates_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", success_rates_MWPM)
                np.savetxt(f"Files_results/observations/observations_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", observations)
                np.savetxt(f"Files_results/results_agent_MWPM/results_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", results)
                np.savetxt(f"Files_results/actions_agent_MWPM/actions_agent_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", actions[:,:,0])
                np.savetxt(f"Files_results/actions_agent_MWPM/actions_MWPM_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", actions[:,:,1])
            else:
                np.savetxt(f"Files_results/success_rates_agent/success_rates_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", success_rates)
                np.savetxt(f"Files_results/success_rates_MWPM/success_rates_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", success_rates_MWPM)
                np.savetxt(f"Files_results/observations/observations_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", observations)
                np.savetxt(f"Files_results/results_agent_MWPM/results_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", results)
                np.savetxt(f"Files_results/actions_agent_MWPM/actions_agent_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", actions[:,:,0])
                np.savetxt(f"Files_results/actions_agent_MWPM/actions_MWPM_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", actions[:,:,1])

        return success_rates, success_rates_MWPM,observations, results, actions
    

    def evaluate_error_rates(self,evaluation_settings, error_rates, render, number_evaluations, max_moves, check_fails, save_files, fixed):
        success_rates=[]
        success_rates_MWPM=[]
        observations_all=[]

        for error_rate in error_rates:
            #SET SETTINGS TO EVALUATE LOADED AGENT ON
            print(f"{error_rate=}")
            evaluation_settings['error_rate'] = error_rate
            evaluation_settings['fixed'] = evaluate_fixed

            self.change_environment_settings(evaluation_settings)
            success_rate, success_rate_MWPM, observations, results, actions = self.evaluate_model(evaluation_settings, render, number_evaluations, max_moves, check_fails)
            success_rates.append(success_rate)
            success_rates_MWPM.append(success_rate_MWPM)
            observations_all.append(observations)
            print(f"{success_rate=}")
            print(f"{success_rate_MWPM=}")



        success_rates=np.array(success_rates)
        success_rates_MWPM=np.array(success_rates_MWPM)
        observations_all=np.array(observations_all)



        evaluation_path =''
        for key, value in evaluation_settings.items():
            evaluation_path+=f"{key}={value}"

        if save_files:
            if fixed:
                np.savetxt(f"Files_results/success_rates_agent/success_rates_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", success_rates)
                np.savetxt(f"Files_results/success_rates_MWPM/success_rates_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", success_rates_MWPM)
                np.savetxt(f"Files_results/observations/observations_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", observations)
                np.savetxt(f"Files_results/results_agent_MWPM/results_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", results)
                np.savetxt(f"Files_results/actions_agent_MWPM/actions_agent_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", actions[:,:,0])
                np.savetxt(f"Files_results/actions_agent_MWPM/actions_MWPM_ppo_{evaluation_path}_{loaded_model_settings['N']}.csv", actions[:,:,1])
            else:
                np.savetxt(f"Files_results/success_rates_agent/success_rates_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", success_rates)
                np.savetxt(f"Files_results/success_rates_MWPM/success_rates_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", success_rates_MWPM)
                np.savetxt(f"Files_results/observations/observations_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", observations)
                np.savetxt(f"Files_results/results_agent_MWPM/results_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", results)
                np.savetxt(f"Files_results/actions_agent_MWPM/actions_agent_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", actions[:,:,0])
                np.savetxt(f"Files_results/actions_agent_MWPM/actions_MWPM_ppo_{evaluation_path}_{loaded_model_settings['error_rate']}.csv", actions[:,:,1])

        return success_rates, success_rates_MWPM,observations, results, actions





#SETTINGS FOR RUNNING THIS SCRIPT
train=False
curriculum=False #if set to True the agent will train on N_curriculum or error_rate_curriculum examples, using the training experience from 
benchmark_MWPM=False
save_files=True#
render=False
number_evaluations=10000
max_moves=200
evaluate=True
check_fails=False

board_size=5
error_rate=0.01
ent_coef=0.05
clip_range=0.1
N=1 #the number of fixed initinal flips N the agent model is trained on or loaded when fixed is set to True
logical_error_reward=5 #the reward the agent gets when it has removed all syndrome points, but the terminal board state claims that there is a logical error.
success_reward=10 #the reward the agent gets when it has removed all syndrome points, and the terminal board state claims that there is no logical error, ans therefore the agent has successfully done its job.
continue_reward=-1 #the reward the agent gets for each action that does not result in the terminal board state. If negative it gets penalized for each move it does, therefore giving the agent an incentive to remove syndromes in as less moves as possible.
illegal_action_reward=-1 #the reward the agent gets when mask_actions is set to False and therefore the agent gets penalized by choosing an illegal action.
total_timesteps=100000
learning_rate= 0.001
mask_actions=True #if set to True action masking is enabled, the illegal actions are masked out by the model. If set to False the agent gets a reward 'illegal_action_reward' when choosing an illegal action.
log = True #if set to True the learning curve during training is registered and saved.
lambda_value=1
fixed=True #if set to True the agent is trained on training examples with a fixed amount of N initial errors. If set to False the agent is trained on training examples given an error rate error_rate for each qubit to have a chance to be flipped.
evaluate_fixed=True #if set to True the trained model is evaluated on examples with a fixed amount of N initial errors. If set to False the trained model is evaluated on examples in which each qubit is flipped with a chance of error_rate.
#N_evaluates = [1,2,3,4,5] #the number of fixed initial flips N the agent is evaluated on if evaluate_fixed is set to True.
N_evaluates=[3]
error_rates_eval=list(np.linspace(0.01,0.1,4))
N_curriculums=[1,2,3]
#N_curriculums=[5]

error_rates_curriculum=list(np.linspace(0.01,0.1,4))
#error_rates_curriculum=[0.1]

#SET SETTINGS TO INITIALISE AGENT ON
initialisation_settings = {'board_size': board_size,
            'error_model': ErrorModel['UNCORRELATED'],
            'error_rate': error_rate,
            'l_reward': logical_error_reward,
            's_reward': success_reward,
            'c_reward':continue_reward,
            'i_reward':illegal_action_reward,
            'lr':learning_rate,
            'total_timesteps': total_timesteps,
            'mask_actions': mask_actions,
            'lambda': lambda_value,
            'fixed':fixed,
            'N':N,
            'ent_coef':ent_coef,
            'clip_range':clip_range
            }

#SET SETTINGS TO LOAD TRAINED AGENT ON
loaded_model_settings = {'board_size': board_size,
            'error_model': ErrorModel['UNCORRELATED'],
            'error_rate': error_rate,
            'l_reward': logical_error_reward,
            's_reward': success_reward,
            'c_reward':continue_reward,
            'i_reward':illegal_action_reward,
            'lr':learning_rate,
            'total_timesteps': total_timesteps,
            'mask_actions': mask_actions,
            'lambda': lambda_value,
            'fixed':fixed,
            'N':N,
            'ent_coef':ent_coef,
            'clip_range':clip_range
            }

evaluation_settings = {'board_size': board_size,
            'error_model': ErrorModel['UNCORRELATED'],
            'error_rate': error_rate,
            'l_reward': logical_error_reward,
            's_reward': success_reward,
            'c_reward':continue_reward,
            'i_reward':illegal_action_reward,
            'lr':learning_rate,
            'total_timesteps': total_timesteps,
            'mask_actions': mask_actions,
            'lambda': lambda_value,
            'fixed':fixed,
            'N':N,
            'ent_coef':ent_coef,
            'clip_range':clip_range
            }



success_rates_all=[]
success_rates_all_MWPM=[]



if fixed:
    curriculums=N_curriculums
else:
    curriculums=error_rates_curriculum



for curriculum_val in curriculums:

    if (train==True) and (curriculum == False) and(curriculums.index(curriculum_val)>0):
        train=False
        curriculum=True


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
        AgentPPO.load_model(load_model_path=load_model_path)
        
    initialisation_settings['total_timesteps']=6000000
    loaded_model_settings['total_timesteps']=6000000
    if curriculum:
        if fixed:
        
            print(f"{curriculum_val=}")
            initialisation_settings['N']=curriculum_val
        else:
            print(f"{curriculum_val=}")
            initialisation_settings['error_rate']=curriculum_val



        save_model_path =''
        for key, value in initialisation_settings.items():
            save_model_path+=f"{key}={value}"

        AgentPPO.change_environment_settings(initialisation_settings)

        AgentPPO.train_model(save_model_path=save_model_path)
        
        if fixed:
            loaded_model_settings['N']=curriculum_val
        else:
            loaded_model_settings['error_rate']=curriculum_val



    if evaluate:

        if evaluate_fixed:
            success_rates, success_rates_MWPM,observations, results, actions = AgentPPO.evaluate_fixed_errors(evaluation_settings, N_evaluates, render, number_evaluations, max_moves, check_fails, save_files)
        else:
            success_rates, success_rates_MWPM,observations, results, actions = AgentPPO.evaluate_error_rates(evaluation_settings, error_rates_eval, render, number_evaluations, max_moves, check_fails, save_files, fixed)


        success_rates_all.append(success_rates)
        success_rates_all_MWPM.append(success_rates_MWPM)



evaluation_path =''
for key, value in evaluation_settings.items():
    evaluation_path+=f"{key}={value}"



success_rates_all=np.array(success_rates_all)
success_rates_all_MWPM=np.array(success_rates_all_MWPM)




if fixed:
    path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_vs_MWPM_{evaluation_path}_{loaded_model_settings['N']}.pdf"
else:
    path_plot = f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project/Results/Figure_results/Results_benchmarks/PPO_vs_MWPM_{evaluation_path}_{loaded_model_settings['error_rate']}.pdf"


plot_benchmark_MWPM(success_rates_all, success_rates_all_MWPM, N_evaluates, error_rates_eval, board_size,path_plot,loaded_model_settings['N'], loaded_model_settings['error_rate'],evaluate_fixed)
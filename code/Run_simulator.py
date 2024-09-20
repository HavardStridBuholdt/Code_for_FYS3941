#%% --------------------------------------------- Importing Packages --------------------------------------------
import gymnasium as gym
from gymnasium.envs.registration import register
from datetime import datetime
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple



# Simglucose objects
from simglucose.simulation.user_interface import simulate
# from simglucose.controller.base import Controller, Action
# from simglucose.simulation.scenario import CustomScenario
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from controllers.my_basal_bolus_ctrller import BBController

import sys

Action = namedtuple("simulation_ctrller_action", ["basal", "bolus", "CHO", "GI"])


class Simulator():
    def __init__(
        self,
        simulator,
        patient_name,
        start_time,
        meal_plan=None,
        ex_start=None,
        ex_end=None,
        HR=None,
        HRb=None,
        beta=None,
        seed=None,
        random_init_bg=True,
        basal_regime=False,
        bolus_regime=False,
        BG_target=140,
        end_time=None,
        record=False,
        to_record_model_eq=False,                # Temporary test parameter
        testing_GI_limits=False,                 # Temporary test parameter
    ):
        self.simulator = simulator
        self.patient_name = patient_name
        self.start_time = start_time
        self.ex_start = ex_start
        self.ex_end = ex_end
        self.basal_regime = basal_regime
        self.bolus_regime = bolus_regime
        self.BBctrller = BBController(target=BG_target)
        self.to_record_model_eq = to_record_model_eq
        self.record = record
        if self.record:
            self.CGM_recording = []
            self.BG_recording = []
        
        implemented_PA_models = ["Breton", "PA"]
        if not simulator in implemented_PA_models:
            raise ValueError(f"Physical Activity model {simulator} is not supported by this simulator. Try one of these models instead {implemented_PA_models}")
        # sys.path.insert(0, "C:/Users/hsb19/OneDrive/Dokumenter/UiT/Semester/V2024/FYS-3941/Simulator/modified_files/"+simulator+"/")
        if simulator == "Breton":
            self.env = make_Breton_environment(
                patiant_name=patient_name,
                start_time=start_time,
                end_time=end_time,
                meal_plan=meal_plan,
                ex_start=ex_start,
                ex_end=ex_end,
                HR=HR,
                HRb=HRb,
                seed=seed,
                random_init_bg=random_init_bg,
                to_record_model_eq=to_record_model_eq,                  # Temporary test parameter
                testing_GI_limits=testing_GI_limits,                    # Temporary test parameter
            )
        elif simulator == "PA":
            self.env = make_PA_environment(
                patiant_name=patient_name,
                start_time=start_time,
                end_time=end_time,
                meal_plan=meal_plan,
                ex_start=ex_start,
                ex_end=ex_end,
                HR=HR,
                HRb=HRb,
                beta=beta,
                seed=seed,
                random_init_bg=random_init_bg,
                to_record_model_eq=to_record_model_eq,                  # Temporary test parameter
                testing_GI_limits=testing_GI_limits,                    # Temporary test parameter
            )
        else:
            raise ValueError(f"Physical Activity model {simulator} is not supported by the currant version of the simulator")
    
    def run(self, N_episodes, render=False):
        # Episode Loop
        for episode in range(N_episodes):
            # print("\n")
            # print(f"############################### Episode [{episode+1}/{N_episodes}] ###############################", end="\r")
            observation, info = self.env.reset()
            if self.record:
                recording = [observation[0]]
                BG_recording = [info["bg"]]
            # print(f"Reset environment, observation: {observation}")
            t = 0
            done = False
            reward = 0
            # Time Step Loop
            while not done:
                # print("")
                # print(f'Step {t}, Time {info["time"]}', f"BG: {info['bg']}",  recording)
                if render:
                    self.env.render()
                
                # Choosing controller action:
                if self.basal_regime or self.bolus_regime:
                    basal, bolus = self.BBctrller.policy(observation, reward, done, **info)
                    basal *= self.basal_regime
                    bolus *= self.bolus_regime
                else:
                    basal, bolus = 0, 0
                CHO = 0
                GI = 0

                # Inputted controller action
                action = Action(basal, bolus, CHO, GI)
                
                observation, reward, terminated, truncated, info = self.env.step(action)
                if self.record:
                    recording.append(observation[0])
                    BG_recording.append(info["bg"])
                # print(
                #     f"Step {t}: observation {observation}, reward {reward}, terminated {terminated}, truncated {truncated}, info {info}"
                # )
                if terminated or truncated:
                    # print("Episode finished after {} timesteps".format(t + 1))
                    done = True
                    self.env.close()
                    if episode < N_episodes-1:
                        pass
                        # plt.close()
                
                t += 1
            
            # Save CGM records of currant episode
            if self.record:
                self.CGM_recording.append(recording)
                self.BG_recording.append(BG_recording)
        
        if self.to_record_model_eq:
            plot_model_equations(self.simulator, self.patient_name, self.start_time, self.ex_start, self.ex_end)
            plt.show()





def plot_model_equations(sim, name, start_time, ex_start, ex_end):
    path = sim + "_Recordings.pkl"
    df = pd.read_pickle(path)
    time = df.index.values/60
    
    fig1, axes = plt.subplots(6)
    start = (ex_start - start_time).total_seconds()/60/60
    end = (ex_end - start_time).total_seconds()/60/60
    for ax in axes:
        ax.set_xlabel("Time [Hours]")
        ax.axvspan(start, end, alpha=0.2, color="gray", lw=0)
    
    if sim == "Breton":
        axes[0].set_title(f"Patient: {name}   |   HR(t),  Y(t),  Z(t),  W(t),  f(t),  Uidt   vs.  Time [min]")
        axes[0].set_ylabel("HR(t)")
        axes[1].set_ylabel("Y(t)")
        axes[2].set_ylabel("Z(t)")
        axes[3].set_ylabel("W(t)")
        axes[4].set_ylabel("f(t)")
        axes[5].set_ylabel("Uidt(t)")
        
        axes[0].plot(time, df["HR"].values)
        axes[1].plot(time, df["Y"].values)
        axes[2].plot(time, df["Z"].values)
        axes[3].plot(time, df["W"].values)
        axes[4].plot(time, df["f"].values)
    
    elif sim == "PA":
        axes[0].set_title(f"Patient: {name}   |   uhr(t),  h(t),  theta(t),  w(t),  phi(t),  Uidt   vs.  Time [min]")
        axes[0].set_ylabel("uhr(t)")
        axes[1].set_ylabel("h(t)")
        axes[2].set_ylabel("theta(t)")
        axes[3].set_ylabel("w(t)")
        axes[4].set_ylabel("phi(t)")
        axes[5].set_ylabel("Uidt(t)")
        
        axes[0].plot(time, df["uhr"].values)
        axes[1].plot(time, df["h"].values)
        axes[2].plot(time, df["theta"].values)
        axes[3].plot(time, df["w"].values)
        axes[4].plot(time, df["phi"].values)
    
    axes[5].plot(time, df["Uidt"].values)
    fig1.show()
    # fig.savefig()


def make_Breton_environment(
    patiant_name:str,
    start_time:datetime,
    end_time:datetime,
    meal_plan:list,
    ex_start:datetime,
    ex_end:datetime,
    HR:float,
    HRb:float,
    seed:int,
    random_init_bg:bool,
    to_record_model_eq:bool,                # Temporary test parameter
    testing_GI_limits:bool                  # Temporary test parameter
):
    sys.path.insert(0, "C:/Users/hsb19/OneDrive/Dokumenter/UiT/Semester/V2024/FYS-3941/Simulator/modifiedB_files/")
    from modifiedB_files.my_scenario import CustomScenario
    
    scenario = CustomScenario(
        start_time=start_time,
        scenario=meal_plan,
        HR=HR,
        HRb=HRb,
        ex_start=ex_start,
        ex_end=ex_end,
        seed=seed
    )
    
    # Calculating the length of the Episode
    episode_length = int((end_time - start_time).total_seconds()/60/3)
    
    # Register environment with Gymnasium
    register(
        id= "simglucose/adult2-v0",
        entry_point="my_simglucose_gym_env:T1DSimGymnaisumEnv",
        max_episode_steps=episode_length,
        kwargs={"patient_name": patiant_name,
            "custom_scenario": scenario,
            "seed": seed,
            "random_init_bg": random_init_bg,
            "to_record_model_eq": to_record_model_eq,           # Temporary test parameter
            "testing_GI_limits": testing_GI_limits},            # Temporary test parameter
    )
    
    # Make environment
    env = gym.make("simglucose/adult2-v0", render_mode="human")
    
    return env


def make_PA_environment(
    patiant_name:str,
    start_time:datetime,
    end_time:datetime,
    meal_plan:list,
    ex_start:datetime,
    ex_end:datetime,
    HR:float,
    HRb:float,
    beta:float,
    seed:int,
    random_init_bg:bool,
    to_record_model_eq:bool,             # Temporary test parameter
    testing_GI_limits:bool,              # Temporary test parameter
):
    sys.path.insert(0, "C:/Users/hsb19/OneDrive/Dokumenter/UiT/Semester/V2024/FYS-3941/Simulator/modified_files/")
    from modified_files.my_scenario import CustomScenario
    # warnings.filterwarnings("ignore", category=UserWarning, message="WARN: Overriding environment simglucose/adult2-v0 already in registry.")
    
    scenario = CustomScenario(
        start_time=start_time,
        scenario=meal_plan,
        HR=HR,
        HRb=HRb,
        ex_start=ex_start,
        ex_end=ex_end,
        beta=beta,
        seed=seed
    )
    
    # Calculating the length of the Episode
    episode_length = int((end_time - start_time).total_seconds()/60/3)
    
    # Register environment with Gymnasium
    # my_id = ("simglucose/adult2-v0" + f"test{meal_plan[0][2]:.1f}".replace(".","_"))      #  Temporary test only
    my_id = "simglucose/adult2-v0"
    register(
        id= my_id,                                                                        #  Temporary test only,  original line: id= "simglucose/adult2-v0".
        entry_point="my_simglucose_gym_env:T1DSimGymnaisumEnv",
        max_episode_steps=episode_length,
        kwargs={"patient_name": patiant_name,
            "custom_scenario": scenario,
            "seed": seed,
            "random_init_bg": random_init_bg,
            "to_record_model_eq": to_record_model_eq,               # Temporary test parameter
            "testing_GI_limits": testing_GI_limits},                # Temporary test parameter    
    )
    
    # Make environment
    env = gym.make(my_id, render_mode="human")
    
    return env

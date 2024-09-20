# from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from my_sim_env import T1DSimEnv as _T1DSimEnv

# from simglucose.patient.t1dpatient import T1DPatient
# from my_t1dpatient_PA import T1DPatient                                   # 1 Split (supports only on type of food per simulation)
# from my_t1dpatient_PA_discreate_split import T1DPatient                   # Testing 3 split
from my_t1dpatient_PA_discreate_split_100 import T1DPatient               # Test 100 splits
# from my_t1dpatient_PA_discreate_split_10 import T1DPatient                # test 10 splits
# from my_t1dpatient_PA_discreate_split_10_Normal import T1DPatient         # Non-Diabetic patient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario

# from simglucose.controller.base import Action
from my_ctrl_base import Action

import numpy as np
import pkg_resources
import gym
from gym import spaces
from gym.utils import seeding
from datetime import datetime
import gymnasium


# # Standard T1D patient parameters:
# PATIENT_PARA_FILE = pkg_resources.resource_filename(
#     "simglucose", "params/vpatient_params.csv"
# )

# Expanded population parameters:
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    __name__, "params/vpatient_params.csv"
)

# # Normal non-diabetic patient parameters:
# PATIENT_PARA_FILE = pkg_resources.resource_filename(
#     __name__, "params\my_normal_vpatient_params.csv"
# )

class T1DSimEnv(gym.Env):
    """
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    """

    metadata = {"render.modes": ["human"]}

    SENSOR_HARDWARE = "Dexcom"
    INSULIN_PUMP_HARDWARE = "Insulet"

    def __init__(
        self,
        patient_name=None,
        custom_scenario=None,
        reward_fun=None,
        seed=None,
        random_init_bg=True,
        to_record_model_eq=False,                       # Temporary test parameter
        testing_GI_limits=False,                        # Temporary test parameter
    ):
        """
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        """
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            patient_name = ["adolescent#001"]

        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.random_init_bg = random_init_bg
        self.np_random, _ = seeding.np_random(seed=seed)
        self.custom_scenario = custom_scenario
        self.to_record_model_eq = to_record_model_eq                    # Temporary test parameter
        self.testing_GI_limits = testing_GI_limits                      # Temporary test parameter
        self.env, _, _, _ = self._create_env()

    def _step(self, action: float):
        # This gym controls meal input. Each meal is represented by it's CHO value [g] and respected GI
        # If self.basal_regime is True then an patient specific basal dose will be administrated per
        # time step (pump = 1 min) in addition to the meal.
        act = Action(basal=action.basal, bolus=action.bolus, CHO=action.CHO, GI=action.GI)
        if self.reward_fun is None:
            return self.env.step(act)
        return self.env.step(act, reward_fun=self.reward_fun)

    def _raw_reset(self):
        return self.env.reset()

    def _reset(self):
        self.env, _, _, _ = self._create_env()
        obs, _, _, _ = self.env.reset()
        return obs

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env()
        return [seed1, seed2, seed3, seed4]

    def _create_env(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        seed4 = seeding.hash_seed(seed3 + 1) % 2**31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)

        if isinstance(self.patient_name, list):
            patient_name = self.np_random.choice(self.patient_name)
            patient = T1DPatient.withName(patient_name, random_init_bg=self.random_init_bg, seed=seed4)
        else:
            patient = T1DPatient.withName(
                self.patient_name, random_init_bg=self.random_init_bg, seed=seed4
            )

        if isinstance(self.custom_scenario, list):
            scenario = self.np_random.choice(self.custom_scenario)
        else:
            scenario = (
                RandomScenario(start_time=start_time, seed=seed3)
                if self.custom_scenario is None
                else self.custom_scenario
            )

        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE) # _Temporary parm_   # Temporary parm_
        env = _T1DSimEnv(patient, sensor, pump, scenario, self.to_record_model_eq, self.testing_GI_limits)
        return env, seed2, seed3, seed4

    def _render(self, mode="human", close=False):
        self.env.render(close=close)

    def _close(self):
        super()._close()
        self.env._close_viewer()

    @property
    def action_space(self):
        ub = self.env.pump._params["max_basal"]
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1000, shape=(1,))

    @property
    def max_basal(self):
        return self.env.pump._params["max_basal"]


class T1DSimGymnaisumEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    MAX_BG = 1000

    def __init__(
        self,
        patient_name=None,
        custom_scenario=None,
        reward_fun=None,
        seed=None,
        random_init_bg=True,
        render_mode=None,
        to_record_model_eq=False,           # Temporary test parameter
        testing_GI_limits=False,            # Temporary test parameter
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.env = T1DSimEnv(
            patient_name=patient_name,
            custom_scenario=custom_scenario,
            reward_fun=reward_fun,
            seed=seed,
            random_init_bg=random_init_bg,
            to_record_model_eq=to_record_model_eq,              # Temporary test parameter
            testing_GI_limits=testing_GI_limits,                # Temporary test parameter
        )
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=self.MAX_BG, shape=(1,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Box(
            low=0, high=self.env.max_basal, shape=(1,), dtype=np.float32
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Truncated will be controlled by TimeLimit wrapper when registering the env.
        # For example,
        # register(
        #     id="simglucose/adolescent2-v0",
        #     entry_point="simglucose.envs:T1DSimGymnaisumEnv",
        #     max_episode_steps=10,
        #     kwargs={"patient_name": "adolescent#002"},
        # )
        # Once the max_episode_steps is set, the truncated value will be overridden.
        truncated = False
        return np.array([obs.CGM], dtype=np.float32), reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, _, _, info = self.env._raw_reset()
        return np.array([obs.CGM], dtype=np.float32), info

    def render(self):
        if self.render_mode == "human":
            self.env.render()

    def close(self):
        self.env.close()

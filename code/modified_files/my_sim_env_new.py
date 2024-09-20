# from simglucose.patient.t1dpatient import Action
from my_t1dpatient_PA_discreate_split_100 import Action

from simglucose.analysis.risk import risk_index
import pandas as pd
from datetime import timedelta
import logging
from collections import namedtuple
# from simglucose.simulation.rendering import Viewer
from my_rendering import Viewer

try:
    from rllab.envs.base import Step
except ImportError:
    _Step = namedtuple("Step", ["observation", "reward", "done", "info"])

    def Step(observation, reward, done, **kwargs):
        """
        Convenience method creating a namedtuple with the results of the
        environment.step method.
        Put extra diagnostic info in the kwargs
        """
        return _Step(observation, reward, done, kwargs)


Observation = namedtuple("Observation", ["CGM"])
logger = logging.getLogger(__name__)


def risk_diff(BG_last_hour):
    if len(BG_last_hour) < 2:
        return 0
    else:
        _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
        _, _, risk_prev = risk_index([BG_last_hour[-2]], 1)
        return risk_prev - risk_current


class T1DSimEnv(object):                                 # _Temporary parm_        # _Temporary parm_
    def __init__(self, patient, sensor, pump, scenario, to_record_model_eq=False, testing_GI_limits=False):
        self.patient = patient
        self.sensor = sensor
        self.pump = pump
        self.scenario = scenario
        self.to_record_model_eq = to_record_model_eq            # Temporary test parameter
        self.testing_GI_limits = testing_GI_limits              # Temporary test parameter
        self._reset()

    @property
    def time(self):
        return self.scenario.start_time + timedelta(minutes=self.patient.t)

    def mini_step(self, action):
        # current action
        patient_action = self.scenario.get_action(self.time)
        basal = self.pump.basal(action.basal)
        bolus = self.pump.bolus(action.bolus)
        insulin = basal + bolus
        
        # Controller meal actions
        start_statment = (not (self.time - self.scenario.start_time).total_seconds()/60 % self.sample_time)
        ctrl_action_meal = action.CHO * start_statment
        ctrl_action_GI = action.GI * start_statment
        
        # Defining the different actions:
        patient_scen_act = Action(insulin=insulin, CHO=patient_action.meal, GI=patient_action.GI)
        patient_ctrl_act = Action(insulin=insulin, CHO=ctrl_action_meal, GI=ctrl_action_GI)
        
        # PA model variables:
        uhr = self.scenario.get_uhr(self.scenario._total_minutes(self.time))
        
        # State update
        self.patient.step(patient_ctrl_act,
                          patient_scen_act,
                          uhr, 
                          self.scenario.ex_start, 
                          self.scenario.ex_end,
                          self.scenario.beta,
                          self.to_record_model_eq,              # Temporary test parameter
                          self.testing_GI_limits)               # Temporary test parameter

        # Total currant meal
        CHO = patient_action.meal + ctrl_action_meal
        
        # next observation
        BG = self.patient.observation.Gsub
        CGM = self.sensor.measure(self.patient)

        return CHO, insulin, BG, CGM

    def step(self, action, reward_fun=risk_diff):
        """
        action is a namedtuple with keys: basal, bolus
        """
        CHO = 0.0
        insulin = 0.0
        BG = 0.0
        CGM = 0.0

        for _ in range(int(self.sample_time)):
            # Compute moving average as the sample measurements
            tmp_CHO, tmp_insulin, tmp_BG, tmp_CGM = self.mini_step(action)
            CHO += tmp_CHO / self.sample_time
            insulin += tmp_insulin / self.sample_time
            BG += tmp_BG / self.sample_time
            CGM += tmp_CGM / self.sample_time

        # Compute risk index
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)

        # Record current action
        self.CHO_hist.append(CHO)
        self.insulin_hist.append(insulin)

        # Record next observation
        self.time_hist.append(self.time)
        self.BG_hist.append(BG)
        self.CGM_hist.append(CGM)
        self.risk_hist.append(risk)
        self.LBGI_hist.append(LBGI)
        self.HBGI_hist.append(HBGI)
        self.uhr_hist.append(self.scenario.get_uhr(self.scenario._total_minutes(self.time)))

        # Compute reward, and decide whether game is over
        window_size = int(60 / self.sample_time)
        BG_last_hour = self.CGM_hist[-window_size:]
        reward = reward_fun(BG_last_hour)
        done = BG < 10 or BG > 600
        obs = Observation(CGM=CGM)

        return Step(
            observation=obs,
            reward=reward,
            done=done,
            sample_time=self.sample_time,
            patient_name=self.patient.name,
            meal=CHO,
            patient_state=self.patient.state,
            time=self.time,
            bg=BG,
            lbgi=LBGI,
            hbgi=HBGI,
            risk=risk,
        )

    def _reset(self):
        self.sample_time = self.sensor.sample_time
        self.viewer = None

        BG = self.patient.observation.Gsub
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)
        CGM = self.sensor.measure(self.patient)
        self.time_hist = [self.scenario.start_time]
        self.BG_hist = [BG]
        self.CGM_hist = [CGM]
        self.risk_hist = [risk]
        self.LBGI_hist = [LBGI]
        self.HBGI_hist = [HBGI]
        self.CHO_hist = []
        self.insulin_hist = []
        self.uhr_hist = [self.scenario.get_uhr(self.scenario._total_minutes(self.time))]

    def reset(self):
        self.patient.reset()
        self.sensor.reset()
        self.pump.reset()
        self.scenario.reset()
        self._reset()
        CGM = self.sensor.measure(self.patient)
        obs = Observation(CGM=CGM)
        return Step(
            observation=obs,
            reward=0,
            done=False,
            sample_time=self.sample_time,
            patient_name=self.patient.name,
            meal=0,
            patient_state=self.patient.state,
            time=self.time,
            bg=self.BG_hist[0],
            lbgi=self.LBGI_hist[0],
            hbgi=self.HBGI_hist[0],
            risk=self.risk_hist[0],
        )

    def render(self, close=False):
        if close:
            self._close_viewer()
            print(f"I am ending recording at time {self.time}")
            self.patient.save_test_data(save_GI=self.testing_GI_limits, save_model_eq=self.to_record_model_eq)
            return

        if self.viewer is None:
            self.viewer = Viewer(self.scenario.start_time, self.patient.name, self.scenario.ex_start, self.scenario.ex_end)

        self.viewer.render(self.show_history())

    def _close_viewer(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def show_history(self):
        df = pd.DataFrame()
        df["Time"] = pd.Series(self.time_hist)
        df["BG"] = pd.Series(self.BG_hist)
        df["CGM"] = pd.Series(self.CGM_hist)
        df["CHO"] = pd.Series(self.CHO_hist)
        df["insulin"] = pd.Series(self.insulin_hist)
        df["LBGI"] = pd.Series(self.LBGI_hist)
        df["HBGI"] = pd.Series(self.HBGI_hist)
        df["Risk"] = pd.Series(self.risk_hist)
        df["uhr"] = pd.Series(self.uhr_hist)
        df = df.set_index("Time")
        return df

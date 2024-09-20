import logging
from collections import namedtuple
from datetime import datetime
from datetime import timedelta
import numpy as np

logger = logging.getLogger(__name__)
Action = namedtuple('scenario_action', ['meal', 'GI'])


class Scenario(object):
    def __init__(self, start_time, seed):
        self.start_time = start_time
        self.random_beta = True
        self.seed = seed
    
    def get_action(self, t):
        raise NotImplementedError
    
    @property
    def generate_beta(self):
        # Found pop procentile value of beta
        beta25p = 0.0148
        beta50p = 0.0446
        beta75p = 0.0678
        
        # Assume that beta is normal distributed and calculate it's mean SD
        betaSD = ((beta75p - beta50p)/0.67 + (beta50p - beta25p)/0.67)/2
        
        # Sample beta from our designed distribution
        beta = self.random_gen.normal(loc=beta50p, scale=betaSD)
        return beta
    
    def reset(self):
        self.random_gen = np.random.RandomState(self.seed)
    
    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()

class CustomScenario(Scenario):
    def __init__(self, start_time, scenario, HR=False, HRb=60, ex_start=None, ex_end=None, beta=None, seed=None):
        """
        start_time - Timedelta representing the starting time of the simulation scenario
        
        scenario   - a list of tuples (time, action), where time is a datetime or
                     timedelta or double, action is a namedtuple defined by
                     scenario.Action. When time is a timedelta, it is
                     interpreted as the time of start_time + time. Time in double
                     type is interpreted as time in timedelta with unit of hours
        
        HR         - list/ndarray or double representing the heart rate [bpm] per minute
                     during physical activity (PA) scenario. If HR inout is of type "double", 
                     then the HR per minute during PA is assumed to follow a step function.
                     Else, if HR is of type list/ndarray then the elements of HR is assumed
                     to represent the heart rate per minute of the exercise session.
                     Defult: HR = HRb
        
        HRb        - base/basal heart rate. Defult: HRb = 60bpm
        
        ex_start   - Tinedelta or double representing the start time of the exercise session.
                     Inputs of type "double" is assumed to represent the number of hours after
                     scenario start time. Defult: ex_start = start_time
        
        ex_start   - Tinedelta or double representing the end time of the exercise session.
                     Inputs of type "double" is assumed to represent the number of hours after
                     scenario start time. Defult: ex_end = ex_start + 30 minutes
        """
        Scenario.__init__(self, start_time=start_time, seed=seed)
        self.scenario = scenario
        self.HRb = HRb
        if not HR:
            self.HR = self.HRb
        else:
            self.HR = HR
        
        if ex_start is None:
            self.ex_start = self.start_time
        else:
            self.ex_start = ex_start
        
        if ex_end is None:
            if isinstance(self.ex_start, datetime):
                self.ex_end = self.ex_start + timedelta(minutes=30)
            else:
                self.ex_end = self.ex_start + 30
        else:
            self.ex_end = ex_end
        
        if beta is not None:
            self.beta = beta
            self.random_beta = False
        
        if isinstance(self.ex_start, datetime):
            self.ex_start = self._total_minutes(self.ex_start)
        if isinstance(self.ex_end, datetime):
            self.ex_end = self._total_minutes(self.ex_end)
        self.ex_session = np.arange(self.ex_start, self.ex_end)
        
        if isinstance(self.HR, (int, float)):
            self.HR = np.ones_like(self.ex_session)*self.HR
        if isinstance(self.HR, list):
            self.HR = np.array(self.HR)
            if len(self.HR) != len(self.ex_session):
                raise ValueError(f'HR must be of the same length as the number of minutes in exercise session or of type "double"\nHR input is of length {len(self.HR)}, expected length {len(self.ex_session)} or 1')
        if not isinstance(self.HR, np.ndarray):
            raise ValueError(f"HR must be an integer or a list, not {type(HR)}")
        if not all(isinstance(element, (int, float)) for element in self.HR):
            raise ValueError(f"All elements of HR must be of type float/int")
    
    def get_action(self, t):
        """ Returns the scenario defined meal (CHO to ingest) at a given time t (DateTime) """
        if not self.scenario:
            return Action(meal=0, GI=0)
        else:
            times, actions, GI = tuple(zip(*self.scenario))
            times2compare = [parseTime(time, self.start_time) for time in times]
            if t in times2compare:
                idx = times2compare.index(t)
                return Action(meal=actions[idx], GI=GI[idx])
            return Action(meal=0, GI=0)
    
    def get_uhr(self, t):
        """ Returns the differance in HR [bpm] based at a given time t (DateTime) """
        t = self._total_minutes(t)
        mask = (self.ex_session == t)
        points_to_compere = np.sum(mask)
        if points_to_compere < 1:
            HRt = self.HRb
        elif points_to_compere > 1:
            raise ValueError(f"To many timestamps to compare. Expected 1 or 0, got {points_to_compere}")
        else:
            HRt = self.HR[mask][0]
        return HRt - self.HRb
    
    def _total_minutes(self, time):
        """ Converts DataTime into total minutes since simulation start """
        return (time - self.start_time).total_seconds()/60
    
    def reset(self):
        """ Generates a random beta parameter if no beta was given at the creation of the scenario """
        super().reset()
        if self.random_beta:
            self.beta = self.generate_beta


def parseTime(time, start_time):
    if isinstance(time, (int, float)):
        t = start_time + timedelta(minutes=round(time * 60.0))
    elif isinstance(time, timedelta):
        t_sec = time.total_seconds()
        t_min = round(t_sec / 60.0)
        t = start_time + timedelta(minutes=t_min)
    elif isinstance(time, datetime):
        t = time
    else:
        raise ValueError('Expect time to be int, float, timedelta, datetime')
    return t

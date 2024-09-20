from simglucose.patient.base import Patient
import numpy as np
from scipy.integrate import ode
import pandas as pd
from collections import namedtuple
import logging
import pkg_resources
from RecordingAndTestingTools import Recorder, GI_tester

logger = logging.getLogger(__name__)

Action = namedtuple("patient_action", ["CHO", "insulin", "kGI"])
Observation = namedtuple("observation", ["Gsub"])

# # Only Simglucose population
# PATIENT_PARA_FILE = pkg_resources.resource_filename(
#     "simglucose", "params/vpatient_params.csv"
# )

# Expanded population:
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    __name__, "params/vpatient_params.csv"
)

# Recording / Testing Tools
recorder = Recorder()
GI_recorder = GI_tester()


def digest(x, kGI, d, kgut, params):
    """ 
    A global function to calculate the digestion part of the model.
    This includes the first three differential equation and Glucose
    rate of appearance.
    
    Inputs:
        - x: a ndarray of shape (3,) 
          containing the values of Qsto1, Qsto2, and Qgut in mg respectively.
        - kGI: the GI based scalar (GI/100) for the given meal d.
        - d: amount of CHO [mg] currently consumed with the respected kGI
        - kgut: a parameter describing the rate of gastric emptying
        - params: a pandas sequence containing the relevant parameters
          kmax, kabs and BW
    
    Returns the calculated differentials dxdt and glucose rate of appearance.
    """
    dxdt = np.zeros(3)
    kgri = params.kmin + (kGI**4)*(params.kmax - params.kmin)
    kabs = (kGI**1.2) * params.kabs
    
    # Stomach solid
    dxdt[0] = -kgri * x[0] + d
    # Stomach liquid
    dxdt[1] = kgri * x[0] - x[1] * kgut
    # Intestine
    dxdt[2] = kgut * x[1] - kabs * x[2]
    # Rate of appearance
    Rat = params.f * kabs * x[2] / params.BW
    return dxdt, Rat


class T1DPatient(Patient):
    SAMPLE_TIME = 1  # min
    EAT_RATE = 5  # g/min CHO
    
    def __init__(self, params, init_state=None, random_init_bg=False, seed=None, t0=0):
        """
        T1DPatient constructor.
        Inputs:
            - params: a pandas sequence
            - init_state: customized initial state.
              If not specified, load the default initial state in
              params.iloc[2:15]
            - t0: simulation start time, it is 0 by default
        """
        self._params = params
        self._init_state = init_state
        self.random_init_bg = random_init_bg
        self._seed = seed
        self.t0 = t0
        self.reset()
    
    @classmethod
    def withID(cls, patient_id, **kwargs):
        """
        Construct patient by patient_id
        id are integers from 1 to 30.
        1  - 10: adolescent#001 - adolescent#010
        11 - 20: adult#001 - adult#001
        21 - 30: child#001 - child#010
        """
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        params = patient_params.iloc[patient_id - 1, :]
        # set fasting blood glucose if given in kwargs
        if 'fasting_bg' in kwargs and kwargs['fasting_bg'] is not None:
            params.Gb = kwargs['fasting_bg']
            params.Gpb = params.Gb * params.Vg
            params.EGPb = params.kp1 - params.kp2 * params.Gpb - params.kp3 * params.Ib
            Gtb = (params.Fsnc - params.EGPb + params.k1 * params.Gpb) / params.k2
            params.Vm0 = (params.EGPb - params.Fsnc) * (params.Km0 + Gtb) / Gtb
            params.iloc[5] = params.Gpb
            params.iloc[6] = Gtb
            params.iloc[14] = params.Gpb
        kwargs.pop('fasting_bg', None)
        return cls(params, **kwargs)
    
    @classmethod
    def withName(cls, name, **kwargs):
        """
        Construct patient by name.
        Names can be
            adolescent#001 - adolescent#010
            adult#001 - adult#001
            child#001 - child#010
        """
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        params = patient_params.loc[patient_params.Name == name].squeeze()
        # set fasting blood glucose if given in kwargs
        if 'fasting_bg' in kwargs and kwargs['fasting_bg'] is not None:
            params.Gb = kwargs['fasting_bg']
            params.Gpb = params.Gb * params.Vg
            params.EGPb = params.kp1 - params.kp2 * params.Gpb - params.kp3 * params.Ib
            Gtb = (params.Fsnc - params.EGPb + params.k1 * params.Gpb) / params.k2
            params.Vm0 = (params.EGPb - params.Fsnc) * (params.Km0 + Gtb) / Gtb
            params.iloc[5] = params.Gpb
            params.iloc[6] = Gtb
            params.iloc[14] = params.Gpb
        kwargs.pop('fasting_bg', None)
        return cls(params, **kwargs)
    
    @property
    def state(self):
        return self._odesolver.y
    
    @property
    def t(self):
        return self._odesolver.t
    
    @property
    def sample_time(self):
        return self.SAMPLE_TIME
    
    @property
    def PA_params(self):
        PAparams = namedtuple("PA_parameters", ["gamma", "epsilon", "tau_h", "k", "tau_theta"])
        # Parameters:
        PAparams.gamma = 1.2 # dimless
        PAparams.epsilon = 0.01 # dimless
        PAparams.tau_h = 10 # min
        PAparams.k = 0.1151 # # min
        PAparams.tau_theta = 180 # min
        return PAparams
    #                                                                       # _Temporary parm_        # _Temporary parm_
    def step(self, action, scenario_action, uhr, ex_start, ex_end, beta, to_record_model_eq=False, testing_GI_limits=False):
        # Convert announcing meal to the meal amount to eat at the moment
        self._add_meal_to_queue(scenario_action.CHO, round(scenario_action.kGI, ndigits=1))
        to_eat, kGI = self._announce_meal(action.CHO, round(action.kGI, ndigits=1))
        action = action._replace(CHO=to_eat, kGI=kGI)
        
        # Detect eating or not and update last digestion amount
        if action.CHO > 0 and self._last_action.CHO <= 0:
            logger.info("t = {}, patient starts eating ...".format(self.t))
            self._last_Qsto = self.state[0] + self.state[1]  # unit: mg
            self._last_foodtaken = 0  # unit: g
            self.is_eating = True
        
        if to_eat > 0:
            logger.debug("t = {}, patient eats {} g".format(self.t, action.CHO))
        
        if self.is_eating:
            self._last_foodtaken += action.CHO  # g
        
        # Detect eating ended
        if action.CHO <= 0 and self._last_action.CHO > 0:
            logger.info("t = {}, Patient finishes eating!".format(self.t))
            self.is_eating = False
        
        # Update last input
        self._last_action = action
        
        # record uhr(t_ex_end)
        if self.t == ex_end-1:
            self.uhr_end = uhr
        
        # Recording / Testing model equations:                                                  # Temporary for test purpose only
        if to_record_model_eq:                                                                  # Temporary for test purpose only
            recorder.pause_recording = False                                                    # Temporary for test purpose only
            recorder.state_recorder(h=self.state[-2], theta=self.state[-1])                     # Temporary for test purpose only
        
        # Recording / Testing GI limits.                                                        # Temporary for test purpose only
        if testing_GI_limits:                                                                   # Temporary for test purpose only
            GI_recorder.pause = False                                                           # Temporary for test purpose only
        
        # ODE solver
        self._odesolver.set_f_params(                                                                                               # _Temporary parm_  # _Temporary parm_
            action, self._params, self._last_Qsto, self._last_foodtaken, self.PA_params, beta, uhr, self.uhr_end, ex_start, ex_end, to_record_model_eq, testing_GI_limits,
        )
        if self._odesolver.successful():
            self._odesolver.integrate(self._odesolver.t + self.sample_time)
        else:
            logger.error("ODE solver failed!!")
            raise
    
    @staticmethod                                                                                            # _Temporary parm_        # _Temporary parm_
    def model(t, x, action, params, last_Qsto, last_foodtaken, PAparams, beta, uhr, uhr_end, t_start, t_end, to_record_model_eq=False, testing_GI_limits=False):
        dxdt = np.zeros(13 + 2 + (10+1)*3)  # + dxdt[-2] = dydx, dxdt[-1] = dzdt   + (10+1) (dxdt[0], dxdt[1], dxdt[2])
        d = action.CHO * 1000  # g -> mg
        insulin = action.insulin * 6000 / params.BW  # U/min -> pmol/kg/min
        basal = params.u2ss * params.BW / 6000  # U/min
        
        # Glucose in the stomach
        qsto = x[0] + x[1]
        
        # NOTE: Dbar is in unit mg, hence last_foodtaken needs to be converted
        # from mg to g. See https://github.com/jxx123/simglucose/issues/41 for
        # details.
        Dbar = last_Qsto + last_foodtaken * 1000  # unit: mg
        
        if Dbar > 0:
            aa = 5 / (2 * Dbar * (1 - params.b))
            cc = 5 / (2 * Dbar * params.d)
            kgut = params.kmin + (params.kmax - params.kmin) / 2 * (
                np.tanh(aa * (qsto - params.b * Dbar))
                - np.tanh(cc * (qsto - params.d * Dbar))
                + 2
            )
        else:
            kgut = params.kmax
        
        
        ############################### Spits ###############################
        # Calculates the differentials of Qsto1, Qsto2, Qgut (dxdt[0:3]) and 
        # the glucose rate of appearance constant (Rat) independently for 11
        # different GI values between 0 and 100. And sums up all the independent
        # calculations to form the actual differentials and rate of appearance
        Rat = 0
        for idx in range(15, len(dxdt), 3):
            kGI_idx = (idx-15)/3/10
            d_idx = 0
            if action.kGI == kGI_idx:
                d_idx = d
            dxdt[idx:idx+3], Rat_idx = digest(x[idx:idx+3], kGI_idx, d_idx, kgut, params)
            dxdt[0:3] += dxdt[idx:idx+3]
            Rat += Rat_idx
        ####################################################################
        
        
        # Glucose Production
        EGPt = params.kp1 - params.kp2 * x[3] - params.kp3 * x[8]
        
        # Glucose Utilization
        Uiit = params.Fsnc
        
        # renal excretion
        if x[3] > params.ke2:
            Et = params.ke1 * (x[3] - params.ke2)
        else:
            Et = 0
        
        # glucose kinetics
        # plus dextrose IV injection input u[2] if needed
        dxdt[3] = max(EGPt, 0) + Rat - Uiit - Et - params.k1 * x[3] + params.k2 * x[4]
        dxdt[3] = (x[3] >= 0) * dxdt[3]
        
        
        ############################### Added PA Model ###############################
        # f(t): [phi(t)]
        phi = uhr/(1 + uhr)
        
        # w(t):
        # This only works for constant HR's (independent of t)?????
        if t < t_start:
            w = 0
        elif t_start <= t < t_end:
            w = uhr
        else:
            w = uhr_end*np.exp(-PAparams.k * (t-t_end))
        
        # dh/dt:
        dxdt[13] = -(1/PAparams.tau_h) * (x[13] - uhr)
        
        # dtheta/dt:
        dxdt[14] = -(phi + (1/PAparams.tau_theta)) * x[14] + phi
        
        # Insulin-dependent utilization: 
        Uidt_numerator = params.Vm0*(1 + beta*x[13]) + params.Vmx*(1 + PAparams.gamma*x[14]) * x[6]
        Uidt_denominator = params.Km0*(1 - PAparams.epsilon *w) + x[4]
        Uidt = Uidt_numerator/Uidt_denominator * x[4]        # x[4] = Gt(t)
        
        # Recording / Testing model equations:                                                                      # Temporary for test purpose only
        if to_record_model_eq:                                                                                      # Temporary for test purpose only
            recorder.record(t=t,                                                                                    # Temporary for test purpose only
                            uhr=uhr,                                                                                # Temporary for test purpose only
                            dh=dxdt[13],                                                                            # Temporary for test purpose only
                            dtheta=dxdt[14],                                                                        # Temporary for test purpose only
                            phi=phi,                                                                                # Temporary for test purpose only
                            w=w,                                                                                    # Temporary for test purpose only
                            Uidt=Uidt)                                                                              # Temporary for test purpose only
        ##############################################################################
        
        
        dxdt[4] = -Uidt + params.k1 * x[3] - params.k2 * x[4]
        dxdt[4] = (x[4] >= 0) * dxdt[4]
        
        # insulin kinetics
        dxdt[5] = (
            -(params.m2 + params.m4) * x[5]
            + params.m1 * x[9]
            + params.ka1 * x[10]
            + params.ka2 * x[11]
        )  # plus insulin IV injection u[3] if needed
        It = x[5] / params.Vi
        dxdt[5] = (x[5] >= 0) * dxdt[5]
        
        # insulin action on glucose utilization
        dxdt[6] = -params.p2u * x[6] + params.p2u * (It - params.Ib)
        
        # insulin action on production
        dxdt[7] = -params.ki * (x[7] - It)
        
        dxdt[8] = -params.ki * (x[8] - x[7])
        
        # insulin in the liver (pmol/kg)
        dxdt[9] = -(params.m1 + params.m30) * x[9] + params.m2 * x[5]
        dxdt[9] = (x[9] >= 0) * dxdt[9]
        
        # subcutaneous insulin kinetics
        dxdt[10] = insulin - (params.ka1 + params.kd) * x[10]
        dxdt[10] = (x[10] >= 0) * dxdt[10]
        
        dxdt[11] = params.kd * x[10] - params.ka2 * x[11]
        dxdt[11] = (x[11] >= 0) * dxdt[11]
        
        # subcutaneous glucose
        dxdt[12] = -params.ksc * x[12] + params.ksc * x[3]
        dxdt[12] = (x[12] >= 0) * dxdt[12]
        
        if action.insulin > basal:
            logger.debug("t = {}, injecting insulin: {}".format(t, action.insulin))
        
        # Recording / Testing Splits
        if testing_GI_limits:                                                               # Temporary for test purpose only
            GI_recorder.record(                                                             # Temporary for test purpose only
                t=t,                                                                        # Temporary for test purpose only
                Ra=Rat,                                                                     # Temporary for test purpose only
                Q_gut=x[2],                                                                 # Temporary for test purpose only
                kGI=action.kGI,                                                             # Temporary for test purpose only
                D=d,                                                                        # Temporary for test purpose only
                Dbar=Dbar,                                                                  # Temporary for test purpose only
                k_gut=kgut,                                                                 # Temporary for test purpose only
                Q_sto1=x[0],                                                                # Temporary for test purpose only
                Q_sto2=x[1],                                                                # Temporary for test purpose only
                Q_sto=qsto,                                                                 # Temporary for test purpose only
                bg=x[12]/params.Vg,                                                         # Temporary for test purpose only
            )                                                                               # Temporary for test purpose only
        return dxdt
    
    @property
    def observation(self):
        """
        return the observation from patient
        for now, only the subcutaneous glucose level is returned
        TODO: add heart rate as an observation
        """
        GM = self.state[12]  # subcutaneous glucose (mg/kg)
        Gsub = GM / self._params.Vg
        observation = Observation(Gsub=Gsub)
        return observation
    
    def _add_meal_to_queue(self, meal, GI):
        """
        Adds a new meal to the queue
        
        If the last spot of the queue already contains a meal with
        same GI as the meal to be added then the new meal is added 
        on to that, otherwise all new meals get added to the end 
        of the queue
        """
        continuing_last_meal = False                                    # Old Version
        if len(self.meals_in_queue) > 0:                                #
            continuing_last_meal = (GI == self.meals_in_queue[-1,1])    #
        if continuing_last_meal:                                        # if GI in self.meals_in_queue[:,1]:
            self.meals_in_queue[-1,0] += meal                           #    self.meals_in_queue[self.meals_in_queue[:,1] == GI, 0] += meal
        else:
            self.meals_in_queue = np.vstack((self.meals_in_queue, [meal, GI]))
    
    def _remove_empty_queue_spots(self):
        """ Removes empty meals (CHO=0) from the queue """
        empty_spots = (self.meals_in_queue[:,0] <= 0)
        self.meals_in_queue = np.delete(self.meals_in_queue, empty_spots, 0)
    
    def _announce_meal(self, meal, GI):
        """
        patient announces meal.
        All empty meals is removed from the queue (self.meals_in_queue)
        The announced meal together with its GI will be added to the queue
        The meal located at the start of the queue will be consumed first
        The amount consumed per timestep is determined by self.EAT_RATE
        The function will return the amount to eat and its respective GI at current time
        """
        self._remove_empty_queue_spots()
        self._add_meal_to_queue(meal, GI)
        if self.meals_in_queue[0,0] > 0:
            to_eat = min(self.EAT_RATE, self.meals_in_queue[0,0])
            self.meals_in_queue[0,0] -= to_eat
            self.meals_in_queue[0,0] = max(0, self.meals_in_queue[0,0])
        else:
            to_eat = 0
        return to_eat, self.meals_in_queue[0,1]
    
    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()
    
    def reset(self):
        """
        Reset the patient state to default intial state
        """
        if self._init_state is None:
            self.init_state = np.copy(self._params.iloc[2:15].values)
        else:
            self.init_state = self._init_state
        
        # Adding y0=0, z0=0 to init state + 3*(10+1) (X[0]0=0, x[1]0=0, x[2]0=0)
        new_init_state = np.zeros(len(self.init_state)+2+3*(10+1))
        new_init_state[:len(self.init_state)] = self.init_state[:] 
        self.init_state = new_init_state
        
        self.random_state = np.random.RandomState(self.seed)
        if self.random_init_bg:
            mean = [
                1.0 * self.init_state[3],
                1.0 * self.init_state[4],
                1.0 * self.init_state[12],
            ]
            cov = np.diag(
                [
                    0.1 * self.init_state[3],
                    0.1 * self.init_state[4],
                    0.1 * self.init_state[12],
                ]
            )
            bg_init = self.random_state.multivariate_normal(mean, cov)
            self.init_state[3] = 1.0 * bg_init[0]
            self.init_state[4] = 1.0 * bg_init[1]
            self.init_state[12] = 1.0 * bg_init[2]
        
        self._last_Qsto = self.init_state[0] + self.init_state[1]
        self._last_foodtaken = 0
        self._last_kGI = 0
        self.name = self._params.Name
        self._odesolver = ode(self.model).set_integrator("dopri5")
        self._odesolver.set_initial_value(self.init_state, self.t0)
        
        self._last_action = Action(CHO=0, insulin=0, kGI=0)
        self.is_eating = False
        self.meals_in_queue = np.empty((0,2))   # resetting meals_in_queue
        self.uhr_end = None     # resetting uhr(ex_end)
        GI_recorder.__init__()                                                  # Temporary test parameter
        recorder.__init__()                                                     # Temporary test parameter
    
    def save_test_data(self, save_GI, save_model_eq):                           # Temporary test parameter
        if save_GI and (len(GI_recorder.time) > 0):                             # Temporary test parameter
            print("Saving GI recording")                                        # Temporary test parameter
            GI_recorder.save(self.name)                                         # Temporary test parameter
        if save_model_eq and (len(recorder.recorded_timesteps) > 0):            # Temporary test parameter
            print("Saving model eq recording")                                  # Temporary test parameter
            recorder.save(self.name)                                            # Temporary test parameter



if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    p = T1DPatient.withName("adolescent#001")
    basal = p._params.u2ss * p._params.BW / 6000  # U/min
    t = []
    CHO = []
    insulin = []
    BG = []
    while p.t < 1000:
        ins = basal
        carb = 0
        if p.t == 100:
            carb = 80
            ins = 80.0 / 6.0 + basal
        # if p.t == 150:
        #     ins = 80.0 / 12.0 + basal
        act = Action(insulin=ins, CHO=carb)
        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(t, BG)
    ax[1].plot(t, CHO)
    ax[2].plot(t, insulin)
    plt.show()

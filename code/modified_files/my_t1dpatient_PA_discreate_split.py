from simglucose.patient.base import Patient
import numpy as np
from scipy.integrate import ode
import pandas as pd
from collections import namedtuple
import logging
import pkg_resources

logger = logging.getLogger(__name__)

Action = namedtuple("patient_action", ["CHO", "insulin", "f"])
Observation = namedtuple("observation", ["Gsub"])

# # Only Simglucose population
# PATIENT_PARA_FILE = pkg_resources.resource_filename(
#     "simglucose", "params/vpatient_params.csv"
# )

# Expanded population
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    __name__, "params/vpatient_params.csv"
)


class Recorder():
    def __init__(self):
        self.pause_recording = True
        self.recorded_timesteps = []
        self.uhr = []
        self.h = []
        self.theta = []
        self.phi = []
        self.w = []
        self.Uidt = []
        self.dh = []
        self.dtheta = []
    
    def record(self, t, uhr=None, dh=None, dtheta=None, phi=None, w=None, Uidt=None):
        if not self.pause_recording:
            # print(f"At time {t}, records uhr={uhr}, h={dh}, theta={dtheta}, phi={phi}, w={w}, Uidt={Uidt}", end="\r")
            self.recorded_timesteps.append(t)
            self.uhr.append(uhr)
            self.dh.append(dh)
            self.dtheta.append(dtheta)
            self.phi.append(phi)
            self.w.append(w)
            self.Uidt.append(Uidt)
            self.pause_recording = True
    
    def state_recorder(self, h=None, theta=None):
        self.h.append(h)
        self.theta.append(theta)

    def __call__(self):
        df = pd.DataFrame()
        df["time"] = pd.Series(self.recorded_timesteps)
        df["uhr"] = pd.Series(self.uhr)
        df["h"] = pd.Series(self.h)
        df["theta"] = pd.Series(self.theta)
        df["phi"] = pd.Series(self.phi)
        df["w"] = pd.Series(self.w)
        df["Uidt"] = pd.Series(self.Uidt)
        df["dh"] = pd.Series(self.dh)
        df["dtheta"] = pd.Series(self.dtheta)
        df = df.set_index("time")
        return df
    
    def save(self, name):
        df = self.__call__()
        df.to_pickle(f"data/eq/{name}/PA_model_eq_Recordings.pkl")


class GI_tester():
    def __init__(self):
        self.Ra = []
        self.Q_gut = []
        self.f = []
        self.D = []
        self.Dbar = []
        self.k_gut = []
        self.Q_sto1 = []
        self.Q_sto2 = []
        self.Q_sto = []
        self.bg = []
        self.time = []
        self.pause = True

    def record(self, t, Ra, Q_gut, f, D, Dbar, k_gut, Q_sto1, Q_sto2, Q_sto, bg):
        if not self.pause:
            self.time.append(t)
            self.Ra.append(Ra)
            self.Q_gut.append(Q_gut)
            self.f.append(f)
            self.k_gut.append(k_gut)
            self.D.append(D)
            self.Dbar.append(Dbar)
            self.Q_sto1.append(Q_sto1)
            self.Q_sto2.append(Q_sto2)
            self.Q_sto.append(Q_sto)
            self.bg.append(bg)
            self.pause = True
    
    def save(self, name):
        df = pd.DataFrame()
        df["time"] = pd.Series(self.time)
        df["Ra"] = pd.Series(self.Ra)
        df["Q_gut"] = pd.Series(self.Q_gut)
        df["f"] = pd.Series(self.f)
        df["D"] = pd.Series(self.D)
        df["Dbar"] = pd.Series(self.Dbar)
        df["k_gut"] = pd.Series(self.k_gut)
        df["Q_sto1"] = pd.Series(self.Q_sto1)
        df["Q_sto2"] = pd.Series(self.Q_sto2)
        df["Q_sto"] = pd.Series(self.Q_sto)
        df["BG"] = pd.Series(self.bg)
        df = df.set_index("time")
        print(f"length of recording = {len(self.f)}")
        # df.to_pickle(f"data/test_limits/{name}/GI_{np.mean(self.f):.3f}__parameters".replace(".", "_") + ".pkl")
        # df.to_pickle("data/test_limits/{name}/no_food_parameters.pkl")
        df.to_pickle(f"data/test_split/{name}/discrete_split.pkl")

recorder = Recorder()
GI_recorder = GI_tester()

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
    def breton_parms(self):
        bparams = namedtuple("Breton_parameters", ["alpha", "beta", "gamma", "a", "THR", "Tin", "Tex", "tz", "n"])
        
        # Parameters 
        bparams.alpha = 3e-4 # dimless
        bparams.beta = 0.01  # bpm^-1
        bparams.gamma = 1e-7 # dimless
        bparams.a = 0.1 # dimless
        bparams.THR = 5 # min
        bparams.Tin = 1 # min
        bparams.Tex = 600 # min
        bparams.tz = 3*bparams.Tex
        bparams.n = 4 # dimless

        return bparams

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
        #                                                # _Temporary parm_        # _Temporary parm_
    def step(self, action, uhr, ex_start, ex_end, beta, to_record_model_eq=False, testing_GI_limits=False):
        # Convert announcing meal to the meal amount to eat at the moment
        to_eat = self._announce_meal(action.CHO)
        # Fix problem when GI = 0 and CHO > 0. A zero GI means that the food eaten has no affect on BG levels.
        # so to make sure the food dont stay around in the gut for ever we change the food input when GI=0 to 0.
        # if to_eat > 0 and action.f <= 0:
        #     to_eat = 0
        # print("time =", self.t, "to_eat =", to_eat, "action.CHO =", action.CHO)
        action = action._replace(CHO=to_eat)

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
        dxdt = np.zeros(13 + 3*3 + 2)  # + 3 new GI bands and  + dxdt[-2] = dydx  &  dxdt[-1] = dzdt
        
        d = action.CHO * 1000  # g -> mg
        
        # This method dose not take self.announce_meal into consideration. If patient still is eating when new GI is add then this solution will produce the wrong result
        d_low, d_mid, d_high = 0, 0, 0
        if 0 <= action.f < 0.4:
            d_low = d
        elif 0.4 <= action.f < 0.8:
            d_mid = d
        else:
            d_high = d
        
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
        
        
        
        ############################### Low GI ###############################
        # x[0](low) = x[13]     Qsto1
        # x[1](low) = x[14]     Qsto2
        # x[2](low) = x[15]     Qgut
        
        # Stomach solid
        dxdt[0+13] = -params.kmax * x[0+13] + d_low
        
        # stomach liquid
        dxdt[1+13] = params.kmax * x[0+13] - x[1+13] * kgut
        
        # intestine
        dxdt[2+13] = kgut * x[1+13] - 0.2 *  params.kabs  * x[2+13]
        
        # Rate of appearance
        Rat_low = 0.2 * params.f * params.kabs * x[2+13] / params.BW
        
        
        
        ############################### Mid GI ###############################
        # x[0](low) = x[16]     Qsto1
        # x[1](low) = x[17]     Qsto2
        # x[2](low) = x[18]     Qgut
        
        # Stomach solid
        dxdt[0+16] = -params.kmax * x[0+16] + d_mid
        
        # stomach liquid
        dxdt[1+16] = params.kmax * x[0+16] - x[1+16] * kgut
        
        # intestine
        dxdt[2+16] = kgut * x[1+16] - 0.6 *  params.kabs  * x[2+16]
        
        # Rate of appearance
        Rat_mid = 0.6 * params.f * params.kabs * x[2+16] / params.BW
        
        
        
        ############################### High GI ###############################
        # x[0](low) = x[19]     Qsto1
        # x[1](low) = x[20]     Qsto2
        # x[2](low) = x[21]     Qgut
        
        # Stomach solid
        dxdt[0+19] = -params.kmax * x[0+19] + d_high
        
        # stomach liquid
        dxdt[1+19] = params.kmax * x[0+19] - x[1+19] * kgut
        
        # intestine
        dxdt[2+19] = kgut * x[1+19] - 1 *  params.kabs  * x[2+19]
        
        # Rate of appearance
        Rat_high = 1 * params.f * params.kabs * x[2+19] / params.BW
        
        
        
        ############################### Merged ###############################
        # low:  +13
        # mid:  +16
        # high: +19
        
        
        # Stomach solid
        dxdt[0] = dxdt[0+13] + dxdt[0+16] + dxdt[0+19]
        
        # stomach liquid
        dxdt[1] = dxdt[1+13] + dxdt[1+16] + dxdt[1+19]
        
        # intestine
        dxdt[2] = dxdt[2+13] + dxdt[2+16] + dxdt[2+19]
        
        # Rate of appearance
        Rat = Rat_low + Rat_mid + Rat_high
        
        
        ############################### Rest ###############################
        
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
        
        # Added PA model: ------------------------------------------------------------------
        
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
        dxdt[-2] = -(1/PAparams.tau_h) * (x[-2] - uhr)
        
        # dtheta/dt:
        dxdt[-1] = -(phi + (1/PAparams.tau_theta)) * x[-1] + phi
        
        
        # Insulin-dependent utilization: 
        Uidt_numerator = params.Vm0*(1 + beta*x[-2]) + params.Vmx*(1 + PAparams.gamma*x[-1]) * x[6]
        Uidt_denominator = params.Km0*(1 - PAparams.epsilon *w) + x[4]
        Uidt = Uidt_numerator/Uidt_denominator * x[4]        # x[4] = Gt(t)
        
        # Recording / Testing model equations:                                                                      # Temporary for test purpose only
        if to_record_model_eq:                                                                                      # Temporary for test purpose only
            recorder.record(t=t,                                                                                    # Temporary for test purpose only
                            uhr=uhr,                                                                                # Temporary for test purpose only
                            dh=dxdt[-2],                                                                            # Temporary for test purpose only
                            dtheta=dxdt[-1],                                                                        # Temporary for test purpose only
                            phi=phi,                                                                                # Temporary for test purpose only
                            w=w,                                                                                    # Temporary for test purpose only
                            Uidt=Uidt)                                                                              # Temporary for test purpose only
        # ----------------------------------------------------------------------------------------
        
        
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
        
        if testing_GI_limits:                                                               # Temporary for test purpose only
            GI_recorder.record(                                                             # Temporary for test purpose only
                t=t,                                                                        # Temporary for test purpose only
                Ra=Rat,                                                                     # Temporary for test purpose only
                Q_gut=x[2],                                                                 # Temporary for test purpose only
                f=action.f,                                                                 # Temporary for test purpose only
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

    def _announce_meal(self, meal):
        """
        patient announces meal.
        The announced meal will be added to self.planned_meal
        The meal is consumed in self.EAT_RATE
        The function will return the amount to eat at current time
        """
        self.planned_meal += meal
        if self.planned_meal > 0:
            to_eat = min(self.EAT_RATE, self.planned_meal)
            self.planned_meal -= to_eat
            self.planned_meal = max(0, self.planned_meal)
        else:
            to_eat = 0
        return to_eat

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
        
        # Adding # GI bands worth of new x[0], x[1], and x[2]'s to the init state
        nGIbands = 3
        for _ in range(nGIbands):
            new_init_state = np.zeros(len(self.init_state)+3)
            new_init_state[:len(self.init_state)] = self.init_state[:]
            new_init_state[-3:] = self.init_state[:3]
            self.init_state = new_init_state
        
        # Adding y0 and z0 to init state
        new_init_state = np.zeros(len(self.init_state)+2)
        new_init_state[:len(self.init_state)] = self.init_state[:] 
        self.init_state = new_init_state # y0 = z0 = 0
        

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

        self._last_action = Action(CHO=0, insulin=0, f=0)
        self.is_eating = False
        self.planned_meal = 0
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

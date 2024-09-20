import numpy as np
import pandas as pd
import os

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



class Data_recorder():
    def __init__(self, score_func):
        self.score_func = score_func
        self.reset()
    
    def record_X(self, CHO, GI, HR, length, seed="seed_not_given", age="age_not_given"):
        self.BG = []
        self.CHO.append(CHO)
        self.GI.append(GI)
        self.HR.append(HR)
        self.length.append(length)
        self.seeds.append(seed)
        self.age.append(age)
    
    def record_Y(self, BG):
        self.raw_BG_data.append(BG)
        self.score.append(self.score_func(np.asarray(BG)))
    
    def save(self, path=""):
        # Create save directory if it does not already exist:
        savepath = self.path + path
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        
        data = pd.DataFrame()
        data["CHO"] = pd.Series(self.CHO)
        data["GI"] = pd.Series(self.GI)
        data["HR"] = pd.Series(self.HR)
        data["length"] = pd.Series(self.length)
        data["score"] = pd.Series(self.score)
        data.to_csv(savepath + "/data.csv")
        
        info = pd.DataFrame()
        info["seed"] = pd.Series(self.seeds)
        info["BG"] = pd.Series(self.raw_BG_data)
        info.to_csv(savepath + "/info.csv")
    
    def reset(self, patient="no_patient_given", simulator="no_simulator_given"):
        self.path = f"data/training_data/{patient}/{simulator}"
        self.CHO = []
        self.GI = []
        self.HR = []
        self.length = []
        self.score = []
        self.raw_BG_data = []
        self.seeds = []
        self.age = []



class GI_tester():
    def __init__(self):
        self.Ra = []
        self.Q_gut = []
        self.kGI = []
        self.D = []
        self.Dbar = []
        self.k_gut = []
        self.Q_sto1 = []
        self.Q_sto2 = []
        self.Q_sto = []
        self.bg = []
        self.time = []
        self.pause = True

    def record(self, t, Ra, Q_gut, kGI, D, Dbar, k_gut, Q_sto1, Q_sto2, Q_sto, bg):
        if not self.pause:
            self.time.append(t)
            self.Ra.append(Ra)
            self.Q_gut.append(Q_gut)
            self.kGI.append(kGI)
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
        df["kGI"] = pd.Series(self.kGI)
        df["D"] = pd.Series(self.D)
        df["Dbar"] = pd.Series(self.Dbar)
        df["k_gut"] = pd.Series(self.k_gut)
        df["Q_sto1"] = pd.Series(self.Q_sto1)
        df["Q_sto2"] = pd.Series(self.Q_sto2)
        df["Q_sto"] = pd.Series(self.Q_sto)
        df["BG"] = pd.Series(self.bg)
        df = df.set_index("time")
        print(f"length of recording = {len(self.kGI)}")
        # df.to_pickle(f"data/test_limits/{name}/GI_{np.mean(self.kGI)*100:.0f}__parameters" + ".pkl")
        # df.to_pickle(f"data/test_limits/{name}/no_food_parameters.pkl")
        # df.to_pickle(f"data/test_split/{name}/discrete_split10.pkl")
        
        # name = name + "/BB"
        
        # # Changes to only kgri
        # if np.max(self.D) <= 0:
        #     df.to_pickle(f"data/test_kgri/only_kgri/{name}/no_food_parameters.pkl")
        # else:
        #     df.to_pickle(f"data/test_kgri/only_kgri/{name}/GI_{np.max(self.kGI)*100:.0f}__parameters" + ".pkl")
        
        # # Changes to only kabs
        # if np.max(self.D) <= 0:
        #     df.to_pickle(f"data/test_kgri/only_kabs/{name}/no_food_parameters.pkl")
        # else:
        #     df.to_pickle(f"data/test_kgri/only_kabs/{name}/GI_{np.max(self.kGI)*100:.0f}__parameters" + ".pkl")
        
        # Changes to both kgri and kabs
        if np.max(self.D) <= 0:
            df.to_pickle(f"data/test_kgri/combined/{name}/no_food_parameters.pkl")
        else:
            df.to_pickle(f"data/test_kgri/combined/{name}/GI_{np.max(self.kGI)*100:.0f}__parameters" + ".pkl")
        
        # # Switching GI soluton out for GR
        # if np.max(self.D) <= 0:
        #     df.to_pickle(f"data/test_kgri/GR/{name}/no_food_parameters.pkl")
        # else:
        #     df.to_pickle(f"data/test_kgri/GR/{name}/GI_{np.max(self.kGI)*100:.0f}__parameters" + ".pkl")

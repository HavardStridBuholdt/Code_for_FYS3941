#%% #################### Import Packages: ####################
from datetime import datetime
from Run_simulator import Simulator
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import pkg_resources

PATIENT_QUEST_FILE = pkg_resources.resource_filename(
    __name__, "modified_files/params/Quest.csv")
QUEST = pd.read_csv(PATIENT_QUEST_FILE)

#%% #################### Hyperparameters: ####################

# Patient
PATIANT_NAME = "adult#004"
AGE =  QUEST[QUEST.Name.str.match(PATIANT_NAME)].Age.values[0]

# Scenrio parameters
START_HOUR = 6
START_TIME = datetime(2018, 1, 1, START_HOUR, 0)
END_TIME = datetime(2018, 1, 1, START_HOUR+12, 0)
MEAL_PLAN = [(15/60, 0, 0)]

# Exercise parameters
EX_START_TIME = datetime(2018, 1, 1, START_HOUR, 30)
EX_END_TIME = datetime(2018, 1, 1, START_HOUR+1, 30)
INTENSITY = 0.70
MAX_HR = 220 - AGE
HR = INTENSITY*MAX_HR
HRB = 72
BETA = 0.0446

# Model parameters
SIMULATOR = "PA"
N_EPISODE = 1
SEED = None
RANDOM_INIT_BG = False
BASAL_REGIME = True
BOLUS_REGIME = False

# Testing/Recording parameters
RECORD_MODEL_EQ = False
RECORD_GI_DATA = False


# %% ###################### Simulation: ######################
# Initialize simulation environment
sim = Simulator(
    simulator=SIMULATOR,
    patient_name=PATIANT_NAME,
    start_time=START_TIME,
    end_time=END_TIME,
    meal_plan=MEAL_PLAN,
    ex_start=EX_START_TIME,
    ex_end=EX_END_TIME,
    HR=HR,
    HRb=HRB,
    beta=BETA,
    seed=SEED,
    random_init_bg=RANDOM_INIT_BG,
    basal_regime =BASAL_REGIME,
    bolus_regime=BOLUS_REGIME,
    to_record_model_eq=RECORD_MODEL_EQ,
    testing_GI_limits=RECORD_GI_DATA,
    record=True
)
start = time.time()
# Run Simulation
sim.run(N_EPISODE, render=True)
end = time.time()
print(f"Simulation run took {end-start:.5f} seconds to finish ")

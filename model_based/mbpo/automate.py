import os
import yaml
import random
import sys
from train_mbpo import train_mbpo
import time
import datetime
import json
cwd =  os.path.dirname(os.path.abspath(__file__))
print(sys.executable)

# Load the configuration from the CFG file
with open(os.path.join(cwd, 'config_one.yaml'), 'r') as file:
    config = yaml.safe_load(file)

def pick_random_params(config):
    params = {}
    print(config.keys())  # Print the top-level keys to confirm structure
    
    # Access the 'parameters' section
    parameters = config.get('parameters', {})
    
    # Iterate through the parameters
    for key, value_str in parameters.items():
        values = value_str.split(',')
        params[key] = random.choice(values).strip()
    
    return params

# Script to run
script_name = os.path.join(cwd, "train_mbpo.py")

# Number of random runs
num_runs = 10
now = datetime.datetime.now()
log_folder = os.path.join(cwd, "Logs" + "_" + str(now.year) + "_" +str( now.month) + "_" + str(now.day)+ "_" + str(now.hour) +"_"+str(now.minute)+"_"+str(now.second))
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
    print(f"Directory '{log_folder}' created.")
else:
    print(f"Directory '{log_folder}' already exists.")
for i in range(num_runs):
    # Pick random parameters
    random_params = pick_random_params(config)
    
    now = str(datetime.datetime.now())
    current_run_log = os.path.join(log_folder, "params" + "_" + "now" + "_run" + str(i)) 
    current_run_log_file = os.path.join(current_run_log, "params.json")
    if not os.path.exists(current_run_log):
        os.makedirs(current_run_log)
        print(f"Directory '{current_run_log}' created.")
    else:
        print(f"Directory '{current_run_log}' already exists.")
    random_params["video_path"] = current_run_log
    with open(current_run_log_file, 'w') as file:
        json.dump(random_params, file, indent=4) 
 
    # Set the environment variables based on random parameter
    trainer = train_mbpo()
    trainer.update_parameters(random_params)
    # Run the script
    result =  trainer.run_training()

    print(result)
    # Output the result
    print(f"Run {i+1} with parameters: {random_params}")

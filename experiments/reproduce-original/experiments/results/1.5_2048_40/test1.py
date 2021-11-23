import pandas as pd
import datetime
from pathlib import Path
import os

# set paths
root_path = Path.cwd()
config_path = root_path / Path('configs.csv')
src_path = root_path / Path('SynapticMetaplasticityBNN/Continual_Learning_Fig-2abcdefgh-3abcd-5cde')
main_path = src_path / Path('main.py')



# set path where main.py is
os.chdir(src_path)


# read configs
configs = pd.read_csv(config_path)
configs = configs.iloc[5::6]

if 'done' not in configs:
    configs['done'] = 0



for i, row in enumerate(configs.iterrows()):
    t0 = datetime.datetime.now()
    row = row[1]
    meta_val = row.meta
    n_h = int(row.num_hid)
    epoch = int(row.epochs)
    print('-'*100)
    print('META = %.2f' %(meta_val), 'NUM_HIDS = %.2f' %(n_h), 'EPOCHS = %.2f' %(epoch))
    os.system(
            "python " \
            f"{main_path} " \
            "--net 'bnn' " \
            f"--hidden-layers {n_h} {n_h} " \
            "--lr 0.005 " \
            "--decay 1e-7 " \
            f"--meta {meta_val} " \
            f"--epochs-per-task {epoch} " \
            "--task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'"
            )
    
    task_time = (datetime.datetime.now() - t0).total_seconds()
    # set task time on df
    configs.loc[i, 'time'] = task_time
    
    # set experiment as done on df
    configs.loc[i, 'done'] = 1

    print(f"Task took: {task_time} seconds")
    configs.to_csv('config_path', index=False)

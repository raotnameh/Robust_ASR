import pandas as pd
import argparse
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('-ld','--log-directory',type=str,nargs='*',help='Path to the log directory')
parser.add_argument('-pd','--plot-directory',type=str,help='Path to save the plots')
args = parser.parse_args()
logs_directory = args.log_directory
save_directory = args.plot_directory
if save_directory[-1] != "/":
    save_directory +="/"

df_list = []
file_names = []
stats = ""

for log_directory in logs_directory:
    if log_directory[-1]!="/":
        log_directory +='/'
    file_name = log_directory.split("/")
    file_name = [f for f in file_name if f][-1]
    file_names.append(file_name)
    stats += f"{file_name} Statistics:\n"
    with open(log_directory+"train.log","r") as f:
        lines = f.readlines()
        separated_list = [line.strip().split(",") for line in lines]
        data_list = separated_list[1:]
        labels = separated_list[0]+['alpha','beta','gamma']
        labels = [label+"_"+file_name for label in labels]

    df = pd.DataFrame(data_list,columns=labels,index=None,dtype=float)
    df_list.append(df)
    stats +=f"Total Number of Epochs: {1+int(df.describe()['epoch_'+file_name]['max'])}\n"
    stats +=f"Mean Time for Epoch: {df.describe()['epoch_time_'+file_name]['mean']}\n"
    stats += f"Min WER: {df.describe()['wer_'+file_name]['min']}\n"
    stats += f"Min CER: {df.describe()['cer_'+file_name]['min']}\n\n"

print(stats)
compiled_df = pd.concat(df_list,axis=1)
time_vs_epoch = compiled_df.plot(x = "epoch_results", y = ["epoch_time_"+f for f in file_names],figsize=(10,10),xlabel="epochs",ylabel="Time taken for Epoch")
wer_vs_epoch = compiled_df.plot(x = "epoch_results", y = ["wer_"+f for f in file_names],figsize=(10,10),xlabel="epochs",ylabel="WER")
cer_vs_epoch = compiled_df.plot(x = "epoch_results", y = ["cer_"+f for f in file_names],figsize=(10,10),xlabel="epochs",ylabel="CER")
time_vs_epoch.figure.savefig(save_directory+"time_vs_epoch")
wer_vs_epoch.figure.savefig(save_directory+"wer_vs_epoch")
cer_vs_epoch.figure.savefig(save_directory+"cer_vs_epoch")

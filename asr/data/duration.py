import glob
from tqdm.auto import tqdm

import subprocess

def get_length(input_video):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_video], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return [input_video,float(result.stdout)]

files = glob.glob("*.csv")
print(files)
max_duration = 15.0
min_duration = 1.0
root_path = '/media/data_dump/hemant/harsh/voxforge_dataset/'
import pandas, os
from concurrent.futures import ProcessPoolExecutor
for file in files:
    with open(file, "r") as f:
        csv = f.readlines()
    wav = [root_path+i.split('|')[0] for i in csv]


    def run(get_length, wav):
            with ProcessPoolExecutor(max_workers=12) as executor:
                results = list(tqdm((executor.map(get_length, wav)), total=len(wav)))
            return results
    temp = run(get_length, wav)
    duration = [i[-1] for i in temp]
    print(f"{file} info: \nmin = {min(duration)} seconds\nmax = {max(duration)} seconds\ntotal = {round(sum(duration)/3600,3)} Hrs and {round(sum(duration)/60,3)} Mins")
    print(f"percentage of files less than {max_duration} seconds: , {len([i for i in duration if i <max_duration])/len(duration)}")
    print(f"percentage of files less than {min_duration} seconds: , {len([i for i in duration if i <min_duration])/len(duration)}")
    print(f"Number files to be delted are: {len([i for i in duration if i >max_duration]) + len([i for i in duration if i <min_duration]) }")
    
    #paths = [i[0] for i in temp if i[-1]>max_duration or i[-1]<min_duration]
    #df = pandas.DataFrame(data={"paths": paths})
    #df.to_csv('/media/data_dump/hemant/rachit/dataset/to_delete/mohit/' +os.path.basename(file), sep=',',index=False)
    
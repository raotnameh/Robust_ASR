import argparse
from tqdm import tqdm
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor


parser = argparse.ArgumentParser(description='Concatenate audio files')
parser.add_argument('--csv',help = 'path to source directory which contains wav files')
parser.add_argument('--dst-dir',help = 'full path to destination directory which will contain final wav files')
parser.add_argument('--num-workers', type=int, help='number of workers')

args = parser.parse_args()


with open(args.csv, "r") as f:
    csv = f.readlines()


def augment_audio_with_sox(input_):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    path, sample_rate, tempo, gain, destination,k = input_
    #print(k)
    augmented_filename = f"{destination}/{tempo}_{path.split('/')[-1]}"
    sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
    sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                                    augmented_filename,
                                                                                    " ".join(sox_augment_params))
    # print(sox_params)
    os.system(sox_params)

try:os.system(f"rm -rf {args.dst_dir}")
except: pass
os.makedirs(args.dst_dir, exist_ok=True)

tempo_value_ = [0.7,0.75,0.8,0.85,0.9,0.95,1,1.1]
low_gain, high_gain = (-4,4)

updated_csv = ''
print("Different tempo vlaues for processing: ",[i for i in tempo_value_])
for tempo_value in tempo_value_:
    inp = []
    for k,i in enumerate(csv):
        paths = i.split(',') 
        wav = paths[0]
        gain_value = np.random.uniform(low=low_gain, high=high_gain)
        inp.append((wav,16000,tempo_value,gain_value,args.dst_dir,k))
        paths[0] = os.path.join(args.dst_dir, f"{tempo_value}_{wav.split('/')[-1]}")
        updated_csv += ",".join(paths)
    print("starting processing for the tempo value of: ", tempo_value, len(inp))
 
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        _ = list(tqdm((executor.map(augment_audio_with_sox, inp)), total=len(inp)))


with open(f"{args.csv.split('.')[0]}_augmented.csv", "w") as f:
    f.write(updated_csv)

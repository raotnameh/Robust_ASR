# Librispeech Data Normalisation For ASR

This branch can be utilized to create normalized CSVs for the librispeech datasets. 

## CSV Data

It also contains a csvs directory which contains csvs corresponding to librispeech data on deathstar. On unzipping the csvs zipped file you will find 3 directories named `individual`, `individual_speaker` and `individual_segregated_speaker`. `individual`contains csvs corresponding to librispeech data without any labels. `individual_speaker` contains csvs corresponding to librispeech data with gender labels in the csvs. `individual_segregated_speaker`contains csvs corresponding to librispeech data which are separated on the basis of gender, for example: a file containing an `_M`in its file name means that it only contains male gender data.

It also contains `libri_home_csvs`, which contains csvs present in the home directory at Deathstar. The details regarding the CSVs are given below.

### outd.csv

Total Time: 10.509 Hrs
min audio time: 1.065 secs
max audio time: 35.155 secs
male time: 19035.96 secs
female time: 18796.42 secs

### 50.csv

Total Time: 50.0 Hrs
min audio time: 1.17 secs
max audio time: 21.46 secs
male time: 88762.90 secs
female time: 91238.75 secs

### new_out.csv

Total Time: 956.991 Hrs
min audio time: 0.83 secs
max audio time: 29.73 secs
male time: 1777701.25 secs
female time: 1667466.091 secs

### new_train_m_f.csv

Total Time: 315.812 Hrs
min audio time: 0.83 secs
max audio time: 15.0 secs
male time: 587330.64 secs
female time: 549591.33 secs

## Python Scripts

The branch contains 4 python files which can be utilised to create these CSVs. 

Please note that the Librispeech data directory has the following hierarchy: `Librispeech/base_data_dir/speaker_dir/chapter_dir/wavfiles`. Here base_data_dir refers to directories like `train-clean-100` etc, speaker_dir is the directory which provides speaker tags for each data point, chapter_dir refers to the chapter read by the speaker.

Follow the following steps to normalize the Librispeech data and create csv for the same. First, normalize wav files and txt from the librispeech data folder and then convert the paths to csvs, finally add the gender labels to your csvs. The procedure is detailed below

## Converting metadata to txt

Utilizing the meta data provided by librispeech to construct txt files.

To run libToTxt.py, run the following command-

`python libToTxt.py --src-path [path to base_dir] --dst-path [path to destn saving txts]`

## Normalise Wav Files

Each audio file in the librispeech dataset is in flac format. We need to convert it to a wav file with 16khz sample rate, 16 bit depth, 1 channel.

To run normalizeWAV.py, run the following command-

`python normalizeWAV.py --src-path [path to base_dir] --dst-path [path to destn saving txts]`

## Creating CSVs

Create csvs from the wav files and txt files.

To run libriToCsv.py, run the following command-

`python libriToCsv.py --wav-path [path to folder containing final wav files] --txt-path [path to folder containing final txt files] --csv-path [path to folder where final csvs need to be placed]`

## Mapping Speaker ID to Gender Labels

Each data point in the csv created has a speaker id associated with it. We need to replace these speaker ids with a gender label.

To run libri_mappingspeaker.py, run the following command-

`python libri_mappingspeaker.py --speakertxt-path [path to file speaker.txt in Librispeech data] --csvs-path [path to folder where final csvs are placed]`





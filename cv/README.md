# Commonvoice Data and Data Normalisation For ASR

This directory contains csv data for [Commonvoice](https://commonvoice.mozilla.org/en/datasets) and python files for data preprocessing for commonvoice. 

The branch contains 4 python files which can be utilised to create these CSVs. 

It also contains a cv_v6 directory which contains train, test and dev data for common voice version 6. Each train, test and dev folder contains a zipped file which contains a `train.csv` which provides path (on Deathstar) to wav and text files corresponding to the data points which have all the labels of age, gender and accent available. The `out.csv` is a csv with all columns similar to `train.csv` but with an additional column of timesteps of each wavfile. The zipped file also contains 3 additional directories of accent, age and gender which provide individual labels in the csv.

For ease of usage, a bash file has been created which can be run to quickly modify common voice data for model training.

## Running bash script

`./run.sh [Path to CV Corpus Language Directory] [Path to labels.json file which contains the relevant labels for training] [label used for discriminator classification. for eg: "age", "gender", "accent". (can also be left out if no discriminator training is required)]`

The above command can be used to run the bash script which would create all the relevant data in common voice language directory. It will create wav/, txt/, csvs/ directories in the common voice language directory where all the data to be used for training would be kept.

## Other ways to use 

If you dont want to use the bash script, you can also use the python files. Follow the following steps to normalize the Common Voice data and create csv for the same. 

### Converting tsvs to csvs

The common voice data has tsvs corresponding to their data. We need to convert these tsvs into csvs for appropriate use.

To run tsvToCsv.py, run the following command-

`python tsvToCsv.py --src-dir [path to cv corpus language folder] --label [label used for discriminator classification. for eg: "age", "gender", "accent". (can also be left out if no discriminator training is required)]`

The final csvs would be kept in `[cv_corpus_lang_directory_path]/csvs/`.

### Converting tsvs to txt

Each tsv has the text corresponding to the audio file, we need to use this text to create txt files for our use.

To run tsvToTxt.py, run the following command-

`python tsvToTxt.py --src-dir [path to cv-corpus en folder] --label-path [path to labels.json]`

Txt files would be kept in `[cv_corpus_lang_directory_path]/txt/`.

### Normalise Wav Files

Each audio file in the common voice dataset is in mp3 format. We need to convert it to a wav file with 16khz sample rate, 16 bit depth, 1 channel.

To run normalizeWAV.py, run the following command-

`python normalizeWAV.py --src-dir [path to cv-corpus en folder]`

WAV files would be kept in `[cv_corpus_lang_directory_path]/wav/`.

### File Exists

Finally, we use the created csv file to check if there is mentioned in the created csv but has a missinng wav or text file. We remove these instances from the csv.

To run file_exists.py, run the following command-

`python file_exists.py --src-dir [path to cv-corpus en folder]

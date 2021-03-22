# Commonvoice Data and Data Normalisation For ASR

This directory contains csv data for [Commonvoice](https://commonvoice.mozilla.org/en/datasets) and python files for data preprocessing for commonvoice. 

The branch contains 4 python files which can be utilised to create these CSVs. 

It also contains a cv_v6 directory which contains train, test and dev data for common voice version 6. Each train, test and dev folder contains a zipped file which contains a `train.csv` which provides path (on Deathstar) to wav and text files corresponding to the data points which have all the labels of age, gender and accent available. The `out.csv` is a csv with all columns similar to `train.csv` but with an additional column of timesteps of each wavfile. The zipped file also contains 3 additional directories of accent, age and gender which provide individual labels in the csv.

Follow the following steps to normalize the Common Voice data and create csv for the same. First, create a wav and txt in the common voice en folder before proceeding with the steps below

## Converting tsvs to csvs

The common voice data has tsvs corresponding to their data. We need to convert these tsvs into csvs for appropriate use.

To run tsvToCsv.py, run the following command-

`python tsvToCsv.py --src-path [path to cv corpus en folder] --csv-path [path to final csv to be created] --label [label to create csv for]`

Add all the tsvs to lst list which you want to consider for conversion.

## Converting tsvs to txt

Each tsv has the text corresponding to the audio file, we need to use this text to create txt files for our use.

To run tsvToTxt.py, run the following command-

`python tsvToTxt.py --src-dir [path to cv-corpus en folder]`

## Normalise Wav Files

Each audio file in the common voice dataset is in mp3 format. We need to convert it to a wav file with 16khz sample rate, 16 bit depth, 1 channel.

To run normalizeWAV.py, run the following command-

`python normalizeWAV.py --source-path [path to source directory which contains mp3 files] --dest-path [path to destination directory which will contain final wav files]`

## File Exists

Finally, we use the created csv file to check if there is mentioned in the created csv but has a missinng wav or text file. We remove these instances from the csv.

To run file_exists.py, run the following command-

`python file_exists.py --src-path [path to CSV]`

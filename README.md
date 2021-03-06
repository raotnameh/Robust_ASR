# Commonvoice Data Normalisation For ASR
## Converting tsvs to csvs

To run commonvoiceAgeData.py, run the following command-

`python tsvToCsv.py --src-path [path to cv corpus en folder] --csv-path [path to final csv to be created] --label [label to create csv for]`

Add all the tsvs to lst list which you want to consider for conversion.

## Converting tsvs to txt

To run tsvToTxt.py, run the following command-

`python tsvToTxt.py --src-dir [path to cv-corpus en folder]`

## Normalise Wav Files
To run normalizeWAV.py, run the following command-

`python normalizeWAV.py --source-path [path to source directory which contains mp3 files] --dest-path [path to destination directory which will contain final wav files]`

## File Exists
To run file_exists.py, run the following command-

`python file_exists.py --src-path [path to CSV]`

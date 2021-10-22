#!/bin/bash

if [ -z "$3"]
then
    python tsvToCsv.py  --src-dir $1
else
    python tsvToCsv.py  --src-dir $1 --label $3
fi
python tsvToTxt.py --src-dir $1 --label-path $2
python normalizeWAV.py --src-dir $1
python file_exists.py --src-dir $1
 

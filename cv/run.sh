#!/bin/bash

python tsvToCsv.py  --src-dir $1 --label $3
python tsvToTxt.py --src-dir $1 --label-path $2
python normalizeWAV.py --src-dir $1
python file_exists.py --src-dir $1
 

# Robust Automatic Speech Recognition

## Dataset

Accent dataset has been extracted from **Mozilla Common Voice corpus**. It contains data from 9 different accents:

**United States English (US)**, **England English (EN)**, **Australian English (AU)**, **Canadian English (CA)**, **Scottish English (SC)**, **Irish English (IR)** and **Welsh English**.

The dataset contains 38.73 Hrs of data. It has been divided into 5 parts:

### train7 
This is the basic training data <br/>
Contains 7 accents: US (32 %), EN (32 %), AU (14 %), CA (12 %), SC (5%), IR (3%), WE (1%) <br />
Hours of speech: 34.3 Hrs<br />

### dev4 
This the validation or development data <br />
Contains 4 accents: US (55%), EN(30%), AU(8%), CA(7%) <br />
Hours of speech: 1.26 Hrs <br />

### test4 
This the basic test data <br />
Contains 4 accents: US (56%), EN(27%), AU(9%), CA(8%) <br />
Hours of speech: 1.26 Hrs <br/>

### Test-NZ 
This is unseen accent data <br />
Contains 1 accent: NZ <br />
Hours of speech: 0.59 Hrs <br />

### Test-IN
This is unseen accent data <br />
Contains 1 accent: IN <br />
Hours of speech: 1.33 <br />

## Command
```
python train.py --train-manifest data/train_sorted.csv --val-manifest data/dev_sorted.csv --cuda --rnn-type gru --hidden-layers 3 --momentum 0.94 --opt-level O1 --loss-scale 1.0 --hidden-size 1024 --epochs 100 --lr 0.0001  --batch-size 24 --gpu-rank 4 --checkpoint --save-folder /media/data_dump/hemant/rachit/invarient_Weights/ --update-rule 4
```

Accent dataset has been extracted from Mozilla Common Voice corpus. It contains data from 9 different accents:

United States English (US), England English (EN), Australian English (AU), Canadian English (CA), Scottish English (SC), Irish English (IR) and Welsh English.

The Entire data set contains 34,901 utterances or 38.73 Hrs. It has been divided into 5 parts:

train7: this is the basic training data
Contains 7 accents: US (32 %), EN (32 %), AU (14 %), CA (12 %), SC (5%), IR (3%), WE (1%)
Hours of speech: 34.3 Hrs 
# utterances: 30,896

dev4: this the validation or development data
Contains 4 accents: US (55%), EN(30%), AU(8%), CA(7%)
Hours of speech: 1.26 Hrs 
# utterances: 1142

test4: this the basic test data
Contains 4 accents: US (56%), EN(27%), AU(9%), CA(8%)
Hours of speech: 1.26 Hrs 
# utterances: 1142

Test-NZ: this is unseen accent data
Contains 1 accent: NZ
Hours of speech: 0.59 Hrs 
# utterances: 536

Test-IN : this is unseen accent data
Contains 1 accent: IN
Hours of speech: 1.33
# utterances: 1200


python train.py --train-manifest data/train_sorted.csv --val-manifest data/dev_sorted.csv --cuda --rnn-type gru --hidden-layers 3 --momentum 0.94 --opt-level O1 --loss-scale 1.0 --hidden-size 1024 --epochs 100 --lr 0.0001  --batch-size 24 --gpu-rank 4 --checkpoint --save-folder /media/data_dump/hemant/rachit/invarient_Weights/ --update-rule 4

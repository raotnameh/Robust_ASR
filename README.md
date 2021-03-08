<sub> **Implentation of the [De-STT: De-entaglement of unwanted Nuisances and Biases in Speech to Text System using Adversarial Forgetting paper](https://arxiv.org/abs/2011.12979).**</sub> <br/>

## Building it from Source
Install [PyTorch](https://github.com/pytorch/pytorch#installation) if you haven't already.

If you want decoding to support beam search with an optional language model, install ctcdecode:
```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
```
Install horovod for multi-gpu training. Make sure [nccl](https://developer.nvidia.com/nccl) is installed.
```
HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]
```

Finally clone this repo and run this within the repo:
```
pip install -r requirements.txt
```

## Training

### Dataset
For dataset used for training and testing please refer to the section 4.1 in the [paper](https://arxiv.org/pdf/2011.12979.pdf).

### Custom Dataset
To create a custom dataset you must create a .csv file containing the locations of the data. This has to be in the format of:
```
/path/to/audio.wav,/path/to/text.txt,EN\n
/path/to/audio2.wav,/path/to/text2.txt,IN\n
...
```

The first path is to the audio file, the second path is to a text file containing the transcript, and the third is the accent label. 

### Merging multiple manifest files

To create bigger manifest files (to train/test on multiple datasets at once) we can merge manifest files together like below using different csvs.
```
cat file_1.csv file_2.cav > combined.csv
```

After this run data/Sortagrad.py on the resulting .csv file.

Sample csv files are present in the data/csvs folder for reference.


### Training a Model
```
python train.py --train-manifest data/csvs/train_sorted_EN_US.csv --val-manifest data/csvs/dev_sorted_EN_US.csv --cuda --rnn-type gru --hidden-layers 5 --hidden-size 1024 --epochs 50 --lr 0.001 --batch-size 32 --gpu-rank 1 --update-rule 1 --exp-name /path/to/save/the/results -mw-alpha 0.1 --mw-beta 0.2 --mw-gamma 0.6 --enco-modules 2 --enco-res --forg-modules 2 --forg-res --num-epochs 1 --checkpoint-per-batch 5000 --checkpoint --continue-from /path/to/saved/model.pth
```

### Multi-GPU training a Model using Horovod.
```
horovodrun -np 2 -H localhost:2 python train_horovod.py --train-manifest data/csvs/train_sorted_EN_US.csv --val-manifest data/csvs/dev_sorted_EN_US.csv --cuda --rnn-type gru --hidden-layers 5 --hidden-size 1024 --epochs 50 --lr 0.001 --batch-size 32 --gpu-rank 1 --update-rule 1 --exp-name /path/to/save/the/results -mw-alpha 0.1 --mw-beta 0.2 --mw-gamma 0.6 --enco-modules 2 --enco-res --forg-modules 2 --forg-res --num-epochs 1 --checkpoint-per-batch 5000 --checkpoint --continue-from /path/to/saved/model.pth
```

Use `python train.py -h` for more parameters and options.


### Tensorboard Logging
1. After starting trainig, start tensorboad server on server at some port XXXX, using following command: tensorboard --logdir=path_to_tbd_logdir_folder_inside_save_directory --port=XXXX
2. On your local-machine execute port forwarding command: ssh -L YYYY:localhost:XXXX hemant@192.168.3.6 
3. Open brower and hit: http://localhost:YYYY/

## Testing/Inference

To evaluate a trained model on a test set (has to be in the same format as the training set):

```
python test.py --test-manifest data/csvs/dev_sorted_EN_US.csv --gpu-rank 3 --model-path /media/data_dump/hemant/janvijay/new_code/Robust_ASR/0.0001_0.2_0.01/models/ckpt_final.pth --f --d --cuda --batch-size 8 --save-output test_save --save-representation test_save
```

Further, to employ a KenLM based LM for beam resoring:

```
python search_lm_params.py --model-path 0.0001_0.2_0.01/models/ckpt_final.pth --saved-output test_save/out.pth --output-path tune_results.json --lm-path lm/train7_sorted.binary --beam-width 128;
python select_lm_params.py --input-path tune_results.json;
```

### Using an ARPA LM

We support using kenlm based LMs. To build your own LM you need to use the KenLM repo found [here](https://github.com/kpu/kenlm). Have a read of the documentation to get a sense of how to train your own LM. The above steps once trained can be used to find the appropriate parameters.

### Alternate Decoders
By default, `test.py` use a `GreedyDecoder` which picks the highest-likelihood output label at each timestep. Repeated and blank symbols are then filtered to give the final output.

A beam search decoder can optionally be used with the installation of the `ctcdecode` library as described in the Installation section. The `test` scripts have a `--decoder` argument. To use the beam decoder, add `--decoder beam`. 

## Sampe-dataset

SOON

## Pre-trained models

SOON

## Experimental results: <br/>
|%                      |dev EN US  |Libri clean|Libri other|TestIN    |TestNZ    |Discriminator accuracy|
|-----------------------|-----------|-----------|-----------|----------|----------|----------------------|
|Baseline               |40.16      |63.90      |77.13      |69.15     |39.62     |83%                   |
|Adversarial forgetting |34.66      |58.62      |73.43      |63.01     |33.50     |72%                   |
|Abs. improvement       |**-5.50**  |**-5.28**  |**-3.70**  |**-6.13** |**-6.02** |**-11%**              |


### Citation

@misc{yadav2020destt,<br/>
      title={De-STT: De-entaglement of unwanted Nuisances and Biases in Speech to Text System using Adversarial Forgetting}, <br/>
      author={Hemant Yadav and Janvijay Singh and Atul Anshuman Singh and Rachit Mittal and Rajiv Ratn Shah},<br/>
      year={2020},<br/>
      eprint={2011.12979},<br/>
      archivePrefix={arXiv},<br/>
      primaryClass={cs.SD}<br/>
}<br/>


###### For any queries, the preferred method is to open an issue. For any personal queries please [EMAIL](hemantya@iiitd.ac.in) me.
#### Acknowledgements
Thanks to [Sean](https://github.com/SeanNaren) to open source his DS2 code!



# Audio To Spectogram/Spectogram to Audio
## Running Audio to Spectogram

To run aud2spec.py, run the following command-

`python aud2spec.py --audio-path [input audio file path] -save-path [spectogram save directory]`

You can also add addition arguments like `--window-size`, `--window-stride` , `--sample-rate`.

## Running Spectogram to Audio

To run spec2aud.py, run the following command-

`python aud2spec.py --mag-path [input spectogram magnitude np matrix file path] --phase-path [input spectogram phase np matrix file path] -save-path [audio save directory]`

You can also add addition arguments like `--window-size`, `--window-stride` , `--sample-rate`.

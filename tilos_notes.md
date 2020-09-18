### espnet setup
* dependencies
```shell script
sudo apt-get install -y flac
sudo apt-get install libfreetype6-dev
```
* espnet
```shell script
cd espnet
sudo apt-get update && sudo apt-get install -qq bc tree sox
pip install -e . && mkdir -p tools/venv/bin && touch tools/venv/bin/activate
```
* warp-ctc
```shell script
git clone https://github.com/espnet/warp-ctc -b pytorch-1.1
cd warp-ctc && mkdir build && cd build && cmake .. && make -j4
cd pytorch_binding && python setup.py install
```
* kaldi: `cd espnet && bash ci/install_kaldi.sh`
* espnet: `cd espnet/tools && ./setup_venv.sh $(command -v python3) && make`

#### run
* adjust LIBRISPEECH path in db.sh
* shell-script: `egs2/librispeech/asr1/run_minimal.sh`

* train-dev-set names in `data.sh` and `run_minimal.sh` must be set to same values
```shell script
train_set="debug_train"
train_dev="debug_dev"
```

* python command
0. `source tools/venv/bin/activate`
1. `cd egs2/librispeech/asr1`
1. `python train_tokenizer.py` -> `data/token_list/bpe_unigram5000/`: `bpe.model  bpe.vocab  tokens.txt  train.txt`
1. only once: `./run_minimal.sh --stage 9 --stop_stage 9` -> `exp/asr_stats_raw/train/`: `speech_shape`, `text_shape.bpe`
2. `LRU_CACHE_CAPACITY=1 python ~/code/SPEECH/espnet/espnet2/bin/main.py` see [Memory leak when evaluating model on CPU with dynamic size tensor input](https://github.com/pytorch/pytorch/issues/29893) and [here](https://raberrytv.wordpress.com/2020/03/25/pytorch-free-your-memory/)

### files really needed
* in `dump/raw/<dataset-name>`: `text`, `wav.scp`, `feats_type` containing only `raw`
    
### strange things
* normalization when reading audio-files; `1<<31` as denominator leads to same behavior as soundfile.read (which cannot read mp3)
* global_mvn see GlobalMVN-class; does normalize globally (over entire dataset) -> why? isn't there a normalization for each signal that would made such globel normalization obsolet?
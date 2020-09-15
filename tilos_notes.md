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
```shell script
/home/tilo/code/SPEECH/espnet/tools/venv/bin/python3 /home/tilo/code/SPEECH/espnet/espnet2/bin/asr_train.py --use_preprocessor true --bpemodel data/token_list/bpe_unigram5000/bpe.model --token_type bpe --token_list data/token_list/bpe_unigram5000/tokens.txt --non_linguistic_symbols none --cleaner none --g2p none --valid_data_path_and_name_and_type dump/raw/debug_dev/wav.scp,speech,sound --valid_data_path_and_name_and_type dump/raw/debug_dev/text,text,text --valid_shape_file exp/asr_stats_raw/valid/speech_shape --valid_shape_file exp/asr_stats_raw/valid/text_shape.bpe --resume true --fold_length 5000 --fold_length 150 --output_dir exp/asr_train_asr_transformer_tiny_raw_bpe --config conf/tuning/train_asr_transformer_tiny.yaml --frontend_conf fs=16k --normalize=global_mvn --normalize_conf stats_file=exp/asr_stats_raw/train/feats_stats.npz --train_data_path_and_name_and_type dump/raw/debug_train/wav.scp,speech,sound --train_data_path_and_name_and_type dump/raw/debug_train/text,text,text --train_shape_file exp/asr_stats_raw/train/speech_shape --train_shape_file exp/asr_stats_raw/train/text_shape.bpe --ngpu 0 --multiprocessing_distributed False
```
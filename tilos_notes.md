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


* python command
1. `cd egs2/librispeech/asr1`
2. `LRU_CACHE_CAPACITY=1 python ../../../espnet2/bin/main.py` see [Memory leak when evaluating model on CPU with dynamic size tensor input](https://github.com/pytorch/pytorch/issues/29893) and [here](https://raberrytv.wordpress.com/2020/03/25/pytorch-free-your-memory/)

    
### strange things
* normalization when reading audio-files; `1<<31` as denominator leads to same behavior as soundfile.read (which cannot read mp3)
* global_mvn see GlobalMVN-class; does normalize globally (over entire dataset) -> why? isn't there a normalization for each signal that would made such globel normalization obsolet?
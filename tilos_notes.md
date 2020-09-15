### espnet setup
* espnet
```shell script
sudo apt-get install libfreetype6-dev
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
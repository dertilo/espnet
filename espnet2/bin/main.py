import os

import argparse

from espnet2.bin.tokenize_text import tokenize, get_parser
from espnet2.tasks.asr import ASRTask
from util import data_io, util_methods
import sentencepiece as spm


def build_manifest_files(
    manifest_path="/tmp",
    dataset_path="/home/tilo/data/asr_data/ENGLISH/LibriSpeech/dev-clean_preprocessed",
    limit=None,  # just for debug
):
    os.makedirs(manifest_path, exist_ok=True)

    g = data_io.read_jsonl(f"{dataset_path}/manifest.jsonl.gz", limit=limit)
    data_io.write_lines(
        f"{manifest_path}/wav.scp",
        (
            f"{d['audio_file'].replace('.mp3', '')}\t{dataset_path}/{d['audio_file']}"
            for d in g
        ),
    )
    g = data_io.read_jsonl(f"{dataset_path}/manifest.jsonl.gz", limit=limit)
    data_io.write_lines(
        f"{manifest_path}/text",
        (f"{d['audio_file'].replace('.mp3', '')}\t{d['text']}" for d in g),
    )
    data_io.write_file(f"{manifest_path}/feats_type", "raw")


def write_vocabulary(tokenizer_dir="data/token_list/bpe_unigram5000"):
    cmd = (
        f"--token_type bpe --input {tokenizer_dir}/train.txt --output {tokenizer_dir}/tokens.txt "
        f"--bpemodel {tokenizer_dir}/bpe.model --field 2- --cleaner none --g2p none "
        f"--write_vocabulary true "
        f"--add_symbol '<blank>:0' --add_symbol '<unk>:1' --add_symbol '<sos/eos>:-1'"
    )
    parser = get_parser()
    args = parser.parse_args(shlex.split(cmd))
    kwargs = vars(args)
    tokenize(**kwargs)


def train_tokenizer(vocab_size=5000, tokenizer_dir="data/token_list/bpe_unigram5000"):

    os.makedirs(tokenizer_dir, exist_ok=True)
    spm_args = dict(
        input=(f"{tokenizer_dir}/train.txt"),
        vocab_size=vocab_size,
        model_type="unigram",
        model_prefix=(f"{tokenizer_dir}/bpe"),
        character_coverage=1.0,
        input_sentence_size=100000000,
    )
    os.system(
        f'<"{train_manifest_path}/text" cut -f 2- -d" "  > {tokenizer_dir}/train.txt'
    )

    spm.SentencePieceTrainer.Train(
        " ".join([f"--{k}={v}" for k, v in spm_args.items()])
    )

    write_vocabulary(tokenizer_dir)


def run_asr_task(
    output_dir, num_gpus=0, is_distributed=False, num_workers=0, collect_stats=False
):

    argString = (
        f"--collect_stats {collect_stats} "
        f"--use_preprocessor true "
        f"--bpemodel {tokenizer_path}/bpe.model "
        f"--seed 42 "
        f"--num_workers {num_workers} "
        f"--token_type bpe "
        f"--token_list {tokenizer_path}/tokens.txt "
        f"--g2p none "
        f"--non_linguistic_symbols none "
        f"--cleaner none "
        f"--resume true "
        f"--fold_length 5000 "
        f"--fold_length 150 "
        f"--config {config} "
        f"--frontend_conf fs=16k "
        f"--output_dir {output_dir} "
        f"--train_data_path_and_name_and_type {train_manifest_path}/wav.scp,speech,sound "
        f"--train_data_path_and_name_and_type {train_manifest_path}/text,text,text "
        f"--valid_data_path_and_name_and_type {dev_manifest_path}/wav.scp,speech,sound "
        f"--valid_data_path_and_name_and_type {dev_manifest_path}/text,text,text "
        f"--ngpu {num_gpus} "
        f"--multiprocessing_distributed {is_distributed} "
    )
    if not collect_stats:
        argString += (
            f"--train_shape_file {stats_dir}/train/speech_shape "
            f"--train_shape_file {stats_dir}/train/text_shape "
            f"--valid_shape_file {stats_dir}/valid/speech_shape "
            f"--valid_shape_file {stats_dir}/valid/text_shape "
        )
    else:
        argString += (
            f"--train_shape_file {train_manifest_path}/wav.scp "
            f"--valid_shape_file {dev_manifest_path}/wav.scp "
        )
    parser = ASRTask.get_parser()
    args = parser.parse_args(shlex.split(argString))
    ASRTask.main(args=args)


if __name__ == "__main__":
    os.environ["LRU_CACHE_CAPACITY"]=str(1) #see [Memory leak when evaluating model on CPU with dynamic size tensor input](https://github.com/pytorch/pytorch/issues/29893) and [here](https://raberrytv.wordpress.com/2020/03/25/pytorch-free-your-memory/)
    import shlex

    num_workers = 0
    vocab_size = 500
    limit = 200 #just for debug

    base_path = "/tmp/espnet_data"
    tokenizer_path = f"{base_path}/bpe_tokenizer_unigram_{vocab_size}"

    stats_dir = f"{base_path}/stats"
    config = "conf/tuning/train_asr_transformer_tiny.yaml"

    manifest_path = f"{base_path}/manifests"

    train_name = "debug_train"
    valid_name = "debug_valid"

    train_manifest_path = f"{manifest_path}/{train_name}"
    dev_manifest_path = f"{manifest_path}/{valid_name}"

    build_manifest_files(train_manifest_path, limit=limit)
    build_manifest_files(dev_manifest_path, limit=limit)

    if not os.path.isdir(tokenizer_path):
        train_tokenizer(vocab_size, tokenizer_path)

    if not os.path.isdir(stats_dir):
        run_asr_task(collect_stats=True, output_dir=f"{stats_dir}")

    assert os.path.isdir(stats_dir)

    num_gpus = 0
    is_distributed = False
    run_asr_task(
        num_workers=0,
        output_dir=f"{base_path}/{config.split('/')[-1].replace('.yaml','')}",
    )

    """
    LRU_CACHE_CAPACITY=1 python ~/code/SPEECH/espnet/espnet2/bin/main.py
    
    [tilo-ThinkPad-X1-Carbon-6th] 2020-09-17 18:14:26,464 (trainer:243) INFO: 1epoch results: [train] iter_time=0.164, forward_time=0.952, loss=480.386, loss_att=191.654, loss_ctc=1.154e+03, acc=1.022e-04, backward_time=0.973, optim_step_time=0.007, lr_0=2.080e-06, train_time=2.105, time=1 minute and 43.19 seconds, total_count=49, [valid] loss=479.955, loss_att=191.637, loss_ctc=1.153e+03, acc=1.030e-04, cer=1.010, wer=1.000, cer_ctc=4.993, time=50.5 seconds, total_count=49, [att_plot] time=3.2 seconds, total_count=0
    [tilo-ThinkPad-X1-Carbon-6th] 2020-09-17 18:14:28,067 (trainer:286) INFO: The best model has been updated: valid.acc
    [tilo-ThinkPad-X1-Carbon-6th] 2020-09-17 18:14:28,068 (trainer:322) INFO: The training was finished at 1 epochs 
    

    """

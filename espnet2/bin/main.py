import os

import argparse

from espnet2.bin.tokenize_text import tokenize, get_parser
from espnet2.tasks.asr import ASRTask
from util import data_io, util_methods
import sentencepiece as spm


def build_manifest_files(
    manifest_path="/tmp",
    dataset_path="/home/tilo/data/asr_data/ENGLISH/LibriSpeech/dev-clean_preprocessed",
):
    os.makedirs(manifest_path, exist_ok=True)

    g = data_io.read_jsonl(f"{dataset_path}/manifest.jsonl.gz")
    data_io.write_lines(
        f"{manifest_path}/wav.scp",
        (
            f"{d['audio_file'].replace('.mp3', '')}\t{dataset_path}/{d['audio_file']}"
            for d in g
        ),
    )
    g = data_io.read_jsonl(f"{dataset_path}/manifest.jsonl.gz")
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


def train_tokenizer(vocab_size=5000, tokenizer_dir="data/token_list/bpe_unigram"):
    tokenizer_dir=f"{tokenizer_dir}{vocab_size}"
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
        f'<"dump/raw/debug_train/text" cut -f 2- -d" "  > data/token_list/bpe_unigram{vocab_size}/train.txt'
    )

    spm.SentencePieceTrainer.Train(
        " ".join([f"--{k}={v}" for k, v in spm_args.items()])
    )

    write_vocabulary(tokenizer_dir)


def dryrun_to_calc_stats(
    train_set="debug_train",
    valid_set="debug_dev",
    test_sets="dev_clean",
    nbpe=5000,
    asr_config="conf/tuning/train_asr_transformer_tiny.yaml",
    inference_config="conf/decode_asr.yaml",
):

    cmd = f"""
    echo $(which python) &&
    ./asr.sh \
    --lang en \
    --ngpu 0 \
    --use_lm false \
    --stage 9 \
    --stop_stage 9 \
    --nbpe {nbpe} \
    --max_wav_duration 30 \
    --asr_config {asr_config} \
    --inference_config {inference_config} \
    --train_set {train_set} \
    --valid_set {valid_set} \
    --test_sets {test_sets} \
    --srctexts "data/{train_set}/text" "$@"
    """

    print(util_methods.exec_command(cmd))


if __name__ == "__main__":
    import shlex

    num_workers = 0
    tokenizer_path = "data/token_list/bpe_unigram5000"
    bpe_model = f"{tokenizer_path}/bpe.model"
    stats_dir = "exp/asr_stats_raw"
    output_dir = "exp/asr_train_asr_transformer_tiny_raw_bpe"
    config = "conf/tuning/train_asr_transformer_tiny.yaml"

    # manifest_path = "/tmp/espnet_manifests"
    manifest_path = "dump/raw"  # stupid asr.sh depends on this

    train_name = "debug_train"
    valid_name = "debug_valid"

    train_manifest_path = f"{manifest_path}/{train_name}"
    dev_manifest_path = f"{manifest_path}/{valid_name}"
    build_manifest_files(train_manifest_path)
    build_manifest_files(dev_manifest_path)

    vocab_size = 5000
    tokenizer_dir="data/token_list/bpe_unigram"
    if not os.path.isdir(tokenizer_dir):
        train_tokenizer(vocab_size,tokenizer_dir)

    if not os.path.isdir(stats_dir):
        dryrun_to_calc_stats(train_set=train_name, valid_set=valid_name)

    num_gpus = 0
    is_distributed = False
    argString = (
        f"--use_preprocessor true --bpemodel {bpe_model} --seed 42 --num_workers {num_workers} --token_type bpe "
        f"--token_list {tokenizer_path}/tokens.txt "
        f"--g2p none --non_linguistic_symbols none --cleaner none "
        f"--valid_data_path_and_name_and_type {dev_manifest_path}/wav.scp,speech,sound "
        f"--valid_data_path_and_name_and_type {dev_manifest_path}/text,text,text "
        f"--resume true --fold_length 5000 --fold_length 150 "
        f"--output_dir {output_dir} --config {config} --frontend_conf fs=16k "
        f"--train_data_path_and_name_and_type {train_manifest_path}/wav.scp,speech,sound "
        f"--train_data_path_and_name_and_type {train_manifest_path}/text,text,text "

        f"--train_shape_file {stats_dir}/train/speech_shape "
        f"--train_shape_file {stats_dir}/train/text_shape.bpe "
        f"--valid_shape_file {stats_dir}/valid/speech_shape "
        f"--valid_shape_file {stats_dir}/valid/text_shape.bpe "

        f"--ngpu {num_gpus} "
        f"--multiprocessing_distributed {is_distributed}"
    )

    parser = ASRTask.get_parser()
    args = parser.parse_args(shlex.split(argString))
    ASRTask.main(args=args)

    """
    LRU_CACHE_CAPACITY=1 python ~/code/SPEECH/espnet/espnet2/bin/main.py
    
    [tilo-ThinkPad-X1-Carbon-6th] 2020-09-17 18:14:26,464 (trainer:243) INFO: 1epoch results: [train] iter_time=0.164, forward_time=0.952, loss=480.386, loss_att=191.654, loss_ctc=1.154e+03, acc=1.022e-04, backward_time=0.973, optim_step_time=0.007, lr_0=2.080e-06, train_time=2.105, time=1 minute and 43.19 seconds, total_count=49, [valid] loss=479.955, loss_att=191.637, loss_ctc=1.153e+03, acc=1.030e-04, cer=1.010, wer=1.000, cer_ctc=4.993, time=50.5 seconds, total_count=49, [att_plot] time=3.2 seconds, total_count=0
    [tilo-ThinkPad-X1-Carbon-6th] 2020-09-17 18:14:28,067 (trainer:286) INFO: The best model has been updated: valid.acc
    [tilo-ThinkPad-X1-Carbon-6th] 2020-09-17 18:14:28,068 (trainer:322) INFO: The training was finished at 1 epochs 
    """

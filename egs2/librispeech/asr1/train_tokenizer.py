import sys

import os
import shlex

import sentencepiece as spm

from espnet2.bin.tokenize_text import get_parser, tokenize

if __name__ == '__main__':
    os.makedirs("data/token_list/bpe_unigram5000",exist_ok=True)
    vocab_size = 5000
    spm_args = dict(
        input="data/token_list/bpe_unigram5000/train.txt",
        vocab_size=vocab_size,
        model_type="unigram",
        model_prefix="data/token_list/bpe_unigram5000/bpe",
        character_coverage=1.0,
        input_sentence_size=100000000
    )

    os.system(f'<"dump/raw/debug_train/text" cut -f 2- -d" "  > data/token_list/bpe_unigram{vocab_size}/train.txt')
    spm.SentencePieceTrainer.Train(" ".join([f"--{k}={v}" for k,v in spm_args.items()]))

    cmd = "--token_type bpe --input data/token_list/bpe_unigram5000/train.txt --output data/token_list/bpe_unigram5000/tokens.txt --bpemodel data/token_list/bpe_unigram5000/bpe.model --field 2- --cleaner none --g2p none --write_vocabulary true --add_symbol '<blank>:0' --add_symbol '<unk>:1' --add_symbol '<sos/eos>:-1'"

    parser = get_parser()
    args = parser.parse_args(shlex.split(cmd))
    kwargs = vars(args)
    tokenize(**kwargs)

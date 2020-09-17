#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

token_listdir=data/token_list
bpemode=unigram     # Mode of BPE (unigram or bpe).
nbpe=5000             # The number of BPE vocabulary.
bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
token_list="${bpedir}"/tokens.txt
bpeprefix="${bpedir}"/bpe
bpe_char_cover=1.0  # character coverage when modeling BPE
bpe_input_sentence_size=100000000 # Size of input sentence for BPE.
bpemodel="${bpeprefix}".model
token_type=bpe      # Tokenization type (char or bpe).
data_feats=dump/raw

mkdir -p "${bpedir}"
# shellcheck disable=SC2002
<"dump/raw/debug_train/text" cut -f 2- -d" "  > "${bpedir}"/train.txt

_opts_spm=""

python ../../../utils/spm_train \
    --input="${bpedir}"/train.txt \
    --vocab_size="${nbpe}" \
    --model_type="${bpemode}" \
    --model_prefix="${bpeprefix}" \
    --character_coverage=${bpe_char_cover} \
    --input_sentence_size="${bpe_input_sentence_size}" \
    ${_opts_spm}

_opts="--bpemodel ${bpemodel}"

# The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
# 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task

cleaner=none     # Text cleaner.
g2p=none         # g2p method (needed if token_type=phn).
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole

python -m espnet2.bin.tokenize_text  \
    --token_type "${token_type}" \
    --input "${bpedir}/train.txt" --output "${token_list}" ${_opts} \
    --field 2- \
    --cleaner "${cleaner}" \
    --g2p "${g2p}" \
    --write_vocabulary true \
    --add_symbol "${blank}:0" \
    --add_symbol "${oov}:1" \
    --add_symbol "${sos_eos}:-1"
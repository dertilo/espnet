from espnet2.tasks.asr import ASRTask

if __name__ == '__main__':
    import shlex
    argString = '--use_preprocessor true --bpemodel data/token_list/bpe_unigram5000/bpe.model --token_type bpe --token_list data/token_list/bpe_unigram5000/tokens.txt --non_linguistic_symbols none --cleaner none --g2p none --valid_data_path_and_name_and_type dump/raw/debug_dev/wav.scp,speech,sound --valid_data_path_and_name_and_type dump/raw/debug_dev/text,text,text --valid_shape_file exp/asr_stats_raw/valid/speech_shape --valid_shape_file exp/asr_stats_raw/valid/text_shape.bpe --resume true --fold_length 5000 --fold_length 150 --output_dir exp/asr_train_asr_transformer_tiny_raw_bpe --config conf/tuning/train_asr_transformer_tiny.yaml --frontend_conf fs=16k --normalize=global_mvn --normalize_conf stats_file=exp/asr_stats_raw/train/feats_stats.npz --train_data_path_and_name_and_type dump/raw/debug_train/wav.scp,speech,sound --train_data_path_and_name_and_type dump/raw/debug_train/text,text,text --train_shape_file exp/asr_stats_raw/train/speech_shape --train_shape_file exp/asr_stats_raw/train/text_shape.bpe --ngpu 0 --multiprocessing_distributed False'
    parser = ASRTask.get_parser()
    args = parser.parse_args(shlex.split(argString))
    ASRTask.main(args=args)

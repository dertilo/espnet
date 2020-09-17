from espnet2.tasks.asr import ASRTask

if __name__ == '__main__':
    import shlex
    argString = '--use_preprocessor true --bpemodel data/token_list/bpe_unigram5000/bpe.model --seed 42 --num_workers 0 --token_type bpe --token_list data/token_list/bpe_unigram5000/tokens.txt --non_linguistic_symbols none --cleaner none --g2p none --valid_data_path_and_name_and_type dump/raw/debug_dev/wav.scp,speech,sound --valid_data_path_and_name_and_type dump/raw/debug_dev/text,text,text --valid_shape_file exp/asr_stats_raw/valid/speech_shape --valid_shape_file exp/asr_stats_raw/valid/text_shape.bpe --resume true --fold_length 5000 --fold_length 150 --output_dir exp/asr_train_asr_transformer_tiny_raw_bpe --config conf/tuning/train_asr_transformer_tiny.yaml --frontend_conf fs=16k --train_data_path_and_name_and_type dump/raw/debug_train/wav.scp,speech,sound --train_data_path_and_name_and_type dump/raw/debug_train/text,text,text --train_shape_file exp/asr_stats_raw/train/speech_shape --train_shape_file exp/asr_stats_raw/train/text_shape.bpe --ngpu 0 --multiprocessing_distributed False'
    parser = ASRTask.get_parser()
    args = parser.parse_args(shlex.split(argString))
    ASRTask.main(args=args)

    """
    LRU_CACHE_CAPACITY=1 python ~/code/SPEECH/espnet/espnet2/bin/main.py
    
    [tilo-ThinkPad-X1-Carbon-6th] 2020-09-17 18:14:26,464 (trainer:243) INFO: 1epoch results: [train] iter_time=0.164, forward_time=0.952, loss=480.386, loss_att=191.654, loss_ctc=1.154e+03, acc=1.022e-04, backward_time=0.973, optim_step_time=0.007, lr_0=2.080e-06, train_time=2.105, time=1 minute and 43.19 seconds, total_count=49, [valid] loss=479.955, loss_att=191.637, loss_ctc=1.153e+03, acc=1.030e-04, cer=1.010, wer=1.000, cer_ctc=4.993, time=50.5 seconds, total_count=49, [att_plot] time=3.2 seconds, total_count=0
    [tilo-ThinkPad-X1-Carbon-6th] 2020-09-17 18:14:28,067 (trainer:286) INFO: The best model has been updated: valid.acc
    [tilo-ThinkPad-X1-Carbon-6th] 2020-09-17 18:14:28,068 (trainer:322) INFO: The training was finished at 1 epochs 
    """
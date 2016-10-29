def get_config_cs2en():
    config = {}

    # Model related -----------------------------------------------------------

    # Sequences longer than this will be discarded
    config['seq_len'] = 50

    # Number of hidden units in encoder/decoder GRU
    config['hid_sz'] = 512
    config['dec_nhids'] = 1024

    # Dimension of the word embedding matrix in encoder/decoder
    config['emb_sz'] = 512
    config['dec_embed'] = 512

    # dimension of the output layer
    config['att_sz'] = 512
    config['lstm_num_of_layers'] = 1

    # Where to save model, this corresponds to 'prefix' in groundhog
    config['models_out_dir'] = 'models_ch2en'
    # test output dir
    config['test_out_dir'] = ''

    # Optimization related ----------------------------------------------------

    # Batch size
    config['batch_size'] = 80

    # This many batches will be read ahead and sorted
    config['sort_k_batches'] = 20

    # Optimization step rule
    config['step_rule'] = 'AdaDelta'

    # Gradient clipping threshold
    config['step_clipping'] = 1.

    # Std of weight initialization
    config['weight_scale'] = 0.01

    # Regularization related --------------------------------------------------

    # Weight noise flag for feed forward layers
    config['weight_noise_ff'] = False

    # Weight noise flag for recurrent layers
    config['weight_noise_rec'] = False

    # Dropout ratio, applied only after readout maxout
    config['dropout'] = 0.5

    # Vocabulary/dataset related ----------------------------------------------

    # Root directory for dataset
    config['datadir'] = './data/'

    # Module name of the stream that will be used
    config['stream'] = 'stream'

    config['prepare_file'] = './prepare_data.py'
    config['preprocess_file'] = './dict_preprocess.py'

    # Source and target vocabularies
    config['src_vocab'] = config['datadir'] + 'vocab.zh-en.zh.pkl'
    config['trg_vocab'] = config['datadir'] + 'vocab.zh-en.en.pkl'

    # Source and target datasets
    config['src_data'] = config['datadir'] + 'train.zh'
    config['trg_data'] = config['datadir'] + 'train.en'
    config['dict_data'] = config['datadir'] + 'train.sent.dict'

    # Source and target vocabulary sizes, should include bos, eos, unk tokens
    config['src_vocab_size'] = 30000
    config['trg_vocab_size'] = 30000

    # Special tokens and indexes
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'

    # Early stopping based on bleu related ------------------------------------

    # Normalize cost according to sequence length after beam-search
    config['normalized_bleu'] = True

    # Bleu script that will be used (moses multi-perl in this case)
    config['bleu_script'] = config['datadir'] + 'multi-bleu.perl'

    # Validation set gold file
    config['val_set_grndtruth'] = config['datadir'] + 'tst.en'

    # Print validation output to file
    config['output_val_set'] = True

    # Validation and test file dir
    config['val_tst_dir'] = '/scratch2/wzhang/3.corpus/2.mt/nist-all/allnist_stanfordseg_jiujiu/'

    # Validation prefix
    config['val_prefix'] = 'nist02'

    # Model prefix
    config['model_prefix'] = config['models_out_dir'] + '/params'

    # Validation set source file
    config['val_set'] = config['val_tst_dir'] + config['val_prefix'] + '.src'

    # Validation output directory
    config['val_out_dir'] = 'valids'
    # Test output directory
    config['tst_out_dir'] = ''

    # Validation output file
    config['val_set_out'] = config['val_out_dir'] + '/trans'

    # Beam-size
    config['beam_size'] = 12

    # Timing/monitoring related -----------------------------------------------

    # Maximum number of updates
    config['max_epoch'] = 2

    # Reload model from files if exist
    config['reload'] = True

    # Save model after this many updates
    #config['save_freq'] = 10000
    config['save_freq'] = 100

    # about 22500 batches for one epoch
    # Show samples from model after this many updates
    #config['sampling_freq'] = 10000
    config['sampling_freq'] = 10

    # Show details
    #config['display_freq'] = 1000
    config['display_freq'] = 10

    # Show this many samples at each sampling, need less than batch size
    config['hook_samples'] = 3

    # Validate bleu after this many updates
    #config['bleu_val_freq'] = 10000
    config['bleu_val_freq'] = 100

    # Start bleu validation after this many updates
    config['val_burn_in'] = 10000
    config['val_burn_in'] = 30

    # whether use fixed sampling or randomly sampling
    config['if_fixed_sampling'] = True

    # Start fix sampling after this many batches
    config['k_batch_start_sample'] = 1000

    # the v^T in Haitao's paper, the top k target words merged into target vocabulary
    config['topk_trg_vocab'] = 50
    config['topk_trg_pkl'] = config['datadir'] + \
        str(config['topk_trg_vocab']) + 'vocab.zh-en.en.pkl'
    config['trg_cands_dict'] = config['datadir'] + 'dict2dict_cands.dict'
    config['trg_cands_pkl'] = config['datadir'] + 'dict2dict_cands.pkl'
    config['lex_f2e'] = '/scratch2/wzhang/1.research/7.extract-phrase/1.8m-moses-nosmooth/moses-train/model/lex.f2e'
    config['phrase_table'] = '/scratch2/wzhang/1.research/7.extract-phrase/1.8m-moses-nosmooth/moses-train/model/phrase-table.gz'

    # valid.sent.dict
    config['valid_sent_dict'] = config['datadir'] + 'valid.sent.dict'

    # whether manipulate vocabulary or not
    config['if_man_vocab'] = True

    config['eval_dir'] = 'eval'

    return config

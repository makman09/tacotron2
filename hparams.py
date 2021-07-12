from text import symbols
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf

def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""
    hp_dict = {
        "epochs": 10000,
        "iters_per_checkpoint": 100,
        "seed": 1234,
        "dynamic_loss_scaling": True,
        "fp16_run": False,
        "distributed_run": False,
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54321",
        "cudnn_enabled": True,
        "cudnn_benchmark": False,
        "ignore_layers": ['embedding.weight'],
        "load_mel_from_disk": False,
        "training_files": 'TRAINING_FILE_PATH',
        "validation_files": 'VALIDATION_FILE_PATH',
        "text_cleaners": ['english_cleaners'],
        "max_wav_value": 32768.0,
        "sampling_rate": 22050,
        "filter_length": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mel_channels": 80,
        "mel_fmin": 0.0,
        "mel_fmax": 8000.0,
        "n_symbols": len(symbols),
        "symbols_embedding_dim": 512,
        "encoder_kernel_size": 5,
        "encoder_n_convolutions": 3,
        "encoder_embedding_dim": 512,
        "n_frames_per_step": 1,
        "decoder_rnn_dim": 1024,
        "prenet_dim": 256,
        "max_decoder_steps": 1000,
        "gate_threshold": 0.5,
        "p_attention_dropout": 0.1,
        "p_decoder_dropout": 0.1,
        "attention_rnn_dim": 1024,
        "attention_dim": 128,
        "attention_location_n_filters": 32,
        "attention_location_kernel_size": 31,
        "postnet_embedding_dim": 512,
        "postnet_kernel_size": 5,
        "postnet_n_convolutions": 5,
        "use_saved_learning_rate": True,
        "learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "grad_clip_thresh": 1.0,
        "batch_size": 32,
        "mask_padding": True,  # set model's padded outputs to padded values
        "drop_frame_rate": 0.2,
        "use_mmi": True,
        "use_gaf": True,
        "max_gaf": 0.5,
        "global_mean_npy": 'ljspeech_global_mean.npy'
    }
    hparams = hp.HParam(hp_dict)

    if hparams_string:
        tf.compat.v1.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.compat.v1.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
 
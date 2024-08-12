import sys
import importlib.util


def load_default_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.DEFAULT_CONFIG


def load_user_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.USER_CONFIG


def extract_peak_from_signal_bs(miRNA_seq, signal, sequence, conservation, bs_start, bs_end, neighborhood_start, neighborhood_end):
    return max(signal[bs_start:bs_end]) 


DEFAULT_CONFIG = {
    'signal_peak' : extract_peak_from_signal_bs,
}
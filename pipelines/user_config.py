import sys
import importlib.util
from funmirtar.models.local_features_config import extract_peak_from_signal_bs, load_user_config


USER_CONFIG = {
    'signal_peak' : extract_peak_from_signal_bs,
}
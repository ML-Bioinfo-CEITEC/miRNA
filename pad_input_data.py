from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
from torch import tensor

def pad_features(input_data_normalized, pad_to_length):
    flatten_input = []
    for sample in input_data_normalized:
        flatten_sample = []
        for bs in sample:
            flatten_sample.extend(bs)
        flatten_input.append(flatten_sample)

    # Convert to tensor
    data_tensor = pad_sequence([tensor(item) for item in flatten_input], batch_first=True)

    # Calculate the padding required
    padding = (0, pad_to_length - data_tensor.size(1))

    # Pad with zeros
    return pad(data_tensor, padding, value=0)
    
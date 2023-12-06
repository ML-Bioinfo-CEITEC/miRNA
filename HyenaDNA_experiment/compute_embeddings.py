from standalone_hyenadna import CharacterTokenizer, HyenaDNAModel
from transformers import PreTrainedModel
import argparse
from tqdm import tqdm
import pandas as pd
import re
import os
import json
import torch


class HyenaDNAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    base_model_prefix = "hyenadna"

    def __init__(self, config):
        pass

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    @classmethod
    def from_pretrained(cls,
                        path,
                        model_name,
                        download=False,
                        config=None,
                        device='cpu',
                        use_head=False,
                        n_classes=2,
                      ):
        # first check if it is a local path
        pretrained_model_name_or_path = os.path.join(path, model_name)
        if os.path.isdir(pretrained_model_name_or_path) and download == False:
            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))
        else:
            hf_url = f'https://huggingface.co/LongSafari/{model_name}'

            subprocess.run(f'rm -rf {pretrained_model_name_or_path}', shell=True)
            command = f'mkdir -p {path} && cd {path} && git lfs install && git clone {hf_url}'
            subprocess.run(command, shell=True)

            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))

        scratch_model = HyenaDNAModel(**config, use_head=use_head, n_classes=n_classes)  # the new model format
        loaded_ckpt = torch.load(
            os.path.join(pretrained_model_name_or_path, 'weights.ckpt'),
            map_location=torch.device(device)
        )

        # need to load weights slightly different if using gradient checkpointing
        if config.get("checkpoint_mixer", False):
            checkpointing = config["checkpoint_mixer"] == True or config["checkpoint_mixer"] == True
        else:
            checkpointing = False

        # grab state dict from both and load weights
        state_dict = load_weights(scratch_model.state_dict(), loaded_ckpt['state_dict'], checkpointing=checkpointing)

        # scratch model has now been updated
        scratch_model.load_state_dict(state_dict)
        print("Loaded pretrained weights ok!")
        return scratch_model


# helper 1
def inject_substring(orig_str):
    """Hack to handle matching keys between models trained with and without
    gradient checkpointing."""

    # modify for mixer keys
    pattern = r"\.mixer"
    injection = ".mixer.layer"

    modified_string = re.sub(pattern, injection, orig_str)

    # modify for mlp keys
    pattern = r"\.mlp"
    injection = ".mlp.layer"

    modified_string = re.sub(pattern, injection, modified_string)

    return modified_string

# helper 2
def load_weights(scratch_dict, pretrained_dict, checkpointing=False):
    """Loads pretrained (backbone only) weights into the scratch state dict."""

    # loop thru state dict of scratch
    # find the corresponding weights in the loaded model, and set it

    # need to do some state dict "surgery"
    for key, value in scratch_dict.items():
        if 'backbone' in key:
            # the state dicts differ by one prefix, '.model', so we add that
            key_loaded = 'model.' + key
            # breakpoint()
            # need to add an extra ".layer" in key
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            try:
                scratch_dict[key] = pretrained_dict[key_loaded]
            except:
                raise Exception('key mismatch in the state dicts!')

    # scratch_dict has been updated
    return scratch_dict

def load_model(pretrained_model_name = 'hyenadna-small-32k-seqlen',
               use_padding = True,
               rc_aug = False,
               add_eos = False,
               use_head = False,
               n_classes = 2,
               backbone_cfg = None,
               download=False):
    
    max_lengths = {
        'hyenadna-tiny-1k-seqlen': 1024,
        'hyenadna-small-32k-seqlen': 32768,
        'hyenadna-medium-160k-seqlen': 160000,
        'hyenadna-medium-450k-seqlen': 450000,  # T4 up to here
        'hyenadna-large-1m-seqlen': 1_000_000,  # only A100 (paid tier)
    }

    max_length = max_lengths[pretrained_model_name]  # auto selects

    # data settings:
    # rc_aug = False  # reverse complement augmentation
    # add_eos = False  # add end of sentence token

    # we need these for the decoder head, if using
    # use_head = False
    # n_classes = 2  # not used for embeddings only

    # you can override with your own backbone config here if you want,
    # otherwise we'll load the HF one in None
    # backbone_cfg = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # instantiate the model (pretrained here)
    if pretrained_model_name in ['hyenadna-tiny-1k-seqlen',
                                 'hyenadna-small-32k-seqlen',
                                 'hyenadna-medium-160k-seqlen',
                                 'hyenadna-medium-450k-seqlen',
                                 'hyenadna-large-1m-seqlen']:
        # use the pretrained Huggingface wrapper instead
        model = HyenaDNAPreTrainedModel.from_pretrained(
            './checkpoints',
            pretrained_model_name,
            download=download,
            config=backbone_cfg,
            device=device,
            use_head=use_head,
            n_classes=n_classes,
        )

    # from scratch
    elif pretrained_model_name is None:
        model = HyenaDNAModel(**backbone_cfg, use_head=use_head, n_classes=n_classes)
        
    return model, max_length
        

def get_tokenizer(max_length):
    return CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
        model_max_length=max_length + 2,  # to account for special tokens, like EOS
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side='left', # since HyenaDNA is causal, we pad on the left
    )

def tokenize_sample(tokenizer, miRNA_seq, mRNA_seq):
    
    return tokenizer(miRNA_seq)["input_ids"] + tokenizer(mRNA_seq)["input_ids"][1:]

def load_data(filepath):
    return pd.read_csv(filepath)

def embedd_sample(model, tok_seq, device):
    
    tok_seq = torch.LongTensor(tok_seq).unsqueeze(0)
    tok_seq = tok_seq.to(device)

    # prep model and forward
    model.to(device)
    model.eval()
    with torch.inference_mode():
        embeddings = model(tok_seq)
        
    # Convert the tensor to float if it's not already
    sequence_embedding = embeddings.float()

    # Mean pooling along the sequence length dimension (axis=1)
    mean_pooled_embedding = torch.mean(sequence_embedding, dim=1)
        
    # extracting embeddings as a simple list of values
    return list(mean_pooled_embedding.detach().cpu().numpy()[0])

def save_result(df, filepath):
    df.to_csv(filepath, index=False)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filepath', type=str,
                        help='Path to input file with data')
    parser.add_argument('--output_filepath', type=str,
                        help='Optional path to output file.'
                        'Default - the same as input, only adding \'embedd\' to the filename.')
    args = parser.parse_args()
    
    if args.output_filepath is None:
        args.output_filepath = args.input_filepath + ".embedd"
    
    model, max_length = load_model()
    tokenizer = get_tokenizer(max_length)
    data = load_data(args.input_filepath)
    
    tqdm.pandas()
    data['tokenized'] = data.progress_apply(lambda x: tokenize_sample(tokenizer, x['miRNA'], x['utr3']), axis=1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    
    # TODO: maybe try batches?
    data['embedded'] = data.progress_apply(lambda x: embedd_sample(model, x['tokenized'], device), axis=1)
    
    # converting list to string on purpose, so we can control saving and then loading from the file
    data["tokenized"] = data["tokenized"].apply(lambda x: ','.join(str(i) for i in x))
    data["embedded"] = data["embedded"].apply(lambda x: ','.join(str(i) for i in x))
    
    save_result(data, args.output_filepath)
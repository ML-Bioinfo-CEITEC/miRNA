import torch
from sklearn.metrics import precision_score, recall_score, f1_score


class miRNA_Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        miRNAs,
        mRNAs,
        labels,
        max_length,
        d_output=2, # default binary classification
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False
    ):

        self.miRNAs = miRNAs
        self.mRNAs = mRNAs
        self.labels = labels
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        
        assert len(self.miRNAs) == len(self.mRNAs)
        assert len(self.miRNAs) == len(self.labels)

    def __len__(self):
        return len(self.miRNAs)

    def __getitem__(self, idx):
        
        x = self.miRNAs[idx] + "NNNNN" + self.mRNAs[idx]
        y = self.labels[idx]

        seq = self.tokenizer(x,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else None,
            max_length=self.max_length,
            truncation=True,
        )  # add cls and eos token (+2)
        seq = seq["input_ids"]  # get input_ids

        # need to handle eos here
        if self.add_eos:
            # append list seems to be faster than append tensor
            seq.append(self.tokenizer.sep_token_id)

        # convert to tensor
        seq = torch.LongTensor(seq)

        # need to wrap in list
        target = torch.LongTensor([y])

        return seq, target
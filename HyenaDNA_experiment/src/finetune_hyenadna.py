import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from hyenadna_helpers import miRNA_Dataset
import torch.nn as nn
from hyenadna_helpers import train, test_metrics
import torch.optim as optim


model_path = "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf"
# loss function
loss_fn = nn.CrossEntropyLoss()
learning_rate = 6e-4  # good default for Hyena
weight_decay = 0.1
num_epochs = 3

# revision is a git commit hash of version of used model, can be found here: https://huggingface.co/LongSafari/hyenadna-tiny-16k-seqlen-d128-hf/commits/main
model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True, revision="e83c7caa155780f5f898017e736c3f6041e559cf")
data = pd.read_csv('../debug/3utr.sequences.refseq_id.mirna_fc.pkl.nlp.clamped_to_zero.cls.seq_len_below.balanced_down')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# add pad token id to model config - it is not there by default and CLS models meeds it
model.config.pad_token_id = tokenizer._vocab_str_to_int[tokenizer.pad_token]

train_df, test_df = train_test_split(data, test_size=0.2)

use_padding = True
batch_size = 16 # lover batch_size to fit sequences into memory
max_length = tokenizer.model_max_length

# create datasets
ds_train = miRNA_Dataset(
    train_df['miRNA'].reset_index(drop=True),
    train_df['utr3'].reset_index(drop=True),
    train_df['cls_label'].reset_index(drop=True),
    max_length = max_length,
    use_padding = use_padding,
    tokenizer=tokenizer,
    add_eos=False,
)

ds_test = miRNA_Dataset(
    test_df['miRNA'].reset_index(drop=True),
    test_df['utr3'].reset_index(drop=True),
    test_df['cls_label'].reset_index(drop=True),
    max_length = max_length,
    use_padding = use_padding,
    tokenizer=tokenizer,
    add_eos=False,
)

train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

# create optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

for epoch in range(num_epochs):
    train(model, device, train_loader, optimizer, epoch, loss_fn, log_interval=200)
    test_metrics(model, device, test_loader, loss_fn)
    optimizer.step()
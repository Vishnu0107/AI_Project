import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer 
from tokenizers.pre_tokenizers import Whitespace # Here the sentence is seperated based on blank spaces
from pathlib import Path

# This function gets the sentence from a particular language
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang): # takes the configuration , the dataset and the language which we are translating
    # config['tokenizer file'] = '../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) # unknown token replaces the words which are not in the vocabulary
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
        # UNK token is used when the word is not in the vocabulary
        # PAD token is used when the paddings are added to the input
        # SOS token is used when the start of the sentence is added
        # EOS token is used when the end of the sentence is added
        # If the word needs to be added to the vocabulary the we need the frequency of the word to be greater than or equal to the min_frequency
        # Else it will be added to the special tokens
        tokenizer.train_from_iterator(get_or_build_tokenizer(ds, lang), trainer = trainer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer 

def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}', split = "train")

    # Build the tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train__ds_raw, val_ds_raw = random_split(ds_raw, {train_ds_size, val_ds_size})

    train_ds = BilingualDataset(train__ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f'Max length of source sentence is {max_len_src}')
    print(f'Max length of target sentence is {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len , vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model



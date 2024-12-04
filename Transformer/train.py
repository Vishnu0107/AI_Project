import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizer.trainers import WordLevelTrainer 
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
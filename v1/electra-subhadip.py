#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.nn as nn
import torch
import torch.optim as optim
import re
import glob
import os.path
import sys
import transformers
from random import randrange, random, shuffle, randint
import numpy as np
import math
import statistics
from datetime import datetime
from pathlib import Path
import logging
# import logging
import tqdm


# In[2]:

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.environ["WANDB_DISABLED"] = "True"


# In[3]:


homeDir = "/home/bangla/Bert/"
# homeDir = "/Users/joy/Desktop/phd_work/Bert/"
Path(homeDir).mkdir(parents=True, exist_ok=True)

dataDir = "/home/bangla/data/"
# dataDir = "/Users/joy/Desktop/phd_work/data/"
relativeDir = "Bangla_Data/"


# In[4]:


# Load the tokenizer
from transformers import BertTokenizer

vocab_file_dir = homeDir+'tokenizer-vocab.txt' 

tokenizer_bangla = BertTokenizer.from_pretrained(vocab_file_dir)

sentence = 'শেষ দিকে সেনাবাহিনীর সদস্যরা এসব ঘর তাঁর প্রশাসনের কাছে হস্তান্তর করেন'

encoded_input = tokenizer_bangla.tokenize(sentence)
print(encoded_input)


# In[5]:


from transformers import LineByLineTextDataset
dataset= LineByLineTextDataset(
    tokenizer = tokenizer_bangla,
    file_path = "merged_news_full_lit_haf.txt",
    
    block_size = 64  # maximum sequence length
)

print('No. of lines: ' +str(len(dataset))) # No of lines in your datset


# In[6]:


# from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import ElectraConfig, ElectraForMaskedLM, DataCollatorForLanguageModeling

# config = BertConfig(
#     vocab_size=50000,
#     hidden_size=768, 
#     num_hidden_layers=5, 
#     num_attention_heads=12,
#     max_position_embeddings=512,
    

# )

config = ElectraConfig(vocab_size=50000)
 
model = ElectraForMaskedLM(config)
device = torch.device("cuda:0")
model.to(device)
print('No of parameters: '+str(model.num_parameters()) )


# In[7]:


from transformers import Trainer, TrainingArguments
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer_bangla,mlm=True, mlm_probability=0.15)


# In[19]:


# training_args = TrainingArguments(
#     output_dir=homeDir+'output-electra/',
#     overwrite_output_dir=True,
#     num_train_epochs=2,
#     per_device_train_batch_size=2,
#     save_steps=10_000,
#     save_total_limit=2,
#     prediction_loss_only=True,
#     # logging_dir=homeDir+'logs/',
#     log_on_each_node=True,
    
#     # log_level=transformers.utils.logging.DEBUG,
    
    
# )

training_args = TrainingArguments(output_dir="./output-electra-3-litfractionedhalf/", overwrite_output_dir=True, 
                                  do_train=True, 
                                  do_eval=False, per_device_train_batch_size=128, 
                                  num_train_epochs=8, log_level="info", 
                                  logging_dir="./logs-electra-3-newsfractioned/", logging_strategy="epoch", 
                                  save_strategy="epoch", 
                                  save_total_limit=15)


# In[22]:


from transformers import EarlyStoppingCallback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)


# In[23]:


trainer.train()


# In[ ]:





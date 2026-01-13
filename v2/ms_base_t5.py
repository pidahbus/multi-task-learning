#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.system("pip install transformers")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


# In[16]:


# import libraries
import gc
from random import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer
# from pandarallel import pandarallel

# pandarallel.initialize(nb_workers=12)
# pd.set_option('max_colwidth', 400)


# In[17]:


# config
SM_CHANNEL_TRAIN = "/opt/ml/input/data/train"
SM_MODEL_DIR = "/opt/ml/model"

# SM_CHANNEL_TRAIN = "pretrain_data/"
# SM_MODEL_DIR = "/opt/ml/model"


# In[18]:


# positional encoding
def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
    
    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)
    
    pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 
    
    return tf.cast(pos_encoding, dtype=tf.float32)


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True, name="shared_embedding") 
        self.output_bias = self.add_weight(shape=(vocab_size,), initializer="zeros", trainable=True, name="output_bias")
    
    def call(self, x, embedding_type):
        if embedding_type == "input":
            x = self.embedding(x)
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            return x
        elif embedding_type == "output":
            x = tf.matmul(x, self.embedding.embeddings, transpose_b=True)
            return x + self.output_bias
        else:
            raise ValueError("embedding_type must be 'input' or 'output'")
        


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pos_encoding = positional_encoding(length=max_len, depth=d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


# In[19]:


# attention
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)
        
        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores
        
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        
        return x

class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask = True)

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


# In[20]:


# feedforward
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
          tf.keras.layers.Dense(dff, activation='relu'),
          tf.keras.layers.Dense(d_model),
          tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        
    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x


# In[21]:


# encoder
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model//num_heads,
            dropout=dropout_rate)
        
        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                 dff, dropout_rate=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        
        # Add dropout.
        x = self.dropout(x)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        
        return x  # Shape `(batch_size, seq_len, d_model)`.


# In[31]:


# decoder
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 d_model,
                 num_heads,
                 dff,
                 dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model//num_heads,
            dropout=dropout_rate)
        
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model//num_heads,
            dropout=dropout_rate)
        
        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)
        
        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores
        
        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 dropout_rate=0.1):
        super(Decoder, self).__init__()
    
        self.d_model = d_model
        self.num_layers = num_layers
    
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]
    
        self.last_attn_scores = None

    def call(self, x, context):
        x = self.dropout(x)
    
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)
    
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
    
        # The shape of x is (batch_size, target_seq_len, d_model).
        return x
     


# In[32]:


# read tokenizer
tokenizer = AutoTokenizer.from_pretrained(os.path.join(SM_CHANNEL_TRAIN, "btokenizer"))

new_spl_tokens = ["[SOS]", "[EOS]", "[SPANEND]"] + [f"[SPAN{i}]" for i in range(50)]

tokenizer.add_tokens(new_tokens=new_spl_tokens, special_tokens=True)


# In[35]:


# config
# num_layers = 12
# d_model = 768
# dff = 3072
# num_heads = 12
# dropout_rate = 0.1
# max_len = 256
# corr_prob = 0.15
# vocab_size = len(tokenizer)

num_layers = 6
d_model = 512
dff = 2048
num_heads = 8
dropout_rate = 0.1
max_len = 256
corr_prob = 0.15
vocab_size = len(tokenizer)
batch_size=512
num_epochs = 20


# In[25]:


# read data

with open(os.path.join(SM_CHANNEL_TRAIN, "raw_text.txt"), "r") as f:
    data = f.read().split("\n")

data = list(set(data))

df = pd.DataFrame({"text": data}).sample(frac=1.0, ignore_index=True)
del data
gc.collect()


# In[26]:


# batch generator
class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size, tokenizer=tokenizer, max_len=max_len, corr_prob=corr_prob):
        self.df = df
        self.batch_size = batch_size
        self.corr_prob = corr_prob
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        if len(self.df) % self.batch_size == 0:
            return len(self.df) // self.batch_size
        return len(self.df) // self.batch_size + 1

    def __create_input_and_output__(self, text):
        corr_count = 0
        text_tokens = text.split(" ")
        input_tokens = ["[SOS]"]
        output_tokens = []
        prev_token_corr = False
        for token in text_tokens:
            if np.random.random() > self.corr_prob:
                input_tokens.append(token)
                prev_token_corr = False
            else:
                if not prev_token_corr:
                    input_tokens.append(f"[SPAN{corr_count}]")
                    
                    output_tokens.append(f"[SPAN{corr_count}]")
                    corr_count += 1
                output_tokens.append(token)
                prev_token_corr = True
                
        input_tokens.append("[EOS]")
        output_tokens.append("[SPANEND]")

        return " ".join(input_tokens), " ".join(output_tokens[:-1]), " ".join(output_tokens[1:])

    def __getitem__(self, idx):
        batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        df_batch = self.df[batch_slice].copy(deep=True).reset_index(drop=True)

        df_batch[["encoder_input_text", "decoder_input_text", "decoder_output_text"]] = df_batch.apply(lambda x: self.__create_input_and_output__(x["text"]), axis=1, result_type="expand")

        df_batch = df_batch[df_batch.decoder_output_text != ""].reset_index(drop=True)
        df_batch["encoder_input_tokens"] = df_batch["encoder_input_text"].apply(lambda x: self.tokenizer(x, add_special_tokens=False, max_length=self.max_len, 
                                                                                             padding="max_length", truncation=True, return_attention_mask=False, 
                                                                                             return_token_type_ids=False)["input_ids"])

        df_batch["decoder_input_tokens"] = df_batch["decoder_input_text"].apply(lambda x: self.tokenizer(x, add_special_tokens=False, max_length=self.max_len, 
                                                                                                         padding="max_length", truncation=True, return_attention_mask=False, 
                                                                                                         return_token_type_ids=False)["input_ids"])

        df_batch["decoder_output_tokens"] = df_batch["decoder_output_text"].apply(lambda x: self.tokenizer(x, add_special_tokens=False, max_length=self.max_len, 
                                                                                                           padding="max_length", truncation=True, return_attention_mask=False, 
                                                                                                           return_token_type_ids=False)["input_ids"])

        encoder_input_array = np.array(df_batch["encoder_input_tokens"].tolist())
        decoder_input_array = np.array(df_batch["decoder_input_tokens"].tolist())
        decoder_output_array = np.array(df_batch["decoder_output_tokens"].tolist())
        
        return (encoder_input_array, decoder_input_array), decoder_output_array


# In[27]:


# schduler
class LinearLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, df, batch_size, num_epochs, initial_lr):
        super().__init__()
    
        self.num_steps = tf.cast((df.shape[0]*1.1 // batch_size) * num_epochs, tf.float32)
        self.initial_lr = tf.cast(initial_lr, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        
    
        return self.initial_lr * (self.num_steps - step)/self.num_steps


# In[28]:


# loss and metrics
def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
    loss = loss_object(label, pred)
    
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred
    
    mask = label != 0
    
    match = match & mask
    
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)


# In[36]:


# get model
# def get_model():
#     encoder_input = tf.keras.layers.Input(shape=(max_len, ))
#     decoder_input = tf.keras.layers.Input(shape=(max_len, ))

#     encoder_output = Encoder(num_layers=num_layers, d_model=d_model,
#                              num_heads=num_heads, dff=dff,
#                              vocab_size=vocab_size,
#                              dropout_rate=dropout_rate)(encoder_input)

#     decoder_output = Decoder(num_layers=num_layers, d_model=d_model,
#                              num_heads=num_heads, dff=dff,
#                              vocab_size=vocab_size,
#                              dropout_rate=dropout_rate)(decoder_input, encoder_output)

#     y = tf.keras.layers.Dense(vocab_size)(decoder_output)

#     try:
#         # Drop the keras mask, so it doesn't scale the losses/metrics.
#         # b/250038731
#         del y._keras_mask
#     except AttributeError:
#         pass

#     model = tf.keras.models.Model([encoder_input, decoder_input], y)
#     return model

# model = get_model()
# model.summary()


def enocoder_layer_consolidated(encoder_input, embedding_layer, pos_embedding_layer, encoder_layer):
    embedding_output = embedding_layer(encoder_input, embedding_type="input")
    pos_embedding_output = pos_embedding_layer(embedding_output)
    encoder_output = encoder_layer(pos_embedding_output)
    return encoder_output

def decoder_layer_consolidated(encoder_output, decoder_input, embedding_layer, pos_embedding_layer, decoder_layer):
    embedding_output = embedding_layer(decoder_input, embedding_type="input")
    pos_embedding_output = pos_embedding_layer(embedding_output)
    decoder_output = decoder_layer(pos_embedding_output, encoder_output)
    return decoder_output


# get model
def get_model():
    ms_encoder_input = tf.keras.layers.Input(shape=(max_len, ), name="ms_encoder_input")
    ms_decoder_input = tf.keras.layers.Input(shape=(max_len, ), name="ms_decoder_input")
    
    embedding_layer = EmbeddingLayer(vocab_size, d_model)
    pos_embedding_layer = PositionalEmbedding(d_model, max_len)
    encoder_layer = Encoder(num_layers=num_layers, d_model=d_model,
                             num_heads=num_heads, dff=dff,
                             dropout_rate=dropout_rate)
    decoder_layer = Decoder(num_layers=num_layers, d_model=d_model,
                             num_heads=num_heads, dff=dff,
                             dropout_rate=dropout_rate)

    ms_encoder_output = enocoder_layer_consolidated(ms_encoder_input, embedding_layer, pos_embedding_layer, encoder_layer)
    ms_decoder_output = decoder_layer_consolidated(ms_encoder_output, ms_decoder_input, embedding_layer, pos_embedding_layer, decoder_layer)
    y_ms = embedding_layer(ms_decoder_output, embedding_type="output")
    
    try:
        # Drop the keras mask, so it doesn't scale the losses/metrics.
        # b/250038731
        del y_ms._keras_mask
    except AttributeError:
        pass

    model = tf.keras.models.Model([ms_encoder_input, ms_decoder_input], y_ms)
    return model


# In[ ]:


# compile and fit
# Create a MirroredStrategy (uses all available GPUs by default)
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    model = get_model()
    print(model.summary())
    learning_rate = LinearLRSchedule(df, batch_size, num_epochs, 5e-5)
    optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=learning_rate)
    
    model.compile(loss=masked_loss, optimizer=optimizer, metrics=masked_accuracy)


checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(SM_MODEL_DIR, "{epoch}.weights.h5"), save_weights_only=True)
logs = tf.keras.callbacks.CSVLogger(os.path.join(SM_MODEL_DIR, "logs.csv"))

gen = BatchGenerator(df, batch_size)
model.fit(gen, epochs=num_epochs, callbacks=[checkpoint, logs])

# In[ ]:





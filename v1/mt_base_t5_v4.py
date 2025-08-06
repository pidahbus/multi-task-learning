#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.system("pip install transformers")


# In[1]:


# import libraries
import os
import gc
from random import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer
# from pandarallel import pandarallel

# pandarallel.initialize(nb_workers=12)
pd.set_option('max_colwidth', 400)

print(tf.config.list_physical_devices('GPU'))


# In[2]:


# config
SM_CHANNEL_TRAIN = "/opt/ml/input/data/train"
SM_MODEL_DIR = "/opt/ml/model"


# SM_CHANNEL_TRAIN = "pretrain_data/"
# SM_MODEL_DIR = "/opt/ml/model"


# In[59]:


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
            return x
        elif embedding_type == "output":
            x = tf.matmul(x, self.embedding.embeddings, transpose_b=True)
            return x + self.output_bias
        else:
            raise ValueError("embedding_type must be 'input' or 'output'")
        


class LearnablePositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=sequence_length,
            output_dim=embedding_dim,
            embeddings_initializer="glorot_uniform"
        )

    def call(self, inputs):
        # inputs: (batch_size, sequence_length, embedding_dim) or (batch_size, sequence_length)
        # Get the sequence length from the input shape
        input_shape = tf.shape(inputs)
        seq_len = input_shape[1]
        # Create a range for positions [0, 1, ..., seq_len-1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        # Shape: (seq_len, embedding_dim)
        pos_embeddings = self.position_embeddings(positions)
        # Expand dims to broadcast over batch
        pos_embeddings = tf.expand_dims(pos_embeddings, axis=0)
        # Add to inputs (broadcast over batch dimension)
        return inputs + pos_embeddings


# In[45]:


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


# In[46]:


# feedforward
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
          tf.keras.layers.Dense(dff, activation='gelu'),
          tf.keras.layers.Dense(d_model),
          tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        
    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x


# In[82]:


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


# In[83]:


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
            key_dim=d_model,
            dropout=dropout_rate)
        
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
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
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
                 dropout_rate=0.1):
        super(Decoder, self).__init__()
    
        self.d_model = d_model
        self.num_layers = num_layers
    
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                 d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]
    
        self.last_attn_scores = None

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
    
        x = self.dropout(x)
    
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)
    
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
    
        # The shape of x is (batch_size, target_seq_len, d_model).
        return x
     


# In[49]:


# read tokenizer
tokenizer = AutoTokenizer.from_pretrained(os.path.join(SM_CHANNEL_TRAIN, "btokenizer"))

new_spl_tokens = ["[SOS]", "[EOS]", "[SPANEND]", "[MS]", "[MT]"] + [f"[SPAN{i}]" for i in range(50)]

tokenizer.add_tokens(new_tokens=new_spl_tokens, special_tokens=True)


# In[89]:


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
batch_size = 16

# In[33]:


# read data

with open(os.path.join(SM_CHANNEL_TRAIN, "raw_text.txt"), "r") as f:
    data = f.read().split("\n")

data = list(set(data))

df = pd.DataFrame({"text": data}).sample(frac=1.0, ignore_index=True)
del data
gc.collect()


# In[94]:


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

    def __create_ms_input_and_output__(self, text):
        corr_count = 0
        text_tokens = text.split(" ")
        input_tokens = ["[MS]"]
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
                
        output_tokens.append("[SPANEND]")

        return " ".join(input_tokens), " ".join(output_tokens[:-1]), " ".join(output_tokens[1:])
    
    def __create_mt_input_and_output__(self, text):
        
        tokens = self.tokenizer(text, add_special_tokens=False, max_length=self.max_len-1, padding="max_length", 
                                truncation=True, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        pad_token_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        mask_token_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        mt_encoder_input_tokens = [self.tokenizer.convert_tokens_to_ids("[MT]")]
        mt_encoder_output_tokens = [pad_token_id]
        mask_count = 0
        for tid in tokens:
            if tid != pad_token_id:
                if np.random.random() > self.corr_prob:
                    mt_encoder_input_tokens.append(tid)
                    mt_encoder_output_tokens.append(pad_token_id)
                else:
                    mt_encoder_input_tokens.append(mask_token_id)
                    mt_encoder_output_tokens.append(tid)
                    mask_count += 1
            else:
                mt_encoder_input_tokens.append(tid)
                mt_encoder_output_tokens.append(tid)
        
        if mask_count > 0:
            return mt_encoder_input_tokens, mt_encoder_output_tokens
        else:
            return mt_encoder_input_tokens, -1
            

    def __getitem__(self, idx):
        batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        df_batch_mt = self.df[batch_slice].copy(deep=True).reset_index(drop=True)
        
        df_batch_mt[["mt_encoder_input_tokens", "mt_encoder_output_tokens"]] = df_batch_mt.apply(lambda x: self.__create_mt_input_and_output__(x["text"]), axis=1, result_type="expand")
        df_batch_mt = df_batch_mt[df_batch_mt.mt_encoder_output_tokens != -1]
        
        mt_encoder_input_array = np.array(df_batch_mt["mt_encoder_input_tokens"].tolist())
        mt_encoder_output_array = np.array(df_batch_mt["mt_encoder_output_tokens"].tolist())
        
        return mt_encoder_input_array, mt_encoder_output_array


# In[68]:


# schduler
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
    
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
    
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
    
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# In[69]:


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


# In[95]:

def enocoder_layer_consolidated(encoder_input, embedding_layer, pos_embedding_layer, encoder_layer):
    embedding_output = embedding_layer(encoder_input, embedding_type="input")
    pos_embedding_output = pos_embedding_layer(embedding_output)
    encoder_output = encoder_layer(pos_embedding_output)
    return encoder_output


# get model
def get_model():
    mt_encoder_input = tf.keras.layers.Input(shape=(max_len, ), name="mt_encoder_input")
    
    embedding_layer = EmbeddingLayer(vocab_size, d_model)
    pos_embedding_layer = LearnablePositionalEmbedding(max_len, d_model)
    encoder_layer = Encoder(num_layers=num_layers, d_model=d_model,
                             num_heads=num_heads, dff=dff,
                             dropout_rate=dropout_rate)

    mt_encoder_output = enocoder_layer_consolidated(mt_encoder_input, embedding_layer, pos_embedding_layer, encoder_layer)
    y_mt = embedding_layer(mt_encoder_output, embedding_type="output")
    
    try:
        # Drop the keras mask, so it doesn't scale the losses/metrics.
        # b/250038731
        del y_mt._keras_mask
    except AttributeError:
        pass

    model = tf.keras.models.Model(mt_encoder_input, y_mt)
    return model


model = get_model()
print(model.summary())

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, amsgrad=False)

model.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=masked_accuracy)


# In[96]:


# compile and fit
# learning_rate = CustomSchedule(d_model)

# optimizer = tf.keras.optimizers.Adam(learning_rate)
# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, amsgrad=True)

# model.compile(
#     loss=masked_loss,
#     optimizer=optimizer,
#     metrics=masked_accuracy)


checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(SM_MODEL_DIR, "{epoch}.weights.h5"), save_weights_only=True)
logs = tf.keras.callbacks.CSVLogger(os.path.join(SM_MODEL_DIR, "logs.csv"))


gen = BatchGenerator(df, batch_size)
model.fit(gen, epochs=20)



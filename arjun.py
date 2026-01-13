import os
import json
import tensorflow as tf
from layers import EmbeddingLayer, PositionalEmbedding, Encoder, Decoder
from transformers import AutoTokenizer

# get model
def enocoder_layer_consolidated(encoder_input, embedding_layer, pos_embedding_layer, encoder_layer):
    embedding_output = embedding_layer(encoder_input, embedding_type="input")
    pos_embedding_output = pos_embedding_layer(embedding_output)
    encoder_output = encoder_layer(pos_embedding_output)
    return encoder_output

def decoder_layer_consolidated(encoder_output, decoder_input, embedding_layer, pos_embedding_layer, decoder_layer, gpt_mode=False):
    embedding_output = embedding_layer(decoder_input, embedding_type="input")
    pos_embedding_output = pos_embedding_layer(embedding_output)
    decoder_output = decoder_layer(pos_embedding_output, encoder_output, gpt_mode=gpt_mode)
    return decoder_output


# get model
class ArjunModel:
    def __new__(cls, config:dict=None):
        return cls.build_model(config)
    
    @classmethod
    def build_model(cls, config):
        ms_encoder_input = tf.keras.layers.Input(shape=(config["max_len"], ), name="ms_encoder_input")
        mt_encoder_input = tf.keras.layers.Input(shape=(config["max_len"], ), name="mt_encoder_input")
        cs_encoder_input = tf.keras.layers.Input(shape=(config["max_len"], ), name="cs_encoder_input")

        ms_decoder_input = tf.keras.layers.Input(shape=(config["max_len"], ), name="ms_decoder_input")
        nt_decoder_input = tf.keras.layers.Input(shape=(config["max_len"], ), name="nt_decoder_input")
        cs_decoder_input = tf.keras.layers.Input(shape=(config["max_len"], ), name="cs_decoder_input")
        
        
        embedding_layer = EmbeddingLayer(config["vocab_size"], config["d_model"])
        pos_embedding_layer = PositionalEmbedding(config["d_model"], config["max_len"])
        encoder_layer = Encoder(num_layers=config["num_layers"], d_model=config["d_model"],
                                num_heads=config["num_heads"], dff=config["dff"],
                                dropout_rate=config["dropout_rate"])
        decoder_layer = Decoder(num_layers=config["num_layers"], d_model=config["d_model"],
                                num_heads=config["num_heads"], dff=config["dff"],
                                dropout_rate=config["dropout_rate"])

        ms_encoder_output = enocoder_layer_consolidated(ms_encoder_input, embedding_layer, pos_embedding_layer, encoder_layer)
        cs_encoder_output = enocoder_layer_consolidated(cs_encoder_input, embedding_layer, pos_embedding_layer, encoder_layer)
        mt_encoder_output = enocoder_layer_consolidated(mt_encoder_input, embedding_layer, pos_embedding_layer, encoder_layer)
        
        ms_decoder_output = decoder_layer_consolidated(ms_encoder_output, ms_decoder_input, embedding_layer, pos_embedding_layer, decoder_layer)
        cs_decoder_output = decoder_layer_consolidated(cs_encoder_output, cs_decoder_input, embedding_layer, pos_embedding_layer, decoder_layer)
        nt_decoder_output = decoder_layer_consolidated(None, nt_decoder_input, embedding_layer, pos_embedding_layer, decoder_layer, gpt_mode=True)

        y_ms = tf.keras.layers.Activation("linear", name="ms_output")(embedding_layer(ms_decoder_output, embedding_type="output"))
        y_cs = tf.keras.layers.Activation("linear", name="cs_output")(embedding_layer(cs_decoder_output, embedding_type="output"))
        y_mt = tf.keras.layers.Activation("linear", name="mt_output")(embedding_layer(mt_encoder_output, embedding_type="output"))
        y_nt = tf.keras.layers.Activation("linear", name="nt_output")(embedding_layer(nt_decoder_output, embedding_type="output"))

        
        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del y_ms._keras_mask
        except AttributeError:
            pass

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del y_cs._keras_mask
        except AttributeError:
            pass

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del y_mt._keras_mask
        except AttributeError:
            pass

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del y_nt._keras_mask
        except AttributeError:
            pass

        model = tf.keras.models.Model([ms_encoder_input, ms_decoder_input, mt_encoder_input, nt_decoder_input, cs_encoder_input, cs_decoder_input], [y_ms, y_mt, y_nt, y_cs])
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        with open(os.path.join(pretrained_model_path, "model_config.json"), "r") as f:
            config = json.load(f)
        model = cls(config)
        model.load_weights(os.path.join(pretrained_model_path, "model.weights.h5"))
        return model
    

class ArjunTokenizer:
    def __init__(self):
        return None
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(pretrained_model_path))
        new_spl_tokens = ["[SOS]", "[EOS]", "[SPANEND]", "[MS]", "[MT]", "[NT]", "[CS]"] + [f"[SPAN{i}]" for i in range(50)]
        tokenizer.add_tokens(new_tokens=new_spl_tokens, special_tokens=True)
        return tokenizer

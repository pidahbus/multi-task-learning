import tensorflow as tf
import numpy as np

class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size, tokenizer, max_len, corr_prob):
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
        
    def __create_nt_input_and_output__(self, text):
        input_tokens = self.tokenizer("[NT]" + text, add_special_tokens=False, max_length=self.max_len, padding="max_length", 
                                      truncation=True, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        output_tokens = self.tokenizer(text + "[EOS]", add_special_tokens=False, max_length=self.max_len, padding="max_length", 
                                      truncation=True, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        
        return input_tokens, output_tokens

    def __create_cs_input_and_output__(self, text):
        text_tokens = text.split(" ")
        if len(text_tokens) <= 3:
            return -1, -1, -1
        split_idx = np.random.choice(range(1, len(text_tokens)-1))

        encoder_input_text = "[CS] " + " ".join(text_tokens[:split_idx])
        decoder_input_text = " ".join(text_tokens[split_idx:])
        decoder_output_text = " ".join(text_tokens[(split_idx+1):]) + " [EOS]"

        encoder_input_tokens = self.tokenizer(encoder_input_text, add_special_tokens=False, max_length=self.max_len, padding="max_length", truncation=True, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        decoder_input_tokens = self.tokenizer(decoder_input_text, add_special_tokens=False, max_length=self.max_len, padding="max_length", truncation=True, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        decoder_output_tokens = self.tokenizer(decoder_output_text, add_special_tokens=False, max_length=self.max_len, padding="max_length", truncation=True, return_attention_mask=False, return_token_type_ids=False)["input_ids"]

        return encoder_input_tokens, decoder_input_tokens, decoder_output_tokens
            

    def __getitem__(self, idx):
        batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        df_batch_ms = self.df[batch_slice].copy(deep=True).reset_index(drop=True)
        df_batch_mt = df_batch_ms.copy(deep=True)
        df_batch_nt = df_batch_ms.copy(deep=True)
        df_batch_cs = df_batch_ms.copy(deep=True)

        df_batch_ms[["ms_encoder_input_text", "ms_decoder_input_text", "ms_decoder_output_text"]] = df_batch_ms.apply(lambda x: self.__create_ms_input_and_output__(x["text"]), axis=1, result_type="expand")

        df_batch_ms = df_batch_ms[df_batch_ms.ms_decoder_output_text != ""].reset_index(drop=True)
        
        
        df_batch_ms["ms_encoder_input_tokens"] = df_batch_ms["ms_encoder_input_text"].apply(lambda x: self.tokenizer(x, add_special_tokens=False, max_length=self.max_len, 
                                                                                             padding="max_length", truncation=True, return_attention_mask=False, 
                                                                                             return_token_type_ids=False)["input_ids"])

        df_batch_ms["ms_decoder_input_tokens"] = df_batch_ms["ms_decoder_input_text"].apply(lambda x: self.tokenizer(x, add_special_tokens=False, max_length=self.max_len, 
                                                                                                         padding="max_length", truncation=True, return_attention_mask=False, 
                                                                                                         return_token_type_ids=False)["input_ids"])

        df_batch_ms["ms_decoder_output_tokens"] = df_batch_ms["ms_decoder_output_text"].apply(lambda x: self.tokenizer(x, add_special_tokens=False, max_length=self.max_len, 
                                                                                                           padding="max_length", truncation=True, return_attention_mask=False, 
                                                                                                           return_token_type_ids=False)["input_ids"])
        
        df_batch_mt[["mt_encoder_input_tokens", "mt_encoder_output_tokens"]] = df_batch_mt.apply(lambda x: self.__create_mt_input_and_output__(x["text"]), axis=1, result_type="expand")
        df_batch_mt = df_batch_mt[df_batch_mt.mt_encoder_output_tokens != -1]
        
        df_batch_nt[["nt_decoder_input_tokens", "nt_decoder_output_tokens"]] = df_batch_nt.apply(lambda x: self.__create_nt_input_and_output__(x["text"]), axis=1, result_type="expand")

        df_batch_cs[["cs_encoder_input_tokens", "cs_decoder_input_tokens", "cs_decoder_output_tokens"]] = df_batch_cs.apply(lambda x: self.__create_cs_input_and_output__(x["text"]), axis=1, result_type="expand")
        df_batch_cs = df_batch_cs[df_batch_cs.cs_decoder_output_tokens != -1]
        
        ms_encoder_input_array = np.array(df_batch_ms["ms_encoder_input_tokens"].tolist())
        ms_decoder_input_array = np.array(df_batch_ms["ms_decoder_input_tokens"].tolist())
        ms_decoder_output_array = np.array(df_batch_ms["ms_decoder_output_tokens"].tolist())
        
        mt_encoder_input_array = np.array(df_batch_mt["mt_encoder_input_tokens"].tolist())
        mt_encoder_output_array = np.array(df_batch_mt["mt_encoder_output_tokens"].tolist())
        
        nt_decoder_input_array = np.array(df_batch_nt["nt_decoder_input_tokens"].tolist())
        nt_decoder_output_array = np.array(df_batch_nt["nt_decoder_output_tokens"].tolist())

        cs_encoder_input_array = np.array(df_batch_cs["cs_encoder_input_tokens"].tolist())
        cs_decoder_input_array = np.array(df_batch_cs["cs_decoder_input_tokens"].tolist())
        cs_decoder_output_array = np.array(df_batch_cs["cs_decoder_output_tokens"].tolist())
        
        return (ms_encoder_input_array, ms_decoder_input_array, mt_encoder_input_array, nt_decoder_input_array, cs_encoder_input_array, cs_decoder_input_array), (ms_decoder_output_array, mt_encoder_output_array, nt_decoder_output_array, cs_decoder_output_array)

package com.otabuzzman.llmj;

public class GPT2Config {
    public int max_seq_len; // max sequence length, e.g. 1024
    public int vocab_size; // vocab size, e.g. 50257
    public int padded_vocab_size; // padded to e.g. %128==0, 50304
    public int num_layers; // number of layers, e.g. 12
    public int num_heads; // number of heads in attention, e.g. 12
    public int channels; // number of channels, e.g. 768
};

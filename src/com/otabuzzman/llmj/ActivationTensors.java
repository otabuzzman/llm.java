package com.otabuzzman.llmj;

public class ActivationTensors {
    public final static int NUM_ACTIVATION_TENSORS = 23;

    public final int encoded; // (B, T, C)
    public final int ln1; // (L, B, T, C)
    public final int ln1_mean; // (L, B, T)
    public final int ln1_rstd; // (L, B, T)
    public final int qkv; // (L, B, T, 3 * C)
    public final int atty; // (L, B, T, C)
    public final int preatt; // (L, B, NH, T, T)
    public final int att; // (L, B, NH, T, T)
    public final int attproj; // (L, B, T, C)
    public final int residual2; // (L, B, T, C)
    public final int ln2; // (L, B, T, C)
    public final int ln2_mean; // (L, B, T)
    public final int ln2_rstd; // (L, B, T)
    public final int fch; // (L, B, T, 4 * C)
    public final int fch_gelu; // (L, B, T, 4 * C)
    public final int fcproj; // (L, B, T, C)
    public final int residual3; // (L, B, T, C)
    public final int lnf; // (B, T, C)
    public final int lnf_mean; // (B, T)
    public final int lnf_rstd; // (B, T)
    public final int logits; // (B, T, V)
    public final int probs; // (B, T, V)
    public final int losses; // (B, T)

    public final int count;

    public ActivationTensors(GPT2Config config, int B, int T) {
        int Vp = config.padded_vocab_size;
        int C = config.channels;
        int NH = config.num_heads;
        int L = config.num_layers;
        encoded = 0; // encoded
        ln1 = encoded + B * T * C; // ln1 (size of previous plus index of this)
        ln1_mean = ln1 + L * B * T * C; // ln1_mean
        ln1_rstd = ln1_mean + L * B * T; // ln1_rstd
        qkv = ln1_rstd + L * B * T; // qkv
        atty = qkv + L * B * T * 3 * C; // atty
        preatt = atty + L * B * T * C; // preatt
        att = preatt + L * B * NH * T * T; // att
        attproj = att + L * B * NH * T * T; // attproj
        residual2 = attproj + L * B * T * C; // residual2
        ln2 = residual2 + L * B * T * C; // ln2
        ln2_mean = ln2 + L * B * T * C; // ln2_mean
        ln2_rstd = ln2_mean + L * B * T; // ln2_rstd
        fch = ln2_rstd + L * B * T; // fch
        fch_gelu = fch + L * B * T * 4 * C; // fch_gelu
        fcproj = fch_gelu + L * B * T * 4 * C; // fcproj
        residual3 = fcproj + L * B * T * C; // residual3
        lnf = residual3 + L * B * T * C; // lnf
        lnf_mean = lnf + B * T * C; // lnf_mean
        lnf_rstd = lnf_mean + B * T; // lnf_rstd
        logits = lnf_rstd + B * T; // logits
        probs = logits + B * T * Vp; // probs
        losses = probs + B * T * Vp; // losses

        count = losses + B * T;
    }
}

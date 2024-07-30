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

    public ActivationTensors(GPT2Config config, int B, int T, int C, int L, int NH) {
        int Vp = config.padded_vocab_size;
        encoded = 0; // index of this plus size of previous
        ln1 = encoded + B * T * C; // encoded (size)
        ln1_mean = ln1 + L * B * T * C; // ln1
        ln1_rstd = ln1_mean + L * B * T; // ln1_mean
        qkv = ln1_rstd + L * B * T; // ln1_rstd
        atty = qkv + L * B * T * 3 * C; // qkv
        preatt = atty + L * B * T * C; // atty
        att = preatt + L * B * NH * T * T; // preatt
        attproj = att + L * B * NH * T * T; // att
        residual2 = attproj + L * B * T * C; // attproj
        ln2 = residual2 + L * B * T * C; // residual2
        ln2_mean = ln2 + L * B * T * C; // ln2
        ln2_rstd = ln2_mean + L * B * T; // ln2_mean
        fch = ln2_rstd + L * B * T; // ln2_rstd
        fch_gelu = fch + L * B * T * 4 * C; // fch
        fcproj = fch_gelu + L * B * T * 4 * C; // fch_gelu
        residual3 = fcproj + L * B * T * C; // fcproj
        lnf = residual3 + L * B * T * C; // residual3
        lnf_mean = lnf + B * T * C; // lnf
        lnf_rstd = lnf_mean + B * T; // lnf_mean
        logits = lnf_rstd + B * T; // lnf_rstd
        probs = logits + B * T * Vp; // logits
        losses = probs + B * T * Vp; // probs

        count = losses + B * T; // losses
    }
}

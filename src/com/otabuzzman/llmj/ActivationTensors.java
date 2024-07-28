package com.otabuzzman.llmj;

public class ActivationTensors {
    public final static int NUM_ACTIVATION_TENSORS = 23;

    public int encoded; // (B, T, C)
    public int ln1; // (L, B, T, C)
    public int ln1_mean; // (L, B, T)
    public int ln1_rstd; // (L, B, T)
    public int qkv; // (L, B, T, 3*C)
    public int atty; // (L, B, T, C)
    public int preatt; // (L, B, NH, T, T)
    public int att; // (L, B, NH, T, T)
    public int attproj; // (L, B, T, C)
    public int residual2; // (L, B, T, C)
    public int ln2; // (L, B, T, C)
    public int ln2_mean; // (L, B, T)
    public int ln2_rstd; // (L, B, T)
    public int fch; // (L, B, T, 4*C)
    public int fch_gelu; // (L, B, T, 4*C)
    public int fcproj; // (L, B, T, C)
    public int residual3; // (L, B, T, C)
    public int lnf; // (B, T, C)
    public int lnf_mean; // (B, T)
    public int lnf_rstd; // (B, T)
    public int logits; // (B, T, V)
    public int probs; // (B, T, V)
    public int losses; // (B, T)

    private final int count;

    public ActivationTensors(GPT2Config config, int B, int T, int C, int L, int NH) {
        int Vp = config.padded_vocab_size;
        encoded = B * T * C; // encoded
        ln1 = L * B * T * C; // ln1
        ln1_mean = L * B * T; // ln1_mean
        ln1_rstd = L * B * T; // ln1_rstd
        qkv = L * B * T * 3*C; // qkv
        atty = L * B * T * C; // atty
        preatt = L * B * NH * T * T; // preatt
        att = L * B * NH * T * T; // att
        attproj = L * B * T * C; // attproj
        residual2 = L * B * T * C; // residual2
        ln2 = L * B * T * C; // ln2
        ln2_mean = L * B * T; // ln2_mean
        ln2_rstd = L * B * T; // ln2_rstd
        fch = L * B * T * 4*C; // fch
        fch_gelu = L * B * T * 4*C; // fch_gelu
        fcproj = L * B * T * C; // fcproj
        residual3 = L * B * T * C; // residual3
        lnf = B * T * C; // lnf
        lnf_mean = B * T; // lnf_mean
        lnf_rstd = B * T; // lnf_rstd
        logits = B * T * Vp; // logits
        probs = B * T * Vp; // probs
        losses = B * T; // losses

        count = encoded + ln1 + ln1_mean + ln1_rstd + qkv + atty + preatt + att + attproj + residual2 + ln2 + ln2_mean + ln2_rstd + fch + fch_gelu + fcproj + residual3 + lnf + lnf_mean + lnf_rstd + logits + probs + losses;
    }

    public int count() {
        return count;
    }
}

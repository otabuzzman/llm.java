package com.otabuzzman.llmj;

public class ActivationTensors {
    public final static int NUM_ACTIVATION_TENSORS = 23;

    public final int encoded; // (B, T, C)
    public final int ln1; // (L, B, T, C)
    public final int ln1_mean; // (L, B, T)
    public final int ln1_rstd; // (L, B, T)
    public final int qkv; // (L, B, T, 3*C)
    public final int atty; // (L, B, T, C)
    public final int preatt; // (L, B, NH, T, T)
    public final int att; // (L, B, NH, T, T)
    public final int attproj; // (L, B, T, C)
    public final int residual2; // (L, B, T, C)
    public final int ln2; // (L, B, T, C)
    public final int ln2_mean; // (L, B, T)
    public final int ln2_rstd; // (L, B, T)
    public final int fch; // (L, B, T, 4*C)
    public final int fch_gelu; // (L, B, T, 4*C)
    public final int fcproj; // (L, B, T, C)
    public final int residual3; // (L, B, T, C)
    public final int lnf; // (B, T, C)
    public final int lnf_mean; // (B, T)
    public final int lnf_rstd; // (B, T)
    public final int logits; // (B, T, V)
    public final int probs; // (B, T, V)
    public final int losses; // (B, T)

    public final int[] array;
    public final int count;

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

        array = new int[] { encoded, ln1, ln1_mean, ln1_rstd, qkv, atty, preatt, att, attproj, residual2, ln2, ln2_mean, ln2_rstd, fch, fch_gelu, fcproj, residual3, lnf, lnf_mean, lnf_rstd, logits, probs, losses };
        count = encoded + ln1 + ln1_mean + ln1_rstd + qkv + atty + preatt + att + attproj + residual2 + ln2 + ln2_mean + ln2_rstd + fch + fch_gelu + fcproj + residual3 + lnf + lnf_mean + lnf_rstd + logits + probs + losses;
    }
}

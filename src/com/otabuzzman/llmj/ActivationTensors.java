package com.otabuzzman.llmj;

import uk.ac.manchester.tornado.api.types.arrays.IntArray;

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

    private int B, T, C, NH;

    public ActivationTensors(GPT2Config config, int B, int T) {
        this.B = B;
        this.T = T;
        int Vp = config.padded_vocab_size;
        C = config.channels;
        NH = config.num_heads;
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

    public static class Indices {
        public static final int encoded = 0;
        public static final int ln1 = 1;
        public static final int ln1_mean = 2;
        public static final int ln1_rstd = 3;
        public static final int qkv = 4;
        public static final int atty = 5;
        public static final int preatt = 6;
        public static final int att = 7;
        public static final int attproj = 8;
        public static final int residual2 = 9;
        public static final int ln2 = 10;
        public static final int ln2_mean = 11;
        public static final int ln2_rstd = 12;
        public static final int fch = 13;
        public static final int fch_gelu = 14;
        public static final int fcproj = 15;
        public static final int residual3 = 16;
        public static final int lnf = 17;
        public static final int lnf_mean = 18;
        public static final int lnf_rstd = 19;
        public static final int logits = 20;
        public static final int probs = 21;
        public static final int losses = 22;
    }

    public void copyForLayerAtIndex(int index, IntArray tensors) {
        tensors.set(Indices.ln1, ln1 + index * B * T * C);
        tensors.set(Indices.ln1_mean, ln1_mean + index * B * T);
        tensors.set(Indices.ln1_rstd, ln1_rstd + index * B * T);
        tensors.set(Indices.qkv, qkv + index * B * T * 3 * C);
        tensors.set(Indices.atty, atty + index * B * T * C);
        tensors.set(Indices.preatt, preatt + index * B * NH * T * T);
        tensors.set(Indices.att, att + index * B * NH * T * T);
        tensors.set(Indices.attproj, attproj + index * B * T * C);
        tensors.set(Indices.residual2, residual2 + index * B * T * C);
        tensors.set(Indices.ln2, ln2 + index * B * T * C);
        tensors.set(Indices.ln2_mean, ln2_mean + index * B * T);
        tensors.set(Indices.ln2_rstd, ln2_rstd + index * B * T);
        tensors.set(Indices.fch, fch + index * B * T * 4 * C);
        tensors.set(Indices.fch_gelu, fch_gelu + index * B * T * 4 * C);
        tensors.set(Indices.fcproj, fcproj + index * B * T * C);
        tensors.set(Indices.residual3, residual3 + index * B * T * C);
    }
}

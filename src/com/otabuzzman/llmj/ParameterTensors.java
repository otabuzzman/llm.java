package com.otabuzzman.llmj;

public class ParameterTensors {
    public final static int NUM_PARAMETER_TENSORS = 16;

    public final int wte; // (V, C)
    public final int wpe; // (maxT, C)
    public final int ln1w; // (L, C)
    public final int ln1b; // (L, C)
    public final int qkvw; // (L, 3*C, C)
    public final int qkvb; // (L, 3*C)
    public final int attprojw; // (L, C, C)
    public final int attprojb; // (L, C)
    public final int ln2w; // (L, C)
    public final int ln2b; // (L, C)
    public final int fcw; // (L, 4*C, C)
    public final int fcb; // (L, 4*C)
    public final int fcprojw; // (L, C, 4*C)
    public final int fcprojb; // (L, C)
    public final int lnfw; // (C)
    public final int lnfb; // (C)

    public final int[] array;
    public final int count;

    // llm.c: fill_in_parameter_sizes(...)
    public ParameterTensors(GPT2Config config) {
        int Vp = config.padded_vocab_size;
        int C = config.channels;
        int maxT = config.max_seq_len;
        int L = config.num_layers;
        wte = Vp * C; // wte
        wpe = maxT * C; // wpe
        ln1w = L * C; // ln1w
        ln1b = L * C; // ln1b
        qkvw = L * (3 * C) * C; // qkvw
        qkvb = L * (3 * C); // qkvb
        attprojw = L * C * C; // attprojw
        attprojb = L * C; // attprojb
        ln2w = L * C; // ln2w
        ln2b = L * C; // ln2b
        fcw = L * (4 * C) * C; // fcw
        fcb = L * (4 * C); // fcb
        fcprojw = L * C * (4 * C); // fcprojw
        fcprojb = L * C; // fcprojb
        lnfw = C; // lnfw
        lnfb = C; // lnfb

        array = new int[] { wte, wpe, ln1w, ln1b, qkvw, qkvb, attprojw, attprojb, ln2w, ln2b, fcw, fcb, fcprojw, fcprojb, lnfw, lnfb };
        count = wte + wpe + ln1w + ln1b + qkvw + qkvb + attprojw + attprojb + ln2w + ln2b + fcw + fcb + fcprojw + fcprojb + lnfw + lnfb;
    }
}

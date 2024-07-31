package com.otabuzzman.llmj;

public class ParameterTensors {
    public final static int NUM_PARAMETER_TENSORS = 16;

    public final int wte; // (V, C)
    public final int wpe; // (maxT, C)
    public final int ln1w; // (L, C)
    public final int ln1b; // (L, C)
    public final int qkvw; // (L, 3 * C, C)
    public final int qkvb; // (L, 3 * C)
    public final int attprojw; // (L, C, C)
    public final int attprojb; // (L, C)
    public final int ln2w; // (L, C)
    public final int ln2b; // (L, C)
    public final int fcw; // (L, 4 * C, C)
    public final int fcb; // (L, 4 * C)
    public final int fcprojw; // (L, C, 4 * C)
    public final int fcprojb; // (L, C)
    public final int lnfw; // (C)
    public final int lnfb; // (C)

    public final int count;

    // llm.c: fill_in_parameter_sizes(...)
    public ParameterTensors(GPT2Config config) {
        int Vp = config.padded_vocab_size;
        int C = config.channels;
        int maxT = config.max_seq_len;
        int L = config.num_layers;
        wte = 0; // wte
        wpe = wte + Vp * C; // wpe (size of previous plus index of this)
        ln1w = wpe + maxT * C; // ln1w
        ln1b = ln1w + L * C; // ln1b
        qkvw = ln1b + L * C; // qkvw
        qkvb = qkvw + L * (3 * C) * C; // qkvb
        attprojw = qkvb + L * (3 * C); // attprojw
        attprojb = attprojw + L * C * C; // attprojb
        ln2w = attprojb + L * C; // ln2w
        ln2b = ln2w + L * C; // ln2b
        fcw = ln2b + L * C; // fcw
        fcb = fcw + L * (4 * C) * C; // fcb
        fcprojw = fcb + L * (4 * C); // fcprojw
        fcprojb = fcprojw + L * C * (4 * C); // fcprojb
        lnfw = fcprojb + L * C; // lnfw
        lnfb = lnfw + C; // lnfb

        count = lnfb + C;
    }
}

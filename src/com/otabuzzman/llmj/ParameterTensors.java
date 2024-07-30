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
        wte = 0; // index of this plus size of previous
        wpe = wte + Vp * C; // wte (size)
        ln1w = wpe + maxT * C; // wpe
        ln1b = ln1w + L * C; // ln1w
        qkvw = ln1b + L * C; // ln1b
        qkvb = qkvw + L * (3 * C) * C; // qkvw
        attprojw = qkvb + L * (3 * C); // qkvb
        attprojb = attprojw + L * C * C; // attprojw
        ln2w = attprojb + L * C; // attprojb
        ln2b = ln2w + L * C; // ln2w
        fcw = ln2b + L * C; // ln2b
        fcb = fcw + L * (4 * C) * C; // fcw
        fcprojw = fcb + L * (4 * C); // fcb
        fcprojb = fcprojw + L * C * (4 * C); // fcprojw
        lnfw = fcprojb + L * C; // fcprojb
        lnfb = lnfw + C; // lnfw

        count = lnfb + C; // lnfb
    }
}

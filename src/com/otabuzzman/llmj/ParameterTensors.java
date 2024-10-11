package com.otabuzzman.llmj;

import uk.ac.manchester.tornado.api.types.arrays.IntArray;

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

    private int C;

    // llm.c: fill_in_parameter_sizes(...)
    public ParameterTensors(GPT2Config config) {
        int Vp = config.padded_vocab_size;
        C = config.channels;
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

    public static class Indices {
        public static final int wte = 0;
        public static final int wpe = 1;
        public static final int ln1w = 2;
        public static final int ln1b = 3;
        public static final int qkvw = 4;
        public static final int qkvb = 5;
        public static final int attprojw = 6;
        public static final int attprojb = 7;
        public static final int ln2w = 8;
        public static final int ln2b = 9;
        public static final int fcw = 10;
        public static final int fcb = 11;
        public static final int fcprojw = 12;
        public static final int fcprojb = 13;
        public static final int lnfw = 14;
        public static final int lnfb = 15;
    }

    public void copyForLayerAtIndex(int index, IntArray tensors) {
        tensors.set(Indices.ln1w, ln1w + index * C);
        tensors.set(Indices.ln1w, ln1w + index * C);
        tensors.set(Indices.ln1b, ln1b + index * C);
        tensors.set(Indices.qkvw, qkvw + index * 3 * C * C);
        tensors.set(Indices.qkvb, qkvb + index * 3 * C);
        tensors.set(Indices.attprojw, attprojw + index * C * C);
        tensors.set(Indices.attprojb, attprojb + index * C);
        tensors.set(Indices.ln2w, ln2w + index * C);
        tensors.set(Indices.ln2b, ln2b + index * C);
        tensors.set(Indices.fcw, fcw + index * 4 * C * C);
        tensors.set(Indices.fcb, fcb + index * 4 * C);
        tensors.set(Indices.fcprojw, fcprojw + index * C * 4 * C);
        tensors.set(Indices.fcprojb, fcprojb + index * C);
    }
}

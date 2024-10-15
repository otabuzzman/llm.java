package com.otabuzzman.llmj;

import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// indirect access to weights and activations
// through tasks in transformer blocks
public class TensorIndices {
    public final static int NUM_TENSOR_INDICES = 40;

    final IntArray tensors = new IntArray(NUM_TENSOR_INDICES);

    final int residual = 0;
    // weights indices
    final int wte = 1;
    final int wpe = 2;
    final int ln1w = 3;
    final int ln1b = 4;
    final int qkvw = 5;
    final int qkvb = 6;
    final int attprojw = 7;
    final int attprojb = 8;
    final int ln2w = 9;
    final int ln2b = 10;
    final int fcw = 11;
    final int fcb = 12;
    final int fcprojw = 13;
    final int fcprojb = 14;
    final int lnfw = 15;
    final int lnfb = 16;
    // activation indices
    final int encoded = 17;
    final int ln1 = 18;
    final int ln1_mean = 19;
    final int ln1_rstd = 20;
    final int qkv = 21;
    final int atty = 22;
    final int preatt = 23;
    final int att = 24;
    final int attproj = 25;
    final int residual2 = 26;
    final int ln2 = 27;
    final int ln2_mean = 28;
    final int ln2_rstd = 29;
    final int fch = 30;
    final int fch_gelu = 31;
    final int fcproj = 32;
    final int residual3 = 33;
    final int lnf = 34;
    final int lnf_mean = 35;
    final int lnf_rstd = 36;
    final int logits = 37;
    final int probs = 38;
    final int losses = 39;

    private ParameterTensors params;
    private ActivationTensors acts;
    private int B, T, C, NH;

    public TensorIndices(ParameterTensors params, ActivationTensors acts, int B, int T, int C, int NH) {
        this.B = B;
        this.T = T;
        this.C = C;
        this.NH = NH;
        this.params = params;
        this.acts = acts;
        reset();
    }

    public void reset() {
        // init weights
        tensors.set(wte, params.wte);
        tensors.set(wpe, params.wpe);
        tensors.set(ln1w, params.ln1w);
        tensors.set(ln1b, params.ln1b);
        tensors.set(qkvw, params.qkvw);
        tensors.set(qkvb, params.qkvb);
        tensors.set(attprojw, params.attprojw);
        tensors.set(attprojb, params.attprojb);
        tensors.set(ln2w, params.ln2w);
        tensors.set(ln2b, params.ln2b);
        tensors.set(fcw, params.fcw);
        tensors.set(fcb, params.fcb);
        tensors.set(fcprojw, params.fcprojw);
        tensors.set(fcprojb, params.fcprojb);
        tensors.set(lnfw, params.lnfw);
        tensors.set(lnfb, params.lnfb);
        // init activations
        tensors.set(encoded, acts.encoded);
        tensors.set(ln1, acts.ln1);
        tensors.set(ln1_mean, acts.ln1_mean);
        tensors.set(ln1_rstd, acts.ln1_rstd);
        tensors.set(qkv, acts.qkv);
        tensors.set(atty, acts.atty);
        tensors.set(preatt, acts.preatt);
        tensors.set(att, acts.att);
        tensors.set(attproj, acts.attproj);
        tensors.set(residual2, acts.residual2);
        tensors.set(ln2, acts.ln2);
        tensors.set(ln2_mean, acts.ln2_mean);
        tensors.set(ln2_rstd, acts.ln2_rstd);
        tensors.set(fch, acts.fch);
        tensors.set(fch_gelu, acts.fch_gelu);
        tensors.set(fcproj, acts.fcproj);
        tensors.set(residual3, acts.residual3);
        tensors.set(lnf, acts.lnf);
        tensors.set(lnf_mean, acts.lnf_mean);
        tensors.set(lnf_rstd, acts.lnf_rstd);
        tensors.set(logits, acts.logits);
        tensors.set(probs, acts.probs);
        tensors.set(losses, acts.losses);
    }

    public void updateForLayer(int index) {
        tensors.set(residual, index == 0 ? acts.encoded : acts.residual3 + (index - 1) * B * T * C);
        // update weights
        tensors.set(ln1w, params.ln1w + index * C);
        tensors.set(ln1b, params.ln1b + index * C);
        tensors.set(qkvw, params.qkvw + index * 3 * C * C);
        tensors.set(qkvb, params.qkvb + index * 3 * C);
        tensors.set(attprojw, params.attprojw + index * C * C);
        tensors.set(attprojb, params.attprojb + index * C);
        tensors.set(ln2w, params.ln2w + index * C);
        tensors.set(ln2b, params.ln2b + index * C);
        tensors.set(fcw, params.fcw + index * 4 * C * C);
        tensors.set(fcb, params.fcb + index * 4 * C);
        tensors.set(fcprojw, params.fcprojw + index * C * 4 * C);
        tensors.set(fcprojb, params.fcprojb + index * C);
        // update activations
        tensors.set(ln1, acts.ln1 + index * B * T * C);
        tensors.set(ln1_mean, acts.ln1_mean + index * B * T);
        tensors.set(ln1_rstd, acts.ln1_rstd + index * B * T);
        tensors.set(qkv, acts.qkv + index * B * T * 3 * C);
        tensors.set(atty, acts.atty + index * B * T * C);
        tensors.set(preatt, acts.preatt + index * B * NH * T * T);
        tensors.set(att, acts.att + index * B * NH * T * T);
        tensors.set(attproj, acts.attproj + index * B * T * C);
        tensors.set(residual2, acts.residual2 + index * B * T * C);
        tensors.set(ln2, acts.ln2 + index * B * T * C);
        tensors.set(ln2_mean, acts.ln2_mean + index * B * T);
        tensors.set(ln2_rstd, acts.ln2_rstd + index * B * T);
        tensors.set(fch, acts.fch + index * B * T * 4 * C);
        tensors.set(fch_gelu, acts.fch_gelu + index * B * T * 4 * C);
        tensors.set(fcproj, acts.fcproj + index * B * T * C);
        tensors.set(residual3, acts.residual3 + index * B * T * C);
    }
}

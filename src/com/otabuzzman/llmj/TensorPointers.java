package com.otabuzzman.llmj;

public class TensorPointers {
    int residual;
    // weights pointers
    int wte;
    int wpe;
    int ln1w;
    int ln1b;
    int qkvw;
    int qkvb;
    int attprojw;
    int attprojb;
    int ln2w;
    int ln2b;
    int fcw;
    int fcb;
    int fcprojw;
    int fcprojb;
    int lnfw;
    int lnfb;
    // activation pointers
    int encoded;
    int ln1;
    int ln1_mean;
    int ln1_rstd;
    int qkv;
    int atty;
    int preatt;
    int att;
    int attproj;
    int residual2;
    int ln2;
    int ln2_mean;
    int ln2_rstd;
    int fch;
    int fch_gelu;
    int fcproj;
    int residual3;
    int lnf;
    int lnf_mean;
    int lnf_rstd;
    int logits;
    int probs;
    int losses;

    private ParameterTensors params;
    private ActivationTensors acts;
    private int B, T, C, NH;

    public TensorPointers(ParameterTensors params, ActivationTensors acts, int B, int T, int C, int NH) {
        this.B = B;
        this.T = T;
        this.C = C;
        this.NH = NH;
        this.params = params;
        this.acts = acts;
        reset();
    }

    public void reset() {
        residual = 0;
        // init weights
        wte = params.wte;
        wpe = params.wpe;
        ln1w = params.ln1w;
        ln1b = params.ln1b;
        qkvw = params.qkvw;
        qkvb = params.qkvb;
        attprojw = params.attprojw;
        attprojb = params.attprojb;
        ln2w = params.ln2w;
        ln2b = params.ln2b;
        fcw = params.fcw;
        fcb = params.fcb;
        fcprojw = params.fcprojw;
        fcprojb = params.fcprojb;
        lnfw = params.lnfw;
        lnfb = params.lnfb;
        // init activations
        encoded = acts.encoded;
        ln1 = acts.ln1;
        ln1_mean = acts.ln1_mean;
        ln1_rstd = acts.ln1_rstd;
        qkv = acts.qkv;
        atty = acts.atty;
        preatt = acts.preatt;
        att = acts.att;
        attproj = acts.attproj;
        residual2 = acts.residual2;
        ln2 = acts.ln2;
        ln2_mean = acts.ln2_mean;
        ln2_rstd = acts.ln2_rstd;
        fch = acts.fch;
        fch_gelu = acts.fch_gelu;
        fcproj = acts.fcproj;
        residual3 = acts.residual3;
        lnf = acts.lnf;
        lnf_mean = acts.lnf_mean;
        lnf_rstd = acts.lnf_rstd;
        logits = acts.logits;
        probs = acts.probs;
        losses = acts.losses;
    }

    public void updateForLayer(int index) {
        residual = index == 0 ? acts.encoded : acts.residual3 + (index - 1) * B * T * C;
        // update weights
        ln1w = params.ln1w + index * C;
        ln1b = params.ln1b + index * C;
        qkvw = params.qkvw + index * 3 * C * C;
        qkvb = params.qkvb + index * 3 * C;
        attprojw = params.attprojw + index * C * C;
        attprojb = params.attprojb + index * C;
        ln2w = params.ln2w + index * C;
        ln2b = params.ln2b + index * C;
        fcw = params.fcw + index * 4 * C * C;
        fcb = params.fcb + index * 4 * C;
        fcprojw = params.fcprojw + index * C * 4 * C;
        fcprojb = params.fcprojb + index * C;
        // update activations
        ln1 = acts.ln1 + index * B * T * C;
        ln1_mean = acts.ln1_mean + index * B * T;
        ln1_rstd = acts.ln1_rstd + index * B * T;
        qkv = acts.qkv + index * B * T * 3 * C;
        atty = acts.atty + index * B * T * C;
        preatt = acts.preatt + index * B * NH * T * T;
        att = acts.att + index * B * NH * T * T;
        attproj = acts.attproj + index * B * T * C;
        residual2 = acts.residual2 + index * B * T * C;
        ln2 = acts.ln2 + index * B * T * C;
        ln2_mean = acts.ln2_mean + index * B * T;
        ln2_rstd = acts.ln2_rstd + index * B * T;
        fch = acts.fch + index * B * T * 4 * C;
        fch_gelu = acts.fch_gelu + index * B * T * 4 * C;
        fcproj = acts.fcproj + index * B * T * C;
        residual3 = acts.residual3 + index * B * T * C;
    }
}

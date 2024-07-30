/*
 This file trains the GPT-2 model.
 This version is the clean, minimal, reference. As such:
 - it runs on CPU.
 - it does not make the code too complex; it is readable.
 - it does not use any processor-specific instructions, intrinsics and such.
 - it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
 There will be other versions of this code that specialize it and make it fast.
 */

package com.otabuzzman.llmj;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.rmi.UnexpectedException;

public class GPT2 {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    FloatBuffer params_memory;
    int num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    FloatBuffer grads_memory = null;
    // buffers for the AdamW optimizer
    FloatBuffer m_memory = null;
    FloatBuffer v_memory = null;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    FloatBuffer acts_memory = null;
    int num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    FloatBuffer grads_acts_memory = null;
    // other run state configuration
    int batch_size = 0; // the batch size (B) of current forward pass
    int seq_len = 0; // the sequence length (T) of current forward pass
    IntBuffer inputs = null; // the input tokens for the current forward pass
    IntBuffer targets = null; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss

    private final static float GELU_SCALING_FACTOR = (float) Math.sqrt(2.0f / Math.PI);

    // llm.c: gpt2_build_from_checkpoint(...)
    public GPT2(String checkpoint_path) throws FileNotFoundException, IOException {
        RandomAccessFile model_file = new RandomAccessFile(checkpoint_path, "r");
        int[] model_header = new int[256];
        for (int i = 0; i < 256; i++) {
            model_header[i] = Integer.reverseBytes(model_file.readInt()); // convert little-endians in file to JVM big-endians
        }
        assert(model_header[0] == 20240326) : "Bad magic in model file";
        assert(model_header[1] == 3) : "Wrong version in model file";

        // read in hyperparameters
        int maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
        config = new GPT2Config();
        config.max_seq_len = maxT = model_header[2];
        config.vocab_size = V = model_header[3];
        config.num_layers = L = model_header[4];
        config.num_heads = NH = model_header[5];
        config.channels = C = model_header[6];
        config.padded_vocab_size = Vp = model_header[7];
        System.out.println("[GPT-2]");
        System.out.println("max_seq_len: " + maxT);
        System.out.println("vocab_size: " + V);
        System.out.println("padded_vocab_size: " + Vp);
        System.out.println("num_layers: " + L);
        System.out.println("num_heads:" + NH);
        System.out.println("channels: " + C);

        // allocate space for all the parameters and read them in
        params = new ParameterTensors(config);
        
        // count the number of parameters
        num_parameters = params.count;
        System.out.println("num_parameters: " + num_parameters);

        // read in all the parameters from file
        ByteBuffer params_memory = ByteBuffer.allocate(num_parameters * 4 /*sizeof(float)*/);
        model_file.getChannel().read(params_memory);
        params_memory.order(ByteOrder.LITTLE_ENDIAN);
        params_memory.flip(); // apply byte order
        this.params_memory = params_memory.asFloatBuffer();
        model_file.close();

        // other inits
        mean_loss = -1.0f; // -1.0f will designate no loss
    }

    // -----------------------------------------------------------------
    // all the individual layers' forward and backward passes
    // B = batch_size, T = sequence_length, C = channels, V = vocab_size

    public void encoder_forward(int out, int wte, int wpe, int B, int T, int C) {
    }

    public void encoder_backward(int dwte, int dwpe, int dout, int B, int T, int C) {
    }

    public void layernorm_forward(int out, int mean, int rstd, int weight, int inp, int bias, int B, int T, int C) {
    }

    public void layernorm_backward(int dinp, int dweight, int dbias, int dout, int inp, int weight, int mean, int rstd, int B, int T, int C) {
    }

    public void matmul_forward_naive(int out, int inp, int weight, int bias, int B, int T, int C, int OC) {
    }

    public void matmul_forward(int out, int inp, int weight, int bias, int B, int T, int C, int OC) {
    }

    public void matmul_backward(int dinp, int dweight, int dbias, int dout, int inp, int weight, int B, int T, int C, int OC) {
    }

    public void attention_forward(int out, int preatt, int att, int inp, int B, int T, int C, int NH) {
    }

    public void attention_backward(int dinp, int dpreatt, int datt, int dout, int inp, int att, int B, int T, int C, int NH) {
    }

    public void gelu_forward(int out, int inp, int N) {
    }

    // we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
    // #pragma float_control(precise, on, push)
    // #if defined(__GNUC__) && !defined(__clang__)
    // __attribute__((optimize("no-finite-math-only")))
    // #endif
    public void gelu_backward(int dinp, int inp, int dout, int N) {
    }
    // #pragma float_control(pop)

    public void residual_forward(int out, float inp1, int inp2, int N) {
    }

    public void residual_backward(int dinp1, int dinp2, int dout, int N) {
    }

    public void softmax_forward(int probs, int logits, int B, int T, int V, int Vp) {
    }

    public void softmax_backward(int losses, int probs, int targets, int B, int T, int Vp) {
    }

    public void crossentropy_forward(int losses, int probs, int B, int T, int Vp) {
    }

    public void crossentropy_softmax_backward(int dlogits, int dlosses, int probs, int B, int T, int V, int Vp) {
    }

    public void forward(int[] inputs, int[] targets, int B, int T) throws UnexpectedException {
        // convenience parameters (size_t to help prevent int overflow)
        int V = config.vocab_size;
        int Vp = config.padded_vocab_size;
        int L = config.num_layers;
        int NH = config.num_heads;
        int C = config.channels;

        // validate inputs, all indices must be in the range [0, V)
        for(int i=0; i < B * T; i++) {
            int v = inputs[i];
            if (0 > v || v >= V) { throw new IndexOutOfBoundsException(); }
            if (targets != null) {
                v = targets[i];
                if (0 > v || v >= V) { throw new IndexOutOfBoundsException(); }
            }
        }
        // allocate space for all the activations if needed (done here, lazily)
        if(acts_memory == null) {
            // record the current B,T as well
            batch_size = B;
            seq_len = T;
            // and now allocate the space
            acts = new ActivationTensors(config, B, T, C, L, NH);

            num_activations = acts.count;
            System.out.println("num_activations: " + num_activations);
            acts_memory = FloatBuffer.allocate(num_activations);
            // also create memory for caching inputs and targets
            this.inputs = IntBuffer.allocate(B * T);
            this.targets = IntBuffer.allocate(B * T); // might be unused if we never have targets but it's small
        } else {
             // validate B,T are not larger than the values used at initialisation
            // (smaller B,T are okay for inference only)
            if (B > batch_size || T > seq_len) {
                throw new UnexpectedException(null);
            }
        }

        // cache the inputs/targets
        this.inputs.put(inputs);
        this.targets.put(targets);

        int residual;
        // forward pass
        encoder_forward(acts.encoded, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
        for (int l = 0 ; l < L ; l++) {
            residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;

            // get the pointers of the weights for this layer
            int l_ln1w = params.ln1w + l * C;
            int l_ln1b = params.ln1b + l * C;
            int l_qkvw = params.qkvw + l * 3 * C * C;
            int l_qkvb = params.qkvb + l * 3 * C;
            int l_attprojw = params.attprojw + l * C * C;
            int l_attprojb = params.attprojb + l * C;
            int l_ln2w = params.ln2w + l * C;
            int l_ln2b = params.ln2b + l * C;
            int l_fcw = params.fcw + l * 4 * C * C;
            int l_fcb = params.fcb + l * 4 * C;
            int l_fcprojw = params.fcprojw + l * C * 4 * C;
            int l_fcprojb = params.fcprojb + l * C;

            // get the pointers of the activations for this layer
            int l_ln1 = acts.ln1 + l * B * T * C;
            int l_ln1_mean = acts.ln1_mean + l * B * T;
            int l_ln1_rstd = acts.ln1_rstd + l * B * T;
            int l_qkv = acts.qkv + l * B * T * 3 * C;
            int l_atty = acts.atty + l * B * T * C;
            int l_preatt = acts.preatt + l * B * NH * T * T;
            int l_att = acts.att + l * B * NH * T * T;
            int l_attproj = acts.attproj + l * B * T * C;
            int l_residual2 = acts.residual2 + l * B * T * C;
            int l_ln2 = acts.ln2 + l * B * T * C;
            int l_ln2_mean = acts.ln2_mean + l * B * T;
            int l_ln2_rstd = acts.ln2_rstd + l * B * T;
            int l_fch = acts.fch + l * B * T * 4 * C;
            int l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C;
            int l_fcproj = acts.fcproj + l * B * T * C;
            int l_residual3 = acts.residual3 + l * B * T * C;

            // now do the forward pass
            layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
            matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
            attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
            matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
            residual_forward(l_residual2, residual, l_attproj, B * T * C);
            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
            matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
            gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
            matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
            residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
        }
        residual = acts.residual3 + (L - 1) * B * T * C; // last residual is in residual3
        layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
        matmul_forward(acts.logits, acts.lnf, params.wte, 0, B, T, C, Vp);
        softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

        // also forward the cross-entropy loss function if we have the targets
        if (targets != null) {
            crossentropy_forward(acts.losses, acts.probs, B, T, Vp);
            // for convenience also evaluate the mean loss
            mean_loss = 0.0f;
            for (int i = 0; i < B * T; i++) { mean_loss += acts_memory.get(acts.losses + i); }
            mean_loss /= B * T;
        } else {
            // if we don't have targets, we don't have a loss
            mean_loss = -1.0f;
        }
    }

    public void zero_grad() {
        if (grads_memory != null) {
            for (int i = 0 ; i < num_parameters ; i++) {
                grads_memory.put(i, 0.0f);
            }
        }
        if (grads_acts_memory != null) {
            for (int i = 0 ; i < num_activations ; i++) {
                grads_acts_memory.put(i, 0.0f);
            }
        }
    }

    public void backward() throws UnexpectedException {
        if (mean_loss == -1.0f) { throw new UnexpectedException(null); }

        if (grads_memory == null) {
            grads_memory = FloatBuffer.allocate(params.count);
            grads_acts_memory = FloatBuffer.allocate(acts.count);
        }

        // convenience shortcuts (and size_t to help prevent int overflow)
        int B = batch_size;
        int T = seq_len;
        int V = config.vocab_size;
        int Vp = config.padded_vocab_size;
        int L = config.num_layers;
        int NH = config.num_heads;
        int C = config.channels;

        // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
        // technically this is a small, inline backward() pass of calculating
        // total, final loss as the mean over all losses over all (B,T) positions in the batch
        float dloss_mean = 1.0f / (B * T);
        for (int i = 0 ; i < B * T ; i++) { grads_acts_memory.put(acts.losses + i, dloss_mean); }

        crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, B, T, V, Vp);
        matmul_backward(grads_acts.lnf, grads.wte, 0, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
        int residual = acts.residual3 + (L - 1) * B * T * C; // last layer's residual
        int dresidual = grads_acts.residual3 + (L - 1) * B * T * C; // write to last layer's residual
        layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

        for (int l = L - 1 ; l >= 0 ; l--) {
            residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;
            dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l - 1) * B * T * C;

            // get the pointers of the weights for this layer
            int l_ln1w = params.ln1w + l * C;
            int l_qkvw = params.qkvw + l * 3 * C * C;
            int l_attprojw = params.attprojw + l * C * C;
            int l_ln2w = params.ln2w + l * C;
            int l_fcw = params.fcw + l * 4 * C * C;
            int l_fcprojw = params.fcprojw + l * C * 4 * C;
            // get the pointers of the gradients of the weights for this layer
            int dl_ln1w = grads.ln1w + l * C;
            int dl_ln1b = grads.ln1b + l * C;
            int dl_qkvw = grads.qkvw + l * 3 * C * C;
            int dl_qkvb = grads.qkvb + l * 3 * C;
            int dl_attprojw = grads.attprojw + l * C * C;
            int dl_attprojb = grads.attprojb + l * C;
            int dl_ln2w = grads.ln2w + l * C;
            int dl_ln2b = grads.ln2b + l * C;
            int dl_fcw = grads.fcw + l * 4 * C * C;
            int dl_fcb = grads.fcb + l * 4 * C;
            int dl_fcprojw = grads.fcprojw + l * C * 4 * C;
            int dl_fcprojb = grads.fcprojb + l * C;
            // get the pointers of the activations for this layer
            int l_ln1 = acts.ln1 + l * B * T * C;
            int l_ln1_mean = acts.ln1_mean + l * B * T;
            int l_ln1_rstd = acts.ln1_rstd + l * B * T;
            int l_qkv = acts.qkv + l * B * T * 3 * C;
            int l_atty = acts.atty + l * B * T * C;
            int l_att = acts.att + l * B * NH * T * T;
            int l_residual2 = acts.residual2 + l * B * T * C;
            int l_ln2 = acts.ln2 + l * B * T * C;
            int l_ln2_mean = acts.ln2_mean + l * B * T;
            int l_ln2_rstd = acts.ln2_rstd + l * B * T;
            int l_fch = acts.fch + l * B * T * 4 * C;
            int l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C;
            // get the pointers of the gradients of the activations for this layer
            int dl_ln1 = grads_acts.ln1 + l * B * T * C;
            int dl_qkv = grads_acts.qkv + l * B * T * 3 * C;
            int dl_atty = grads_acts.atty + l * B * T * C;
            int dl_preatt = grads_acts.preatt + l * B * NH * T * T;
            int dl_att = grads_acts.att + l * B * NH * T * T;
            int dl_attproj = grads_acts.attproj + l * B * T * C;
            int dl_residual2 = grads_acts.residual2 + l * B * T * C;
            int dl_ln2 = grads_acts.ln2 + l * B * T * C;
            int dl_fch = grads_acts.fch + l * B * T * 4 * C;
            int dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4 * C;
            int dl_fcproj = grads_acts.fcproj + l * B * T * C;
            int dl_residual3 = grads_acts.residual3 + l * B * T * C;

            // backprop this layer
            residual_backward(dl_residual2, dl_fcproj, dl_residual3, B * T * C);
            matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4 * C, C);
            gelu_backward(dl_fch, l_fch, dl_fch_gelu, B * T * 4 * C);
            matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4 * C);
            layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
            residual_backward(dresidual, dl_attproj, dl_residual2, B*T*C);
            matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
            attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
            matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C);
            layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
        }
        encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, B, T, C);
    }

    public void gpt2_update(float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
        // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

        // lazily allocate the memory for m_memory and v_memory
        if (m_memory == null) {
            m_memory = FloatBuffer.allocate(num_parameters);
            v_memory = FloatBuffer.allocate(num_parameters);
        }

        for (int i = 0 ; i < num_parameters ; i++) {
            float param = params_memory.get(i);
            float grad = grads_memory.get(i);

            // update the first moment (momentum)
            float m = beta1 * m_memory.get(i) + (1.0f - beta1) * grad;
            // update the second moment (RMSprop)
            float v = beta2 * v_memory.get(i) + (1.0f - beta2) * grad * grad;
            // bias-correct both moments
            float m_hat = (float) (m / (1.0f - Math.pow(beta1, t)));
            float v_hat = (float) (v / (1.0f - Math.pow(beta2, t)));

            // update
            m_memory.put(i, m);
            v_memory.put(i, v);

            params_memory.put(i, params_memory.get(i) - learning_rate * (m_hat / ((float) Math.sqrt(v_hat) + eps) + weight_decay * param));
        }
    }
}

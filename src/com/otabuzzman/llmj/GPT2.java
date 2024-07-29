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
import java.util.Arrays;
import java.util.stream.IntStream;

public class GPT2 {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    IntBuffer param_sizes;
    FloatBuffer params_memory;
    int num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    FloatBuffer grads_memory;
    // buffers for the AdamW optimizer
    FloatBuffer m_memory;
    FloatBuffer v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    IntBuffer act_sizes;
    FloatBuffer acts_memory;
    int num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    FloatBuffer grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    IntBuffer inputs; // the input tokens for the current forward pass
    IntBuffer targets; // the target tokens for the current forward pass
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
        param_sizes = IntBuffer.wrap(params.array);
        
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
        acts_memory = null;
        grads_memory = null;
        m_memory = null;
        v_memory = null;
        grads_acts_memory = null;
        inputs = null;
        targets = null;
        batch_size = 0;
        seq_len = 0;
        mean_loss = -1.0f; // -1.0f will designate no loss
    }

    // -----------------------------------------------------------------
    // all the individual layers' forward and backward passes
    // B = batch_size, T = sequence_length, C = channels, V = vocab_size

    public void encoder_forward(int out, int inp, int wte, int wpe, int B, int T, int C) {
    }

    public void encoder_backward(int dwte, int dwpe, int dout, int inp, int B, int T, int C) {
    }

    public void layernorm_forward(int out, int mean, int rstd, int inp, int weight, int bias, int B, int T, int C) {
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

    public void residual_forward(int out, int inp1, int inp2, int N) {
    }

    public void residual_backward(int dinp1, int dinp2, int dout, int inp2, int N) {
    }

    public void softmax_forward(int probs, int logits, int B, int T, int V, int Vp) {
    }

    public void softmax_backward(int losses, int probs, int targets, int B, int T, int Vp) {
    }

    public void crossentropy_forward(int losses, int probs, int targets, int B, int T, int Vp) {
    }

    public void crossentropy_softmax_backward(int dlogits, int dlosses, int probs, int targets, int B, int T, int V, int Vp) {
    }
}

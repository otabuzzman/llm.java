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
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.rmi.UnexpectedException;
import java.util.stream.IntStream;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

public class GPT2 {
    public GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    FloatArray params_memory;
    public int num_parameters;
    // gradients of the weights
    public ParameterTensors grads;
    public FloatArray grads_memory = null;
    // buffers for the AdamW optimizer
    FloatBuffer m_memory = null;
    FloatBuffer v_memory = null;
    // the activations of the model, and their sizes
    public ActivationTensors acts;
    public FloatArray acts_memory = null;
    int num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    FloatArray grads_acts_memory = null;
    // other run state configuration
    int batch_size = 0; // the batch size (B) of current forward pass
    int seq_len = 0; // the sequence length (T) of current forward pass
    IntArray inputs = null; // the input tokens for the current forward pass
    IntArray targets = null; // the target tokens for the current forward pass
    public float mean_loss; // after a forward pass with targets, will be populated with the mean loss

    private final static float GELU_SCALING_FACTOR = (float) Math.sqrt(2.0f / Math.PI);

    private final static boolean UseVectorAPI = "true".equalsIgnoreCase(System.getProperty("UseVectorAPI", "false"));

    // https://stackoverflow.com/a/36335119 on assigning lambda to variable
    // https://stackoverflow.com/a/29220300/
    @FunctionalInterface
    public interface MatmulForward {
        void apply(int out, int inp, int weight, int bias, int B, int T, int C, int OC);
    }
   
    final MatmulForward matmulForward = this::matmul_forward_vecapi;
    // final MatmulForward matmulForward = this::matmul_forward_naive;
    // final MatmulForward matmulForward = this::matmul_forward;
   
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
        ByteBuffer _params_memory = ByteBuffer.allocateDirect(num_parameters * 4 /*sizeof(float)*/).order(ByteOrder.LITTLE_ENDIAN);
        model_file.getChannel().read(_params_memory);
        model_file.close();

        _params_memory.rewind();
        this.params_memory = FloatArray.fromSegment(MemorySegment.ofBuffer(_params_memory.asFloatBuffer()));

        // other inits
        mean_loss = -1.0f; // -1.0f will designate no loss
    }

    // -----------------------------------------------------------------
    // the individual TornadoVM layers' forward and backward passes
    // B = batch_size, T = sequence_length, C = channels, V = vocab_size

    private static void layernorm_forward(FloatArray params, FloatArray acts, IntArray pointers, int out, int mean, int rstd, int inp, int weight, int bias, int B, int T, int C) {
        // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        // both inp and out are (B,T,C) of the activations
        // mean and rstd are (B,T) buffers, to be used later in backward pass
        // at each position (b,t) of the input, the C-dimensional vector
        // of activations gets normalized, then scaled and shifted
        float eps = 1e-5f;
        for (@Parallel int b = 0 ; b < B ; b++) {
            for (@Parallel int t = 0 ; t < T ; t++) {
                // seek to the input position inp[b,t,:]
                int x = pointers.get(inp) + b * T * C + t * C;
                // calculate the mean
                float m = 0.0f;
                for (int i = 0 ; i < C ; i++) {
                    m += acts.get(x + i);
                }
                m = m / C;
                // calculate the variance (without any bias correction)
                float v = 0.0f;
                for (int i = 0 ; i < C ; i++) {
                    float xshift = acts.get(x + i) - m;
                    v += xshift * xshift;
                }
                v = v / C;
                // calculate the rstd (reciprocal standard deviation)
                float s = 1.0f / TornadoMath.sqrt(v + eps);
                // seek to the output position in out[b,t,:]
                int out_bt = pointers.get(out) + b * T * C + t * C;
                for (int i = 0 ; i < C ; i++) {
                    float n = (s * (acts.get(x + i) - m)); // normalize
                    float o = n * params.get(pointers.get(weight) + i) + params.get(pointers.get(bias) + i); // scale and shift
                    acts.set(out_bt + i, o); // write
                }
                // cache the mean and rstd for the backward pass later
                acts.set(pointers.get(mean) + b * T + t, m);
                acts.set(pointers.get(rstd) + b * T + t, s);
            }
        }
    }

    private static void matmul_forward(FloatArray params, FloatArray acts, IntArray pointers, int out, int inp, int weight, int bias, int B, int T, int C, int OC) {
        // the most naive implementation of matrix multiplication
        // this serves as an algorithmic reference, and as a fallback for
        // unfriendly input shapes inside matmul_forward(), below.
        // #pragma omp parallel for collapse(2)
        for (@Parallel int b = 0 ; b < B ; b++) {
            for (@Parallel int t = 0 ; t < T ; t++) {
                int bt = b * T + t;
                for (@Parallel int o = 0 ; o < OC ; o++) {
                    int _bias = pointers.get(bias);
                    float val = (_bias != -1) ? params.get(_bias + o) : 0.0f;
                    for (int i = 0 ; i < C ; i++) {
                        val += acts.get(pointers.get(inp) + bt * C + i) * params.get(pointers.get(weight) + o * C + i);
                    }
                    acts.set(pointers.get(out) + bt * OC + o, val);
                }
            }
        }
    }

    private static void attention_forward_1st(FloatArray acts, IntArray pointers, FloatArray handover, int preatt, int inp, int B, int T, int C, int NH) {
        // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
        // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
        // that holds the pre-attention and post-attention scores (used in backward)
        // output is (B, T, C)
        // attention is the only layer that mixes information across time
        // every other operation is applied at every (b,t) position independently
        // (and of course, no layer mixes information across batch)
        int C3 = C * 3;
        int hs = C / NH; // head size
        float scale = 1.0f / TornadoMath.sqrt(hs);
        int _inp = pointers.get(inp);

        // #pragma omp parallel for collapse(3)
        for (@Parallel int b = 0 ; b < B ; b++) {
            for (@Parallel int t = 0 ; t < T ; t++) {
                for (@Parallel int h = 0 ; h < NH ; h++) {
                    int query_t = _inp + b * T * C3 + t * C3 + h * hs;
                    int preatt_bth = pointers.get(preatt) + b * NH * T * T + h * T * T + t * T;
                    int current_bth = b * T * NH + t * NH + h;

                    // pass 1: calculate query dot key and maxval
                    float maxval = -10000.0f; // TODO something better
                    for (int t2 = 0; t2 <= t; t2++) {
                        int key_t2 = _inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                        // (query_t) dot (key_t2)
                        float val = 0.0f;
                        for (int i = 0; i < hs; i++) {
                            val += acts.get(query_t + i) * acts.get(key_t2 + i);
                        }
                        val *= scale;
                        if (val > maxval) {
                            maxval = val;
                        }

                        acts.set(preatt_bth + t2, val);
                    }
                    handover.set(current_bth, maxval);
                }
            }
        }
    }

    private static void attention_forward_2nd(FloatArray acts, IntArray pointers, FloatArray maxval, int preatt, int att, int B, int T, int C, int NH) {
        // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
        // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
        // that holds the pre-attention and post-attention scores (used in backward)
        // output is (B, T, C)
        // attention is the only layer that mixes information across time
        // every other operation is applied at every (b,t) position independently
        // (and of course, no layer mixes information across batch)

        // #pragma omp parallel for collapse(3)
        for (@Parallel int b = 0 ; b < B ; b++) {
            for (@Parallel int t = 0 ; t < T ; t++) {
                for (@Parallel int h = 0 ; h < NH ; h++) {
                    int preatt_bth = pointers.get(preatt) + b * NH * T * T + h * T * T + t * T;
                    int att_bth = pointers.get(att) + b * NH * T * T + h * T * T + t * T;
                    int current_bth = b * T * NH + t * NH + h;

                    // pass 2: calculate the exp and keep track of sum
                    // maxval is being calculated and subtracted only for numerical stability
                    float _maxval = maxval.get(current_bth);
                    float expsum = 0.0f;
                    for (int t2 = 0; t2 <= t; t2++) {
                        float expv = TornadoMath.exp(acts.get(preatt_bth + t2) - _maxval);
                        expsum += expv;
                        acts.set(att_bth + t2, expv);
                    }
                    // ERROR : clBuildProgram -> Returned: -11
                    // float expsum_inv = 0.0f;
                    // if (expsum > 0.0f || 0.0f > expsum) {
                    //     expsum_inv = 1.0f / expsum;
                    // }
                    // ERROR : clBuildProgram -> Returned: -11
                    // float expsum_inv = (expsum == 0.0f) ? 0.0f : 1.0f / expsum;
                    maxval.set(current_bth, expsum); // expsum_inv calculation moved to next task
                }
            }
        }
    }

    private static void attention_forward_3rd(FloatArray acts, IntArray pointers, FloatArray expsum, int out, int att, int inp, int B, int T, int C, int NH) {
        // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
        // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
        // that holds the pre-attention and post-attention scores (used in backward)
        // output is (B, T, C)
        // attention is the only layer that mixes information across time
        // every other operation is applied at every (b,t) position independently
        // (and of course, no layer mixes information across batch)
        int C3 = C * 3;
        int hs = C / NH; // head size

        // #pragma omp parallel for collapse(3)
        for (@Parallel int b = 0 ; b < B ; b++) {
            for (@Parallel int t = 0 ; t < T ; t++) {
                for (@Parallel int h = 0 ; h < NH ; h++) {
                    int att_bth = pointers.get(att) + b * NH * T * T + h * T * T + t * T;
                    int current_bth = b * T * NH + t * NH + h;

                    float expsum_inv = 0.0f;
                    float _expsum = expsum.get(current_bth);
                    if (_expsum > 0.0f || 0.0f > _expsum) {
                        expsum_inv = 1.0f / _expsum;
                    }

                    // pass 3: normalize to get the softmax
                    for (int t2 = 0; t2 < T; t2++) {
                        if (t2 <= t) {
                            acts.set(att_bth + t2, acts.get(att_bth + t2) * expsum_inv);
                        } else {
                            // causal attention mask. not strictly necessary to set to zero here
                            // only doing this explicitly for debugging and checking to PyTorch
                            acts.set(att_bth + t2, 0.0f);
                        }
                    }

                    // pass 4: accumulate weighted values into the output of attention
                    int out_bth = pointers.get(out) + b * T * C + t * C + h * hs;
                    for (int i = 0; i < hs; i++) { acts.set(out_bth + i, 0.0f); }
                    for (int t2 = 0; t2 <= t; t2++) {
                        int value_t2 = pointers.get(inp) + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                        float att_btht2 = acts.get(att_bth + t2);
                        for (int i = 0; i < hs; i++) {
                            acts.set(out_bth + i, acts.get(out_bth + i) + att_btht2 * acts.get(value_t2 + i));
                        }
                    }
                 }
            }
        }
    }

    private static void gelu_forward(FloatArray acts, IntArray pointers, int out, int inp, int N) {
        // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
        for (@Parallel int i = 0 ; i < N ; i++) {
            float x = acts.get(pointers.get(inp) + i);
            float cube = 0.044715f * x * x * x;
            acts.set(pointers.get(out) + i, 0.5f * x * (1.0f + TornadoMath.tanh(GELU_SCALING_FACTOR * (x + cube))));
        }
    }

    private static void residual_forward(FloatArray acts, IntArray pointers, int out, int inp1, int inp2, int N) {
        for (@Parallel int i = 0 ; i < N ; i++) {
            acts.set(pointers.get(out) + i, acts.get(pointers.get(inp1) + i) + acts.get(pointers.get(inp2) + i));
        }
    }

    public static void softmax_forward(FloatArray acts, IntArray pointers, int probs, int logits, int B, int T, int V, int Vp) {
        // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
        // input: logits is (B,T,Vp) of the unnormalized log probabilities
        // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
        // example: Vp is 50304 and V is 50257
        // #pragma omp parallel for collapse(2)
        for (@Parallel int b = 0 ; b < B ; b++) {
            for (@Parallel int t = 0 ; t < T ; t++) {
                // probs <- softmax(logits)
                int logits_bt = pointers.get(logits) + b * T * Vp + t * Vp;
                int probs_bt = pointers.get(probs) + b * T * Vp + t * Vp;

                // maxval is only calculated and subtracted for numerical stability
                float maxval = -10000.0f; // TODO something better
                for (int i = 0 ; i < V ; i++) {
                    if (acts.get(logits_bt + i) > maxval) {
                        maxval = acts.get(logits_bt + i);
                    }
                }
                float sum = 0.0f;
                for (int i = 0 ; i < V ; i++) {
                    acts.set(probs_bt + i, TornadoMath.exp(acts.get(logits_bt + i) - maxval));
                    sum += acts.get(probs_bt + i);
                }
                // note we only loop to V, leaving the padded dimensions
                for (int i = 0 ; i < V ; i++) {
                    acts.set(probs_bt + i, acts.get(probs_bt + i) / sum);
                }
                // for extra super safety we may wish to include this too,
                // forcing the probabilities here to be zero, but it shouldn't matter
                for (int i = V; i < Vp; i++) {
                    acts.set(probs_bt + i, 0.0f);
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // all the individual layers' forward and backward passes
    // B = batch_size, T = sequence_length, C = channels, V = vocab_size

    // acts, params, params
    public void encoder_forward(int out, int wte, int wpe, int B, int T, int C) {
        // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
        // inp is (B,T) of integers, holding the token ids at each (b,t) position
        // wte is (V,C) of token embeddings, short for "weight token embeddings"
        // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
        for (int b = 0 ; b < B ; b++) {
            for (int t = 0 ; t < T ; t++) {
                // seek to the output position in out[b,t,:]
                int out_bt = out + b * T * C + t * C;
                // get the index of the token at inp[b, t]
                int ix = inputs.get(b * T + t);
                // seek to the position in wte corresponding to the token
                int wte_ix = wte + ix * C;
                // seek to the position in wpe corresponding to the position
                int wpe_t = wpe + t * C;
                // add the two vectors and store the result in out[b,t,:]
                for (int i = 0 ; i < C ; i++) {
                    acts_memory.set(out_bt + i, params_memory.get(wte_ix + i) + params_memory.get(wpe_t + i));
                }
            }
        }
    }

    // grads, grads, grads_acts
    public void encoder_backward(int dwte, int dwpe, int dout, int B, int T, int C) {
        for (int b = 0 ; b < B ; b++) {
            for (int t = 0 ; t < T; t++) {
                int dout_bt = dout + b * T * C + t * C;
                int ix = inputs.get(b * T + t);
                int dwte_ix = dwte + ix * C;
                int dwpe_t = dwpe + t * C;
                for (int i = 0 ; i < C ; i++) {
                    float d = grads_acts_memory.get(dout_bt + i);
                    grads_memory.set(dwte_ix + i, grads_memory.get(dwte_ix + i) + d);
                    grads_memory.set(dwpe_t + i, grads_memory.get(dwpe_t + i) + d);
                }
            }
        }
    }

    // acts, acts, acts, acts, params, params
    public void layernorm_forward(int out, int mean, int rstd, int inp, int weight, int bias, int B, int T, int C) {
        // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        // both inp and out are (B,T,C) of the activations
        // mean and rstd are (B,T) buffers, to be used later in backward pass
        // at each position (b,t) of the input, the C-dimensional vector
        // of activations gets normalized, then scaled and shifted
        float eps = 1e-5f;
        for (int b = 0 ; b < B ; b++) {
            for (int t = 0 ; t < T ; t++) {
                // seek to the input position inp[b,t,:]
                int x = inp + b * T * C + t * C;
                // calculate the mean
                float m = 0.0f;
                for (int i = 0 ; i < C ; i++) {
                    m += acts_memory.get(x + i);
                }
                m = m / C;
                // calculate the variance (without any bias correction)
                float v = 0.0f;
                for (int i = 0 ; i < C ; i++) {
                    float xshift = acts_memory.get(x + i) - m;
                    v += xshift * xshift;
                }
                v = v / C;
                // calculate the rstd (reciprocal standard deviation)
                float s = 1.0f / TornadoMath.sqrt(v + eps);
                // seek to the output position in out[b,t,:]
                int out_bt = out + b * T * C + t * C;
                for (int i = 0 ; i < C ; i++) {
                    float n = (s * (acts_memory.get(x + i) - m)); // normalize
                    float o = n * params_memory.get(weight + i) + params_memory.get(bias + i); // scale and shift
                    acts_memory.set(out_bt + i, o); // write
                }
                // cache the mean and rstd for the backward pass later
                acts_memory.set(mean + b * T + t, m);
                acts_memory.set(rstd + b * T + t, s);
            }
        }
    }

    // grads_acts, grads, grads, grads_acts, acts, params, acts, acts
    public void layernorm_backward(int dinp, int dweight, int dbias, int dout, int inp, int weight, int mean, int rstd, int B, int T, int C) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                int dout_bt = dout + b * T * C + t * C;
                int inp_bt = inp + b * T * C + t * C;
                int dinp_bt = dinp + b * T * C + t * C;
                float mean_bt = acts_memory.get(mean + b * T + t);
                float rstd_bt = acts_memory.get(rstd + b * T + t);

                // first: two reduce operations
                float dnorm_mean = 0.0f;
                float dnorm_norm_mean = 0.0f;
                for (int i = 0; i < C; i++) {
                    float norm_bti = (acts_memory.get(inp_bt + i) - mean_bt) * rstd_bt;
                    float dnorm_i = params_memory.get(weight + i) * grads_acts_memory.get(dout_bt + i);
                    dnorm_mean += dnorm_i;
                    dnorm_norm_mean += dnorm_i * norm_bti;
                }
                dnorm_mean = dnorm_mean / C;
                dnorm_norm_mean = dnorm_norm_mean / C;

                // now iterate again and accumulate all the gradients
                for (int i = 0; i < C; i++) {
                    float norm_bti = (acts_memory.get(inp_bt + i) - mean_bt) * rstd_bt;
                    float dnorm_i = params_memory.get(weight + i) * grads_acts_memory.get(dout_bt + i);
                    // gradient contribution to bias
                    grads_memory.set(dbias + i, grads_memory.get(dbias + i) + grads_acts_memory.get(dout_bt + i));
                    // gradient contribution to weight
                    grads_memory.set(dweight + i, grads_memory.get(dweight + i) + norm_bti * grads_acts_memory.get(dout_bt + i));
                    // gradient contribution to input
                    float dval = 0.0f;
                    dval += dnorm_i; // term 1
                    dval -= dnorm_mean; // term 2
                    dval -= norm_bti * dnorm_norm_mean; // term 3
                    dval *= rstd_bt; // final scale
                    grads_acts_memory.set(dinp_bt + i, grads_acts_memory.get(dinp_bt + i) + dval);
                }
            }
        }
    }

    // acts, acts, params, params
    @SuppressWarnings("unused")
    private void matmul_forward_vecapi(int out, int inp, int weight, int bias, int B, int T, int C, int OC) {
        // Java vector API implementation of matrix multiplication
        // #pragma omp parallel for collapse(2)
        IntStream.range(0, B * T).parallel().forEach( bt -> {
            MemorySegment params_memory = this.params_memory.getSegment();
            MemorySegment acts_memory =  this.acts_memory.getSegment();
            int inp_base = inp + bt * C;
            IntStream.range(0, OC).parallel().forEach(o -> {
                int weight_base = weight + o * C;
                float val = (bias != -1) ? this.params_memory.get(bias + o) : 0.0f;
                int i = 0;
                if (UseVectorAPI) {
                    VectorSpecies<Float> species = FloatVector.SPECIES_MAX;
                    FloatVector sum0 = FloatVector.zero(species);
                    FloatVector sum1 = FloatVector.zero(species);
                    FloatVector sum2 = FloatVector.zero(species);
                    FloatVector sum3 = FloatVector.zero(species);
                    int lanes = species.length();
                    int upperBound = C - C % (4 * lanes);
                    for (; i < upperBound; i += 4 * lanes) {
                        // native memory segment `acts_memory´ modified on `this´ machine has native byte order.
                        var a0 = FloatVector.fromMemorySegment(species, acts_memory, (inp_base + i + 0 * lanes) * Float.BYTES, ByteOrder.nativeOrder());
                        var a1 = FloatVector.fromMemorySegment(species, acts_memory, (inp_base + i + 1 * lanes) * Float.BYTES, ByteOrder.nativeOrder());
                        var a2 = FloatVector.fromMemorySegment(species, acts_memory, (inp_base + i + 2 * lanes) * Float.BYTES, ByteOrder.nativeOrder());
                        var a3 = FloatVector.fromMemorySegment(species, acts_memory, (inp_base + i + 3 * lanes) * Float.BYTES, ByteOrder.nativeOrder());
                        // native memory segment `params_memory´ was explicitly loaded with little-endian data.
                        var p0 = FloatVector.fromMemorySegment(species, params_memory, (weight_base + i + 0 * lanes) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                        var p1 = FloatVector.fromMemorySegment(species, params_memory, (weight_base + i + 1 * lanes) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                        var p2 = FloatVector.fromMemorySegment(species, params_memory, (weight_base + i + 2 * lanes) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                        var p3 = FloatVector.fromMemorySegment(species, params_memory, (weight_base + i + 3 * lanes) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                        sum0 = a0.fma(p0, sum0);
                        sum1 = a1.fma(p1, sum1);
                        sum2 = a2.fma(p2, sum2);
                        sum3 = a3.fma(p3, sum3);
                    }
                    val += sum0.add(sum1).add(sum2).add(sum3).reduceLanes(VectorOperators.ADD);
                }

                // JIT compiler's auto-vectorization
                int upperBound = C & ~3;
                float[] sum = new float[4];
                for (; i < upperBound ; i += 4) {
                    sum[0] += this.acts_memory.get(inp_base + i + 0) * this.params_memory.get(weight_base + i + 0);
                    sum[1] += this.acts_memory.get(inp_base + i + 1) * this.params_memory.get(weight_base + i + 1);
                    sum[2] += this.acts_memory.get(inp_base + i + 2) * this.params_memory.get(weight_base + i + 2);
                    sum[3] += this.acts_memory.get(inp_base + i + 3) * this.params_memory.get(weight_base + i + 3);
                }
                val += sum[0] + sum[1] + sum[2] + sum[3];

                // process any tail elements
                for (; i < C ; i++) {
                    val += this.acts_memory.get(inp_base + i) * this.params_memory.get(weight_base + i);
                }
                this.acts_memory.set(out + bt * OC + o, val);
            });
        });
    }

    // acts, acts, params, params
    @SuppressWarnings("unused")
    private void matmul_forward_naive(int out, int inp, int weight, int bias, int B, int T, int C, int OC) {
        // the most naive implementation of matrix multiplication
        // this serves as an algorithmic reference, and as a fallback for
        // unfriendly input shapes inside matmul_forward(), below.
        // #pragma omp parallel for collapse(2)
        IntStream.range(0, B * T).parallel().forEach( bt -> {
            // unroll bt to b, t
            // int b = bt / T;
            // int t = bt % T;
            // int bt = b * T + t;
            for (int o = 0 ; o < OC ; o++) {
                float val = (bias != -1) ? params_memory.get(bias + o) : 0.0f;
                for (int i = 0 ; i < C ; i++) {
                    val += acts_memory.get(inp + bt * C + i) * params_memory.get(weight + o * C + i);
                }
                acts_memory.set(out + bt * OC + o, val);
            }
        });
    }

    // acts, acts, params, params
    @SuppressWarnings("unused")
    private void matmul_forward(int out, int inp, int weight, int bias, int B, int T, int C, int OC) {
        // most of the running time is spent here and in matmul_backward
        // therefore, the implementation below is very mildly optimized
        // this function is otherwise identical to that of matmul_forward_naive()
        // OC is short for "output channels"
        // inp is (B,T,C), weight is (OC, C), bias is (OC)
        // out will be (B,T,OC)

        // make sure the tiled loop will be correct or fallback to naive version
        int LOOP_UNROLL = 8;
        if (B * T % LOOP_UNROLL != 0) {
            matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
            return;
        }

        // collapse the B and T loops into one and turn it into a strided loop.
        // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
        // #pragma omp parallel for
        IntStream.range(0, B * T / LOOP_UNROLL).parallel().forEach( obt -> {
            obt *= LOOP_UNROLL;
            for (int o = 0; o < OC; o++) {
                // we'll keep LOOP_UNROLL many results in registers
                float[] result = new float[LOOP_UNROLL];
                // initialize the bias, if it exists
                for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    result[ibt] = (bias != -1) ? params_memory.get(bias + o) : 0.0f;
                }
                // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
                // the value of weight[i + o * C] and reuse it.
                // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
                for (int i = 0; i < C; i++) {
                    float w = params_memory.get(weight + i + o * C);
                    for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                        int bt = obt + ibt;
                        result[ibt] += acts_memory.get(inp + bt * C + i) * w;
                    }
                }
                // write back results to main memory
                for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    int bt = obt + ibt;
                    acts_memory.set(out + bt * OC + o, result[ibt]);
                }
            }
        });
    }

    // grads_acts, grads, grads, grads_acts, acts, params
    public void matmul_backward(int dinp, int dweight, int dbias, int dout, int inp, int weight, int B, int T, int C, int OC) {
        // most of the running time is spent here and in matmul_forward
        // this backward could be done in a single "round" of loops
        // but that doesn't afford an efficient parallelization strategy

        // backward into inp first, parallelize over B,T
        // #pragma omp parallel for collapse(2)
        IntStream.range(0, B * T).parallel().forEach( x -> {
            int b = x / T;
            int t = x % T;
            int dout_bt = dout + b * T * OC + t * OC;
            int dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0 ; o < OC ; o++) {
                int wrow = weight + o * C;
                float d = grads_acts_memory.get(dout_bt + o);
                for (int i = 0; i < C; i++) {
                    grads_acts_memory.set(dinp_bt + i, grads_acts_memory.get(dinp_bt + i) + params_memory.get(wrow + i) * d);
                }
            }
        });
        IntStream.range(0, OC).parallel().forEach( o -> {
            for (int b = 0 ; b < B ; b++) {
                for (int t = 0 ; t < T ; t++) {
                    int dout_bt = dout + b * T * OC + t * OC;
                    int inp_bt = inp + b * T * C + t * C;
                    int dwrow = dweight + o*C;
                    float d = grads_acts_memory.get(dout_bt + o);
                    if (dbias != -1) { grads_memory.set(dbias + o, grads_memory.get(dbias + o) + d); }
                    for (int i = 0 ; i < C ; i++) {
                        grads_memory.set(dwrow + i, grads_memory.get(dwrow + i) + acts_memory.get(inp_bt + i) * d);
                    }
                }
            }
        });
    }

    // acts, acts, acts, acts
    public void attention_forward(int out, int preatt, int att, int inp, int B, int T, int C, int NH) {
        // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
        // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
        // that holds the pre-attention and post-attention scores (used in backward)
        // output is (B, T, C)
        // attention is the only layer that mixes information across time
        // every other operation is applied at every (b,t) position independently
        // (and of course, no layer mixes information across batch)
        int C3 = C * 3;
        int hs = C / NH; // head size
        float scale = 1.0f / TornadoMath.sqrt(hs);

        // #pragma omp parallel for collapse(3)
        IntStream.range(0, B).parallel().forEach( b -> {
            IntStream.range(0, T).parallel().forEach( t -> {
                IntStream.range(0, NH).parallel().forEach( h -> {
                    int query_t = inp + b * T * C3 + t * C3 + h * hs;
                    int preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
                    int att_bth = att + b * NH * T * T + h * T * T + t * T;

                    // pass 1: calculate query dot key and maxval
                    float maxval = -10000.0f; // TODO something better
                    for (int t2 = 0; t2 <= t; t2++) {
                        int key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                        // (query_t) dot (key_t2)
                        float val = 0.0f;
                        for (int i = 0; i < hs; i++) {
                            val += acts_memory.get(query_t + i) * acts_memory.get(key_t2 + i);
                        }
                        val *= scale;
                        if (val > maxval) {
                            maxval = val;
                        }

                        acts_memory.set(preatt_bth + t2, val);
                    }

                    // pass 2: calculate the exp and keep track of sum
                    // maxval is being calculated and subtracted only for numerical stability
                    float expsum = 0.0f;
                    for (int t2 = 0; t2 <= t; t2++) {
                        float expv = TornadoMath.exp(acts_memory.get(preatt_bth + t2) - maxval);
                        expsum += expv;
                        acts_memory.set(att_bth + t2, expv);
                    }
                    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                    // pass 3: normalize to get the softmax
                    for (int t2 = 0; t2 < T; t2++) {
                        if (t2 <= t) {
                            acts_memory.set(att_bth + t2, acts_memory.get(att_bth + t2) * expsum_inv);
                        } else {
                            // causal attention mask. not strictly necessary to set to zero here
                            // only doing this explicitly for debugging and checking to PyTorch
                            acts_memory.set(att_bth + t2, 0.0f);
                        }
                    }

                    // pass 4: accumulate weighted values into the output of attention
                    int out_bth = out + b * T * C + t * C + h * hs;
                    for (int i = 0; i < hs; i++) { acts_memory.set(out_bth + i, 0.0f); }
                    for (int t2 = 0; t2 <= t; t2++) {
                        int value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                        float att_btht2 = acts_memory.get(att_bth + t2);
                        for (int i = 0; i < hs; i++) {
                            acts_memory.set(out_bth + i, acts_memory.get(out_bth + i) + att_btht2 * acts_memory.get(value_t2 + i));
                        }
                    }
                });
            });
        });
    }

    // grads_acts, grads_acts, grads_acts, grads_acts, acts, acts
    public void attention_backward(int dinp, int dpreatt, int datt, int dout, int inp, int att, int B, int T, int C, int NH) {
        // inp/dinp are (B, T, 3C) Q,K,V
        // att/datt/dpreatt are (B, NH, T, T)
        // dout is (B, T, C)
        int C3 = C * 3;
        int hs = C / NH; // head size
        float scale = 1.0f / TornadoMath.sqrt(hs);

        for (int b = 0 ; b < B ; b++) {
            for (int t = 0 ; t < T ; t++) {
                for (int h = 0 ; h < NH ; h++) {
                    int att_bth = att + b * NH * T * T + h * T * T + t * T;
                    int datt_bth = datt + b * NH * T * T + h * T * T + t * T;
                    int dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T;
                    int dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                    int query_t = inp + b * T * C3 + t * C3 + h * hs;

                    // backward pass 4, through the value accumulation
                    int dout_bth = dout + b * T * C + t * C + h * hs;
                    for (int t2 = 0 ; t2 <= t ; t2++) {
                        int value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
                        int dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C * 2;
                        for (int i = 0 ; i < hs ; i++) {
                            // in the forward pass this was:
                            // out_bth[i] += att_bth[t2] * value_t2[i];
                            // so now we have:
                            grads_acts_memory.set(datt_bth + t2, grads_acts_memory.get(datt_bth + t2) + acts_memory.get(value_t2 + i) * grads_acts_memory.get(dout_bth + i));
                            grads_acts_memory.set(dvalue_t2 + i, grads_acts_memory.get(dvalue_t2 + i) + acts_memory.get(att_bth + t2) * grads_acts_memory.get(dout_bth + i));
                        }
                    }

                    // backward pass 2 & 3, the softmax
                    // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                    for (int t2 = 0 ; t2 <= t ; t2++) {
                        for (int t3 = 0 ; t3 <= t ; t3++) {
                            float indicator = t2 == t3 ? 1.0f : 0.0f;
                            float local_derivative = acts_memory.get(att_bth + t2) * (indicator - acts_memory.get(att_bth + t3));
                            grads_acts_memory.set(dpreatt_bth + t3, grads_acts_memory.get(dpreatt_bth + t3) + local_derivative * grads_acts_memory.get(datt_bth + t2));
                        }
                    }

                    // backward pass 1, the query @ key matmul
                    for (int t2 = 0 ; t2 <= t ; t2++) {
                        int key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                        int dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                        for (int i = 0 ; i < hs ; i++) {
                            // in the forward pass this was:
                            // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                            // so now we have:
                            grads_acts_memory.set(dquery_t + i, grads_acts_memory.get(dquery_t + i) + acts_memory.get(key_t2 + i) * grads_acts_memory.get(dpreatt_bth + t2) * scale);
                            grads_acts_memory.set(dkey_t2 + i, grads_acts_memory.get(dkey_t2 + i) + acts_memory.get(query_t + i) * grads_acts_memory.get(dpreatt_bth + t2) * scale);
                        }
                    }
                }
            }
        }
    }

    // acts, acts
    public void gelu_forward(int out, int inp, int N) {
        // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
        for (int i = 0 ; i < N ; i++) {
            float x = acts_memory.get(inp + i);
            float cube = 0.044715f * x * x * x;
            acts_memory.set(out + i, 0.5f * x * (1.0f + TornadoMath.tanh(GELU_SCALING_FACTOR * (x + cube))));
        }
    }

    // we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
    // #pragma float_control(precise, on, push)
    // #if defined(__GNUC__) && !defined(__clang__)
    // __attribute__((optimize("no-finite-math-only")))
    // #endif
    // grads_acts, acts, grads_acts
    public void gelu_backward(int dinp, int inp, int dout, int N) {
        for (int i = 0 ; i < N ; i++) {
            float x = acts_memory.get(inp + i);
            float cube = 0.044715f * x * x * x;
            float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            float tanh_out = TornadoMath.tanh(tanh_arg);
            float coshf_out = (TornadoMath.exp(tanh_arg) + TornadoMath.exp(-tanh_arg)) / 2.0f; // due to missing TornadoMath.cosh()
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            grads_acts_memory.set(dinp + i, grads_acts_memory.get(dinp + i) + local_grad * grads_acts_memory.get(dout + i));
        }
    }
    // #pragma float_control(pop)

    // acts, acts, acts
    public void residual_forward(int out, int inp1, int inp2, int N) {
        for (int i = 0 ; i < N ; i++) {
            acts_memory.set(out + i, acts_memory.get(inp1 + i) + acts_memory.get(inp2 + i));
        }
    }

    // grads_acts, grads_acts, grads_acts
    public void residual_backward(int dinp1, int dinp2, int dout, int N) {
        for (int i = 0 ; i < N ; i++) {
            grads_acts_memory.set(dinp1 + i, grads_acts_memory.get(dinp1 + i) + grads_acts_memory.get(dout + i));
            grads_acts_memory.set(dinp2 + i, grads_acts_memory.get(dinp2 + i) + grads_acts_memory.get(dout + i));
        }
    }

    // acts, acts
    public void softmax_forward(int probs, int logits, int B, int T, int V, int Vp) {
        // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
        // input: logits is (B,T,Vp) of the unnormalized log probabilities
        // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
        // example: Vp is 50304 and V is 50257
        // #pragma omp parallel for collapse(2)
        IntStream.range(0, B).parallel().forEach( b -> {
            IntStream.range(0, T).parallel().forEach( t -> {
                // probs <- softmax(logits)
                int logits_bt = logits + b * T * Vp + t * Vp;
                int probs_bt = probs + b * T * Vp + t * Vp;

                // maxval is only calculated and subtracted for numerical stability
                float maxval = -10000.0f; // TODO something better
                for (int i = 0 ; i < V ; i++) {
                    if (acts_memory.get(logits_bt + i) > maxval) {
                        maxval = acts_memory.get(logits_bt + i);
                    }
                }
                float sum = 0.0f;
                for (int i = 0 ; i < V ; i++) {
                    acts_memory.set(probs_bt + i, TornadoMath.exp(acts_memory.get(logits_bt + i) - maxval));
                    sum += acts_memory.get(probs_bt + i);
                }
                // note we only loop to V, leaving the padded dimensions
                for (int i = 0 ; i < V ; i++) {
                    acts_memory.set(probs_bt + i, acts_memory.get(probs_bt + i) / sum);
                }
                // for extra super safety we may wish to include this too,
                // forcing the probabilities here to be zero, but it shouldn't matter
                for (int i = V; i < Vp; i++) {
                    acts_memory.set(probs_bt + i, 0.0f);
                }
            });
        });
    }

    // acts, acts
    public void crossentropy_forward(int losses, int probs, int B, int T, int Vp) {
        // output: losses is (B,T) of the individual losses at each position
        // input: probs are (B,T,Vp) of the probabilities
        // input: targets is (B,T) of integers giving the correct index in logits
        for (int b = 0 ; b < B ; b++) {
            for (int t = 0 ; t < T ; t++) {
                // loss = -log(probs[target])
                int probs_bt = probs + b * T * Vp + t * Vp;
                int ix = targets.get(b * T + t);
                acts_memory.set(losses + b * T + t, -TornadoMath.log(acts_memory.get(probs_bt + ix)));
            }
        }
    }

    // grads_acts, grads_acts, acts
    public void crossentropy_softmax_backward(int dlogits, int dlosses, int probs, int B, int T, int V, int Vp) {
        // backwards through both softmax and crossentropy
        for (int b = 0 ; b < B ; b++) {
            for (int t = 0 ; t < T ; t++) {
                int dlogits_bt = dlogits + b * T * Vp + t * Vp;
                int probs_bt = probs + b * T * Vp + t * Vp;
                float dloss = grads_acts_memory.get(dlosses + b * T + t);
                int ix = targets.get(b * T + t);
                // note we only loop to V, leaving the padded dimensions
                // of dlogits untouched, so gradient there stays at zero
                for (int i = 0 ; i < V ; i++) {
                    float p = acts_memory.get(probs_bt + i);
                    float indicator = i == ix ? 1.0f : 0.0f;
                    grads_acts_memory.set(dlogits_bt + i, grads_acts_memory.get(dlogits_bt + i) + (p - indicator) * dloss);
                }
            }
        }
    }

    public void forward(IntBuffer inputs, IntBuffer targets, int B, int T) throws UnexpectedException {
        // convenience parameters (size_t to help prevent int overflow)
        int V = config.vocab_size;
        int Vp = config.padded_vocab_size;
        int L = config.num_layers;
        int NH = config.num_heads;
        int C = config.channels;

        // validate inputs, all indices must be in the range [0, V)
        for(int i=0 ; i < B * T ; i++) {
            int v = inputs.get(i);
            if (0 > v || v >= V) { throw new IndexOutOfBoundsException(); }
            if (targets != null) {
                v = targets.get(i);
                if (0 > v || v >= V) { throw new IndexOutOfBoundsException(); }
            }
        }
        // allocate space for all the activations if needed (done here, lazily)
        if (acts_memory == null) {
            // record the current B,T as well
            batch_size = B;
            seq_len = T;
            // and now allocate the space
            acts = new ActivationTensors(config, B, T);

            num_activations = acts.count;
            System.out.println("num_activations: " + num_activations);
            acts_memory = FloatArray.fromArray(new float[num_activations]);
            // also create memory for caching inputs and targets
            this.inputs = IntArray.fromArray(new int[B * T]);
            this.targets = IntArray.fromArray(new int[B * T]); // might be unused if we never have targets but it's small
        } else {
            // validate B,T are not larger than the values used at initialisation
            // (smaller B,T are okay for inference only)
            if (B > batch_size || T > seq_len) {
                throw new UnexpectedException(null);
            }
        }

        // cache the inputs/targets
        for (int i = 0 ; i < inputs.capacity() ; i++) this.inputs.set(i, inputs.get(i));
        if (targets != null ) {
            for (int i = 0 ; i < targets.capacity() ; i++) this.targets.set(i, targets.get(i));
        }

        IntArray pointers = new IntArray(40); // weights and activations pointers buffer
        int residual = 0;
        // weights indices
        int o_wte = 1;
        int l_ln1w = 2;
        int l_ln1b = 3;
        int l_qkvw = 4;
        int l_qkvb = 5;
        int l_attprojw = 6;
        int l_attprojb = 7;
        int l_ln2w = 8;
        int l_ln2b = 9;
        int l_fcw = 10;
        int l_fcb = 11;
        int l_fcprojw = 12;
        int l_fcprojb = 13;
        int o_lnfw = 14;
        int o_lnfb = 15;
        // weights indices
        int l_ln1 = 16;
        int l_ln1_mean = 17;
        int l_ln1_rstd = 18;
        int l_qkv = 19;
        int l_atty = 20;
        int l_preatt = 21;
        int l_att = 22;
        int l_attproj = 23;
        int l_residual2 = 24;
        int l_ln2 = 25;
        int l_ln2_mean = 26;
        int l_ln2_rstd = 27;
        int l_fch = 28;
        int l_fch_gelu = 29;
        int l_fcproj = 30;
        int l_residual3 = 31;
        int o_lnf = 32;
        int o_lnf_mean = 33;
        int o_lnf_rstd = 34;
        int o_logits = 35;
        int o_probs = 36;
        int o_losses = 37;

        FloatArray handover = new FloatArray(B * T * NH); // to forward data in attention layers

        TaskGraph transformer_block = new TaskGraph("s0")
        .transferToDevice(DataTransferMode.FIRST_EXECUTION, params_memory)
        .transferToDevice(DataTransferMode.EVERY_EXECUTION, pointers)
        .task("t0", GPT2::layernorm_forward, params_memory, acts_memory, pointers, l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C)
        .task("t1", GPT2::matmul_forward, params_memory, acts_memory, pointers, l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C)
        .task("t2", GPT2::attention_forward_1st, acts_memory, pointers, handover, l_preatt, l_qkv, B, T, C, NH)
        .task("t3", GPT2::attention_forward_2nd, acts_memory, pointers, handover, l_preatt, l_att, B, T, C, NH)
        .task("t4", GPT2::attention_forward_3rd, acts_memory, pointers, handover, l_atty, l_att, l_qkv, B, T, C, NH)
        .task("t5", GPT2::matmul_forward, params_memory, acts_memory, pointers, l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C)
        .task("t6", GPT2::residual_forward, acts_memory, pointers, l_residual2, residual, l_attproj, B * T * C)
        .task("t7", GPT2::layernorm_forward, params_memory, acts_memory, pointers, l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C)
        .task("t8", GPT2::matmul_forward, params_memory, acts_memory, pointers, l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C)
        .task("t9", GPT2::gelu_forward, acts_memory, pointers, l_fch_gelu, l_fch, B * T * 4 * C)
        .task("t10", GPT2::matmul_forward, params_memory, acts_memory, pointers, l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C)
        .task("t11", GPT2::residual_forward, acts_memory, pointers, l_residual3, l_residual2, l_fcproj, B * T * C)
        .transferToHost(DataTransferMode.UNDER_DEMAND, acts_memory);
        TornadoExecutionPlan transformer_runner = new TornadoExecutionPlan(transformer_block.snapshot());
        TornadoExecutionResult transformer_result = null;

        // forward pass
        long t1 = System.currentTimeMillis();
        encoder_forward(acts.encoded, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
        for (int l = 0 ; l < L ; l++) {
            pointers.set(residual, l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C);

            // get the pointers of the weights for this layer
            pointers.set(l_ln1w, params.ln1w + l * C);
            pointers.set(l_ln1b, params.ln1b + l * C);
            pointers.set(l_qkvw, params.qkvw + l * 3 * C * C);
            pointers.set(l_qkvb, params.qkvb + l * 3 * C);
            pointers.set(l_attprojw, params.attprojw + l * C * C);
            pointers.set(l_attprojb, params.attprojb + l * C);
            pointers.set(l_ln2w, params.ln2w + l * C);
            pointers.set(l_ln2b, params.ln2b + l * C);
            pointers.set(l_fcw, params.fcw + l * 4 * C * C);
            pointers.set(l_fcb, params.fcb + l * 4 * C);
            pointers.set(l_fcprojw, params.fcprojw + l * C * 4 * C);
            pointers.set(l_fcprojb, params.fcprojb + l * C);

            // get the pointers of the activations for this layer
            pointers.set(l_ln1, acts.ln1 + l * B * T * C);
            pointers.set(l_ln1_mean, acts.ln1_mean + l * B * T);
            pointers.set(l_ln1_rstd, acts.ln1_rstd + l * B * T);
            pointers.set(l_qkv, acts.qkv + l * B * T * 3 * C);
            pointers.set(l_atty, acts.atty + l * B * T * C);
            pointers.set(l_preatt, acts.preatt + l * B * NH * T * T);
            pointers.set(l_att, acts.att + l * B * NH * T * T);
            pointers.set(l_attproj, acts.attproj + l * B * T * C);
            pointers.set(l_residual2, acts.residual2 + l * B * T * C);
            pointers.set(l_ln2, acts.ln2 + l * B * T * C);
            pointers.set(l_ln2_mean, acts.ln2_mean + l * B * T);
            pointers.set(l_ln2_rstd, acts.ln2_rstd + l * B * T);
            pointers.set(l_fch, acts.fch + l * B * T * 4 * C);
            pointers.set(l_fch_gelu, acts.fch_gelu + l * B * T * 4 * C);
            pointers.set(l_fcproj, acts.fcproj + l * B * T * C);
            pointers.set(l_residual3, acts.residual3 + l * B * T * C);

            // now do the forward pass
            transformer_result = transformer_runner.execute();
        }
        transformer_result.transferToHost(acts_memory);
        try { transformer_runner.close(); } catch (TornadoExecutionPlanException e) { throw new UnexpectedException(null); }

        TaskGraph output_layer = new TaskGraph("s0")
        .transferToDevice(DataTransferMode.EVERY_EXECUTION, params_memory, pointers) // only one execution here
        .task("t0", GPT2::layernorm_forward, params_memory, acts_memory, pointers, o_lnf, o_lnf_mean, o_lnf_rstd, residual, o_lnfw, o_lnfb, B, T, C)
        .task("t1", GPT2::matmul_forward, params_memory, acts_memory, pointers, o_logits, o_lnf, o_wte, l_qkvb, B, T, C, Vp)
        .task("t2", GPT2::softmax_forward, acts_memory, pointers, o_probs, o_logits, B, T, V, Vp)
        .transferToHost(DataTransferMode.EVERY_EXECUTION, acts_memory);
        TornadoExecutionPlan output_runner = new TornadoExecutionPlan(output_layer.snapshot());

        pointers.set(residual, acts.residual3 + (L - 1) * B * T * C); // last residual is in residual3
        pointers.set(o_wte, params.wte);
        pointers.set(l_qkvb, -1);
        pointers.set(o_lnfw, params.lnfw);
        pointers.set(o_lnfb, params.lnfb);
        pointers.set(o_lnf, acts.lnf);
        pointers.set(o_lnf_mean, acts.lnf_mean);
        pointers.set(o_lnf_rstd, acts.lnf_rstd);
        pointers.set(o_logits, acts.logits);
        pointers.set(o_probs, acts.probs);

        output_runner.execute();
        try { output_runner.close(); } catch (TornadoExecutionPlanException e) { throw new UnexpectedException(null); }

        // also forward the cross-entropy loss function if we have the targets
        if (targets != null) {
            crossentropy_forward(acts.losses, acts.probs, B, T, Vp);
            // for convenience also evaluate the mean loss
            mean_loss = 0.0f;
            for (int i = 0 ; i < B * T ; i++) { mean_loss += acts_memory.get(acts.losses + i); }
            mean_loss /= B * T;
        } else {
            // if we don't have targets, we don't have a loss
            mean_loss = -1.0f;
        }
        System.err.printf("forward pass took %d ms\n", System.currentTimeMillis() - t1);
    }

    public void zero_grad() {
        if (grads_memory != null) {
            for (int i = 0 ; i < num_parameters ; i++) {
                grads_memory.set(i, 0.0f);
            }
        }
        if (grads_acts_memory != null) {
            for (int i = 0 ; i < num_activations ; i++) {
                grads_acts_memory.set(i, 0.0f);
            }
        }
    }

    public void backward() throws UnexpectedException {
        // double check we forwarded previously, with targets
        if (mean_loss == -1.0f) { throw new UnexpectedException(null); }

        // convenience shortcuts (and size_t to help prevent int overflow)
        int B = batch_size;
        int T = seq_len;
        int V = config.vocab_size;
        int Vp = config.padded_vocab_size;
        int L = config.num_layers;
        int NH = config.num_heads;
        int C = config.channels;

        if (grads_memory == null) {
            grads = new ParameterTensors(config);
            grads_memory = FloatArray.fromArray(new float[params.count]);
            grads_acts = new ActivationTensors(config, B, T);
            grads_acts_memory = FloatArray.fromArray(new float[acts.count]);
        }

        // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
        // technically this is a small, inline backward() pass of calculating
        // total, final loss as the mean over all losses over all (B,T) positions in the batch
        float dloss_mean = 1.0f / (B * T);
        for (int i = 0 ; i < B * T ; i++) { grads_acts_memory.set(grads_acts.losses + i, dloss_mean); }

        long t1 = System.currentTimeMillis();
        crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, B, T, V, Vp);
        long t0 = System.currentTimeMillis();
        matmul_backward(grads_acts.lnf, grads.wte, -1, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
        System.err.printf("initial matmul_backward took %d ms\n", System.currentTimeMillis() - t0);
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
        System.err.printf("backward pass took %d ms\n", System.currentTimeMillis() - t1);
    }

    public void update(float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
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
            float m_hat = m / (1.0f - TornadoMath.pow(beta1, t));
            float v_hat = v / (1.0f - TornadoMath.pow(beta2, t));

            // update
            m_memory.put(i, m);
            v_memory.put(i, v);

            params_memory.set(i, params_memory.get(i) - learning_rate * (m_hat / (TornadoMath.sqrt(v_hat) + eps) + weight_decay * param));
        }
    }
}

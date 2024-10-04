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

public class GPT2 {
    public GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    FloatBuffer params_memory;
    public int num_parameters;
    // gradients of the weights
    public ParameterTensors grads;
    public FloatBuffer grads_memory = null;
    // buffers for the AdamW optimizer
    FloatBuffer m_memory = null;
    FloatBuffer v_memory = null;
    // the activations of the model, and their sizes
    public ActivationTensors acts;
    public FloatBuffer acts_memory = null;
    int num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    FloatBuffer grads_acts_memory = null;
    // other run state configuration
    int batch_size = 0; // the batch size (B) of current forward pass
    int seq_len = 0; // the sequence length (T) of current forward pass
    IntBuffer inputs = null; // the input tokens for the current forward pass
    IntBuffer targets = null; // the target tokens for the current forward pass
    public float mean_loss; // after a forward pass with targets, will be populated with the mean loss

    private final static float GELU_SCALING_FACTOR = (float) Math.sqrt(2.0f / Math.PI);

    private final static boolean UseVectorAPI = "true".equalsIgnoreCase(System.getProperty("UseVectorAPI", "true"));

    // https://stackoverflow.com/a/36335119 on assigning lambda to variable
    // https://stackoverflow.com/a/29220300/
    @FunctionalInterface
    public interface MatmulForward {
        void apply(int out, int inp, int weight, int bias, int B, int T, int C, int OC);
    }
   
    final MatmulForward matmul_forward = this::matmul_forward_vecapi;
    // final MatmulForward matmul_forward = this::matmul_forward_simple;
    // final MatmulForward matmul_forward = this::matmul_forward_default;
   
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
        ByteBuffer _params_memory = ByteBuffer.allocate(num_parameters * 4 /*sizeof(float)*/).order(ByteOrder.LITTLE_ENDIAN);
        this.params_memory = _params_memory.asFloatBuffer();

        model_file.getChannel().read(_params_memory);
        model_file.close();

        // other inits
        mean_loss = -1.0f; // -1.0f will designate no loss
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
                    acts_memory.put(out_bt + i, params_memory.get(wte_ix + i) + params_memory.get(wpe_t + i));
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
                    grads_memory.put(dwte_ix + i, grads_memory.get(dwte_ix + i) + d);
                    grads_memory.put(dwpe_t + i, grads_memory.get(dwpe_t + i) + d);
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
                float s = 1.0f / (float) Math.sqrt(v + eps);
                // seek to the output position in out[b,t,:]
                int out_bt = out + b * T * C + t * C;
                for (int i = 0 ; i < C ; i++) {
                    float n = (s * (acts_memory.get(x + i) - m)); // normalize
                    float o = n * params_memory.get(weight + i) + params_memory.get(bias + i); // scale and shift
                    acts_memory.put(out_bt + i, o); // write
                }
                // cache the mean and rstd for the backward pass later
                acts_memory.put(mean + b * T + t, m);
                acts_memory.put(rstd + b * T + t, s);
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
                    grads_memory.put(dbias + i, grads_memory.get(dbias + i) + grads_acts_memory.get(dout_bt + i));
                    // gradient contribution to weight
                    grads_memory.put(dweight + i, grads_memory.get(dweight + i) + norm_bti * grads_acts_memory.get(dout_bt + i));
                    // gradient contribution to input
                    float dval = 0.0f;
                    dval += dnorm_i; // term 1
                    dval -= dnorm_mean; // term 2
                    dval -= norm_bti * dnorm_norm_mean; // term 3
                    dval *= rstd_bt; // final scale
                    grads_acts_memory.put(dinp_bt + i, grads_acts_memory.get(dinp_bt + i) + dval);
                }
            }
        }
    }

    // acts, acts, params, params
    @SuppressWarnings("unused")
    private void matmul_forward_vecapi(int out, int inp, int weight, int bias, int B, int T, int C, int OC) {
        // Java vector API implementation of matrix multiplication
        // #pragma omp parallel for collapse(2)
//       for (int b = 0 ; b < B ; b++) {
//           for (int t = 0 ; t < T ; t++) {
//               int bt = b * T + t;
            IntStream.range(0, B * T).parallel().forEach( bt -> {
                // unroll bt to b, t
                // int b = bt / T;
                // int t = bt % T;
                // int bt = b * T + t;
                MemorySegment params_memory = MemorySegment.ofBuffer(this.params_memory);
                MemorySegment acts_memory = MemorySegment.ofBuffer(this.acts_memory);
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
                            var a0 = FloatVector.fromMemorySegment(species, acts_memory, (inp_base + i + 0 * lanes) * Float.BYTES, ByteOrder.BIG_ENDIAN);
                            var a1 = FloatVector.fromMemorySegment(species, acts_memory, (inp_base + i + 1 * lanes) * Float.BYTES, ByteOrder.BIG_ENDIAN);
                            var a2 = FloatVector.fromMemorySegment(species, acts_memory, (inp_base + i + 2 * lanes) * Float.BYTES, ByteOrder.BIG_ENDIAN);
                            var a3 = FloatVector.fromMemorySegment(species, acts_memory, (inp_base + i + 3 * lanes) * Float.BYTES, ByteOrder.BIG_ENDIAN);
                            var p0 = FloatVector.fromMemorySegment(species, params_memory, (weight_base + i + 0 * lanes) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                            var p1 = FloatVector.fromMemorySegment(species, params_memory, (weight_base + i + 1 * lanes) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                            var p2 = FloatVector.fromMemorySegment(species, params_memory, (weight_base + i + 2 * lanes) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                            var p3 = FloatVector.fromMemorySegment(species, params_memory, (weight_base + i + 3 * lanes) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                            sum0 = p0.fma(a0, sum0);
                            sum1 = p1.fma(a1, sum1);
                            sum2 = p2.fma(a2, sum2);
                            sum3 = p3.fma(a3, sum3);
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
                    this.acts_memory.put(out + bt * OC + o, val);
                });
            });
//            }
//        }
    }

    // acts, acts, params, params
    @SuppressWarnings("unused")
    private void matmul_forward_simple(int out, int inp, int weight, int bias, int B, int T, int C, int OC) {
        // the most naive implementation of matrix multiplication
        // this serves as an algorithmic reference, and as a fallback for
        // unfriendly input shapes inside matmul_forward(), below.
        // #pragma omp parallel for collapse(2)
//        for (int b = 0 ; b < B ; b++) {
//            for (int t = 0 ; t < T ; t++) {
//                int bt = b * T + t;
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
                    acts_memory.put(out + bt * OC + o, val);
                }
            });
//            }
//        }
    }

    // acts, acts, params, params
    @SuppressWarnings("unused")
    private void matmul_forward_default(int out, int inp, int weight, int bias, int B, int T, int C, int OC) {
        // most of the running time is spent here and in matmul_backward
        // therefore, the implementation below is very mildly optimized
        // this function is otherwise identical to that of matmul_forward_naive()
        // OC is short for "output channels"
        // inp is (B,T,C), weight is (OC, C), bias is (OC)
        // out will be (B,T,OC)

        // make sure the tiled loop will be correct or fallback to naive version
        int LOOP_UNROLL = 8;
        if (B * T % LOOP_UNROLL != 0) {
            matmul_forward_simple(out, inp, weight, bias, B, T, C, OC);
            return;
        }

        // collapse the B and T loops into one and turn it into a strided loop.
        // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
        // #pragma omp parallel for
//        for (int obt = 0; obt < B * T; obt += LOOP_UNROLL) {
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
                    acts_memory.put(out + bt * OC + o, result[ibt]);
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
//        for (int b = 0 ; b < B ; b++) {
//            for (int t = 0 ; t < T ; t++) {
            IntStream.range(0, B * T).parallel().forEach( x -> {
                int b = x / T;
                int t = x % T;
                int dout_bt = dout + b * T * OC + t * OC;
                int dinp_bt = dinp + b * T * C + t * C;
                for (int o = 0 ; o < OC ; o++) {
                    int wrow = weight + o * C;
                    float d = grads_acts_memory.get(dout_bt + o);
                    for (int i = 0; i < C; i++) {
                        grads_acts_memory.put(dinp_bt + i, grads_acts_memory.get(dinp_bt + i) + params_memory.get(wrow + i) * d);
                    }
                }
            });
//        }
        // backward into weight/bias, parallelize over output channels OC
        // #pragma omp parallel for
//        for (int o = 0 ; o < OC ; o++) {
        IntStream.range(0, OC).parallel().forEach( o -> {
            for (int b = 0 ; b < B ; b++) {
                for (int t = 0 ; t < T ; t++) {
                    int dout_bt = dout + b * T * OC + t * OC;
                    int inp_bt = inp + b * T * C + t * C;
                    int dwrow = dweight + o*C;
                    float d = grads_acts_memory.get(dout_bt + o);
                    if (dbias != -1) { grads_memory.put(dbias + o, grads_memory.get(dbias + o) + d); }
                    for (int i = 0 ; i < C ; i++) {
                        grads_memory.put(dwrow + i, grads_memory.get(dwrow + i) + acts_memory.get(inp_bt + i) * d);
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
        float scale = 1.0f / (float) Math.sqrt(hs);

        // #pragma omp parallel for collapse(3)
        for (int b = 0 ; b < B ; b++) {
            for (int t = 0 ; t < T ; t++) {
                for (int h = 0 ; h < NH ; h++) {
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

                        acts_memory.put(preatt_bth + t2, val);
                    }

                    // pass 2: calculate the exp and keep track of sum
                    // maxval is being calculated and subtracted only for numerical stability
                    float expsum = 0.0f;
                    for (int t2 = 0; t2 <= t; t2++) {
                        float expv = (float) Math.exp(acts_memory.get(preatt_bth + t2) - maxval);
                        expsum += expv;
                        acts_memory.put(att_bth + t2, expv);
                    }
                    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                    // pass 3: normalize to get the softmax
                    for (int t2 = 0; t2 < T; t2++) {
                        if (t2 <= t) {
                            acts_memory.put(att_bth + t2, acts_memory.get(att_bth + t2) * expsum_inv);
                        } else {
                            // causal attention mask. not strictly necessary to set to zero here
                            // only doing this explicitly for debugging and checking to PyTorch
                            acts_memory.put(att_bth + t2, 0.0f);
                        }
                    }

                    // pass 4: accumulate weighted values into the output of attention
                    int out_bth = out + b * T * C + t * C + h * hs;
                    for (int i = 0; i < hs; i++) { acts_memory.put(out_bth + i, 0.0f); }
                    for (int t2 = 0; t2 <= t; t2++) {
                        int value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                        float att_btht2 = acts_memory.get(att_bth + t2);
                        for (int i = 0; i < hs; i++) {
                            acts_memory.put(out_bth + i, acts_memory.get(out_bth + i) + att_btht2 * acts_memory.get(value_t2 + i));
                        }
                    }
                }
            }
        }
    }

    // grads_act, grads_act, grads_act, grads_act, acts, acts
    public void attention_backward(int dinp, int dpreatt, int datt, int dout, int inp, int att, int B, int T, int C, int NH) {
        // inp/dinp are (B, T, 3C) Q,K,V
        // att/datt/dpreatt are (B, NH, T, T)
        // dout is (B, T, C)
        int C3 = C * 3;
        int hs = C / NH; // head size
        float scale = 1.0f / (float) Math.sqrt(hs);

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
                            grads_acts_memory.put(datt_bth + t2, grads_acts_memory.get(datt_bth + t2) + acts_memory.get(value_t2 + i) * grads_acts_memory.get(dout_bth + i));
                            grads_acts_memory.put(dvalue_t2 + i, grads_acts_memory.get(dvalue_t2 + i) + acts_memory.get(att_bth + t2) * grads_acts_memory.get(dout_bth + i));
                        }
                    }

                    // backward pass 2 & 3, the softmax
                    // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                    for (int t2 = 0 ; t2 <= t ; t2++) {
                        for (int t3 = 0 ; t3 <= t ; t3++) {
                            float indicator = t2 == t3 ? 1.0f : 0.0f;
                            float local_derivative = acts_memory.get(att_bth + t2) * (indicator - acts_memory.get(att_bth + t3));
                            grads_acts_memory.put(dpreatt_bth + t3, grads_acts_memory.get(dpreatt_bth + t3) + local_derivative * grads_acts_memory.get(datt_bth + t2));
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
                            grads_acts_memory.put(dquery_t + i, grads_acts_memory.get(dquery_t + i) + acts_memory.get(key_t2 + i) * grads_acts_memory.get(dpreatt_bth + t2) * scale);
                            grads_acts_memory.put(dkey_t2 + i, grads_acts_memory.get(dkey_t2 + i) + acts_memory.get(query_t + i) * grads_acts_memory.get(dpreatt_bth + t2) * scale);
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
            acts_memory.put(out + i, 0.5f * x * (1.0f + (float) Math.tanh(GELU_SCALING_FACTOR * (x + cube))));
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
            float tanh_out = (float) Math.tanh(tanh_arg);
            float coshf_out = (float) Math.cosh(tanh_arg);
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            grads_acts_memory.put(dinp + i, grads_acts_memory.get(dinp + i) + local_grad * grads_acts_memory.get(dout + i));
        }
    }
    // #pragma float_control(pop)

    // acts, acts, acts
    public void residual_forward(int out, int inp1, int inp2, int N) {
        for (int i = 0 ; i < N ; i++) {
            acts_memory.put(out + i, acts_memory.get(inp1 + i) + acts_memory.get(inp2 + i));
        }
    }

    // grads_acts, grads_acts, grads_acts
    public void residual_backward(int dinp1, int dinp2, int dout, int N) {
        for (int i = 0 ; i < N ; i++) {
            grads_acts_memory.put(dinp1 + i, grads_acts_memory.get(dinp1 + i) + grads_acts_memory.get(dout + i));
            grads_acts_memory.put(dinp2 + i, grads_acts_memory.get(dinp2 + i) + grads_acts_memory.get(dout + i));
        }
    }

    // acts, acts
    public void softmax_forward(int probs, int logits, int B, int T, int V, int Vp) {
        // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
        // input: logits is (B,T,Vp) of the unnormalized log probabilities
        // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
        // example: Vp is 50304 and V is 50257
        // #pragma omp parallel for collapse(2)
        for (int b = 0 ; b < B ; b++) {
            for (int t = 0 ; t < T ; t++) {
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
                    acts_memory.put(probs_bt + i, (float) Math.exp(acts_memory.get(logits_bt + i) - maxval));
                    sum += acts_memory.get(probs_bt + i);
                }
                // note we only loop to V, leaving the padded dimensions
                for (int i = 0 ; i < V ; i++) {
                    acts_memory.put(probs_bt + i, acts_memory.get(probs_bt + i) / sum);
                }
                // for extra super safety we may wish to include this too,
                // forcing the probabilities here to be zero, but it shouldn't matter
                for (int i = V; i < Vp; i++) {
                    acts_memory.put(probs_bt + i, 0.0f);
                }
            }
        }
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
                acts_memory.put(losses + b * T + t, (float) -Math.log(acts_memory.get(probs_bt + ix)));
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
                    grads_acts_memory.put(dlogits_bt + i, grads_acts_memory.get(dlogits_bt + i) + (p - indicator) * dloss);
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
        if(acts_memory == null) {
            // record the current B,T as well
            batch_size = B;
            seq_len = T;
            // and now allocate the space
            acts = new ActivationTensors(config, B, T);

            num_activations = acts.count;
            System.out.println("num_activations: " + num_activations);
            ByteBuffer _acts_memory = ByteBuffer.allocate(num_activations * 4);
            acts_memory = _acts_memory.asFloatBuffer();
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
        inputs.rewind();
        this.inputs.rewind();
        this.inputs.put(inputs);
        if (targets != null ) {
            targets.rewind();
            this.targets.rewind();
            this.targets.put(targets);
        }

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
            matmul_forward.apply(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
            attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
            matmul_forward.apply(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
            residual_forward(l_residual2, residual, l_attproj, B * T * C);
            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
            matmul_forward.apply(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
            gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
            matmul_forward.apply(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
            residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
        }
        residual = acts.residual3 + (L - 1) * B * T * C; // last residual is in residual3
        layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
        matmul_forward.apply(acts.logits, acts.lnf, params.wte, -1, B, T, C, Vp);
        softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

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
            grads_memory = FloatBuffer.allocate(params.count);
            grads_acts = new ActivationTensors(config, B, T);
            grads_acts_memory = FloatBuffer.allocate(acts.count);
        }

        // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
        // technically this is a small, inline backward() pass of calculating
        // total, final loss as the mean over all losses over all (B,T) positions in the batch
        float dloss_mean = 1.0f / (B * T);
        for (int i = 0 ; i < B * T ; i++) { grads_acts_memory.put(grads_acts.losses + i, dloss_mean); }

        crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, B, T, V, Vp);
        matmul_backward(grads_acts.lnf, grads.wte, -1, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
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
            float m_hat = m / (1.0f - (float) Math.pow(beta1, t));
            float v_hat = v / (1.0f - (float) Math.pow(beta2, t));

            // update
            m_memory.put(i, m);
            v_memory.put(i, v);

            params_memory.put(i, params_memory.get(i) - learning_rate * (m_hat / ((float) Math.sqrt(v_hat) + eps) + weight_decay * param));
        }
    }
}

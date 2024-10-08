package com.otabuzzman.llmj;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.rmi.UnexpectedException;

public class TestGpt2 {

    private FloatBuffer grads_memory;
    private FloatBuffer expected_grads_memory;

    // poor man's tensor checker
    private boolean check_tensor(int a, int b, int n, String label) {
        int print_upto = 5;
        boolean ok = true;
        float maxdiff = 0.0f;
        float tol = 2e-2f;
        System.out.println(label);
        for (int i = 0; i < n; i++) {
            // look at the diffence at position i of these two tensors
            float diff = Math.abs(grads_memory.get(a + i) - expected_grads_memory.get(b + i));

            // keep track of the overall error
            ok = ok && (diff <= tol);
            if (diff > maxdiff) { maxdiff = diff; }

            // for the first few elements of each tensor, pretty print
            // the actual numbers, so we can do a visual, qualitative proof/assessment
            if (i < print_upto) {
                if (diff <= tol) {
                    System.out.print("OK ");
                } else {
                    System.out.print("NOT OK ");
                }
                System.out.printf("%f %f\n", grads_memory.get(a + i), expected_grads_memory.get(b + i));
            }
        }
        // print the final result for this tensor
        if (ok) {
            System.out.printf("TENSOR OK, maxdiff = %e\n", maxdiff);
        } else {
            System.out.printf("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
        }
        return ok;
    }

    public void run() throws FileNotFoundException, IOException, UnexpectedException {
        // build the GPT-2 model from a checkpoint
        GPT2 model = new GPT2("gpt2_124M.bin");

        int C = model.config.channels;
        int V = model.config.vocab_size;
        int Vp = model.config.padded_vocab_size;
        int maxT = model.config.max_seq_len;
        int L = model.config.num_layers;

        RandomAccessFile state_file = new RandomAccessFile("gpt2_124M_debug_state.bin", "r");
        int[] state_header = new int[256];
        for (int i = 0; i < 256; i++) {
            state_header[i] = Integer.reverseBytes(state_file.readInt()); // convert little-endians in file to JVM big-endians
        }
        assert(state_header[0] == 20240327) : "Bad magic in model file (try `python train_gpt2.py`)";
        assert(state_header[1] == 2) : "Wrong version in model file (try `python train_gpt2.py`)";

        int B = state_header[2]; // batch size, e.g. 4
        int T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
        System.out.println("[State]");
        System.out.println("batch_size: " + B);
        System.out.println("seq_len: " + T);

        ParameterTensors expected_grads = new ParameterTensors(model.config);
        ByteBuffer _expected_grads_memory = ByteBuffer.allocate(model.num_parameters * 4 /*sizeof(float)*/).order(ByteOrder.LITTLE_ENDIAN);
        this.expected_grads_memory = _expected_grads_memory.asFloatBuffer();

        // inputs and expected outputs, only used for error checking
        ByteBuffer _x = ByteBuffer.allocate(B * T * 4 /*sizeof(int)*/).order(ByteOrder.LITTLE_ENDIAN);
        IntBuffer x = _x.asIntBuffer();

        ByteBuffer _y = ByteBuffer.allocate(B * T * 4 /*sizeof(int)*/).order(ByteOrder.LITTLE_ENDIAN);
        IntBuffer y = _y.asIntBuffer();

        ByteBuffer _expected_logits = ByteBuffer.allocate(B * T * V * 4 /*sizeof(float)*/).order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer expected_logits = _expected_logits.asFloatBuffer();

        ByteBuffer _expected_loss = ByteBuffer.allocate(1 * 4 /*sizeof(float)*/).order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer expected_loss = _expected_loss.asFloatBuffer();

        state_file.getChannel().read(_x);
        state_file.getChannel().read(_y);
        state_file.getChannel().read(_expected_logits);
        state_file.getChannel().read(_expected_loss);
        state_file.getChannel().read(_expected_grads_memory);
        state_file.close();

        // overall OK signal for the test
        boolean allok = true;

        // let's do 10 training iterations, following the pytorch code
        float expected_losses[] = {
            5.270007133483887f,
            4.059706687927246f,
            3.3751230239868164f,
            2.8007826805114746f,
            2.315382242202759f,
            1.8490285873413086f,
            1.3946564197540283f,
            0.9991465210914612f,
            0.6240804195404053f,
            0.37651097774505615f
        };

        for (int step = 0 ; step < 10 ; step++) {
            long start = System.currentTimeMillis();

            model.forward(x, y, B, T);
            model.zero_grad();
            model.backward();

            long end = System.currentTimeMillis();

            if (step == 0) {
                // error checking at step 0 for reference activations/gradients
                // at this point, target should be equal to expected_logits, let's compare
                boolean logits_ok = true;
                int calculated_logits = model.acts.logits;
                float max_diff = 0.0f;
                for (int bt = 0 ; bt < B * T ; bt++) {
                    for (int v = 0 ; v < V ; v++) { // note we only loop to V (ignoring padding)
                        int i = bt * Vp + v; // linearized index, using Vp
                        if (i < 10) {
                            System.out.printf("%f, %f\n", expected_logits.get(i), model.acts_memory.get(calculated_logits + i));
                        }
                        float diff = Math.abs(expected_logits.get(bt * V + v) - model.acts_memory.get(calculated_logits + i));
                        max_diff = Math.max(max_diff, diff);
                        if (diff >= 1e-2f) {
                            System.out.printf("MISMATCH AT INDEX %d,%d: ", bt, v);
                            System.out.printf("%f %f\n", expected_logits.get(bt * V + v), model.acts_memory.get(calculated_logits + i));
                            logits_ok = false;
                            bt = B*T; // to break out of both loops
                            break;
                        }
                    }
                }
                if (!logits_ok) { System.out.printf("NOT "); }
                System.out.printf("OK (LOGITS), max_diff = %e\n", max_diff);
                allok = allok && logits_ok;

                // compare the achieved loss
                if (Math.abs(model.mean_loss - expected_loss.get(0)) >= 1e-2f) {
                    System.out.printf("LOSS MISMATCH: %f %f\n", model.mean_loss, expected_loss.get(0));
                    allok = false;
                } else {
                    System.out.printf("LOSS OK: %f %f\n", model.mean_loss, expected_loss.get(0));
                }

                // finally check all the gradients
                boolean[] gradoks = new boolean[16];
                ParameterTensors grads = model.grads;
                grads_memory = model.grads_memory;
                gradoks[0] = check_tensor(grads.wte, expected_grads.wte, V * C, "dwte");
                gradoks[1] = check_tensor(grads.wpe, expected_grads.wpe, maxT * C, "dwpe");
                gradoks[2] = check_tensor(grads.ln1w, expected_grads.ln1w, L * C, "dln1w");
                gradoks[3] = check_tensor(grads.ln1b, expected_grads.ln1b, L * C, "dln1b");
                gradoks[4] = check_tensor(grads.qkvw, expected_grads.qkvw, L * 3 * C * C, "dqkvw");
                gradoks[5] = check_tensor(grads.qkvb, expected_grads.qkvb, L * 3 * C, "dqkvb");
                gradoks[6] = check_tensor(grads.attprojw, expected_grads.attprojw, L * C * C, "dattprojw");
                gradoks[7] = check_tensor(grads.attprojb, expected_grads.attprojb, L * C, "dattprojb");
                gradoks[8] = check_tensor(grads.ln2w, expected_grads.ln2w, L * C, "dln2w");
                gradoks[9] = check_tensor(grads.ln2b, expected_grads.ln2b, L * C, "dln2b");
                gradoks[10] = check_tensor(grads.fcw, expected_grads.fcw, L * 4 * C * C, "dfcw");
                gradoks[11] = check_tensor(grads.fcb, expected_grads.fcb, L * 4 * C, "dfcb");
                gradoks[12] = check_tensor(grads.fcprojw, expected_grads.fcprojw, L * C * 4 * C, "dfcprojw");
                gradoks[13] = check_tensor(grads.fcprojb, expected_grads.fcprojb, L * C, "dfcprojb");
                gradoks[14] = check_tensor(grads.lnfw, expected_grads.lnfw, C, "dlnfw");
                gradoks[15] = check_tensor(grads.lnfb, expected_grads.lnfb, C, "dlnfb");
                for (int i = 0 ; i < 16 ; i++) {
                    allok = allok && gradoks[i];
                }
            }

            model.update(1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step + 1);

            // compare the losses
            float eXpected_loss = expected_losses[step];
            float actual_loss = model.mean_loss;
            boolean step_loss_ok = Math.abs(eXpected_loss - actual_loss) < 1e-2f;
            allok = allok && step_loss_ok;

            // print the timing information at the end
            System.out.printf("step %d: loss %f (took %d ms) OK = %b\n", step, model.mean_loss, end - start, step_loss_ok);
        }

        // final judgement
        System.out.printf("overall okay: %b\n", allok);
    }

    public static void main(String[] args) {
        TestGpt2 gpt = new TestGpt2();
        try {
            gpt.run();
        } catch (Exception e) {
            System.err.println(e);
        }
    }
}

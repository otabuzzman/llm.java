package com.otabuzzman.llmj;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.rmi.UnexpectedException;

public class TrainGpt2 {

    private long state = 1337;
    private long random_u32() {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        state ^= state >>> 12;
        state ^= state << 25;
        state ^= state >>> 27;
        long u32 = (state * 0x2545F4914F6CDD1Dl) >>> 32;
        return (u32 < 0) ? -u32 : u32;
    }

    private float random_f32() { // random float32 in [0,1)
        return (random_u32() >>> 8) / 16777216.0f;
    }

    public void run() throws FileNotFoundException, IOException, UnexpectedException {
        // build the GPT-2 model from a checkpoint
        GPT2 model = new GPT2("gpt2_124M.bin");

        // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
        String tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
        String tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
        String tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
        String tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
        String train_tokens = Files.exists(Paths.get(tiny_shakespeare_train)) ? tiny_shakespeare_train : tiny_stories_train;
        String val_tokens = Files.exists(Paths.get(tiny_shakespeare_val)) ? tiny_shakespeare_val : tiny_stories_val;
        int B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
        int T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
        DataLoader train_loader = new DataLoader(train_tokens, B, T, 0, 1, true);
        DataLoader val_loader = new DataLoader(val_tokens, B, T, 0, 1, false);
        System.out.println("train dataset num_batches: " + train_loader.num_tokens / (B * T));
        System.out.println("val dataset num_batches: " + val_loader.num_tokens / (B * T));
        int val_num_batches = 5;

        // build the Tokenizer
        Tokenizer tokenizer = new Tokenizer("gpt2_tokenizer.bin");

        // some memory for generating samples from the model
        IntBuffer gen_tokens = IntBuffer.allocate(B * T);
        int genT = 64; // number of steps of inference we will do

        // train
        long start, end;
        for (int step = 0 ; step <= 40 ; step++) {

            // once in a while estimate the validation loss
            if (step % 10 == 0) {
                float val_loss = 0.0f;
                val_loader.reset();
                for (int i = 0; i < val_num_batches; i++) {
                    val_loader.next_batch();
                    model.forward(val_loader.inputs, val_loader.targets, B, T);
                    val_loss += model.mean_loss;
                }
                val_loss /= val_num_batches;
                System.out.println("val loss " + val_loss);
            }

            // once in a while do model inference to print generated text
            if (step > 0 && step % 20 == 0) {
                // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
                for(int i = 0 ; i < B * T ; ++i) {
                    gen_tokens.put(i, tokenizer.eot_token);
                }
                // now sample from the model autoregressively
                System.out.print("generating:\n---\n");
                for (int t = 1 ; t < genT ; t++) {
                    // note that inference is very wasteful here because for each token
                    // we re-calculate the forward pass for all of (B,T) positions from scratch
                    // but the inference here is just for sanity checking anyway
                    // and we can maybe optimize a bit more later, with careful tests
                    model.forward(gen_tokens, null, B, T);
                    // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                    // we're in principle running B "inference streams" in parallel here
                    // but only using position 0
                    // get the Vp-dimensional vector probs[0, t-1, :]
                    int probs = model.acts.probs + (t - 1) * model.config.padded_vocab_size;
                    float coin = random_f32();
                    // note we're only sampling from the first V elements, ignoring padding
                    // (the probabilities in the padded region should be zero anyway)
                    int next_token = model.config.vocab_size - 1; // in case of rounding errors
                    // sample index from probabilities (they must sum to 1!)
                    // coin is a random number in [0, 1), usually from random_f32()
                    float cdf = 0.0f;
                    for (int i = 0; i < model.config.vocab_size; i++) {
                        cdf += model.acts_memory.get(probs + i);
                        if (coin < cdf) {
                            next_token = i;
                            break;
                        }
                    }

                    gen_tokens.put(t, next_token);
                    // print the generated token, either using the Tokenizer or a fallback
                    if (tokenizer.init_ok) {
                        String token_str = tokenizer.decode(next_token);
                        if (tokenizer.isprint(token_str)) {
                            System.out.print(token_str);
                        }
                    } else {
                        // fall back to printing the token id
                        System.out.print(String.valueOf(next_token) + " ");
                    }
                    System.out.flush();
                }
                System.out.print("\n---\n");
            }

            // do a training step
            start = System.currentTimeMillis();
            train_loader.next_batch();
            model.forward(train_loader.inputs, train_loader.targets, B, T);
            model.zero_grad();
            model.backward();
            model.update(1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step + 1);
            end = System.currentTimeMillis();
            System.out.printf("step %d: train loss %f (took %d ms)\n", step, model.mean_loss, end - start);
        }
    }

    public static void main(String[] args) {
        TrainGpt2 gpt = new TrainGpt2();
        try {
            gpt.run();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

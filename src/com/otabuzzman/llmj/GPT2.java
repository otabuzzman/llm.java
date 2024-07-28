package com.otabuzzman.llmj;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

public class GPT2 {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    int[] param_sizes = new int[ParameterTensors.NUM_PARAMETER_TENSORS];
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
    int[] act_sizes = new int[ActivationTensors.NUM_ACTIVATION_TENSORS];
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
        num_parameters = params.count();
        System.out.println("num_parameters: " + num_parameters);

        // read in all the parameters from file
        //params_memory = FloatBuffer.allocate(num_parameters);
        ByteBuffer params_memory = ByteBuffer.allocate(num_parameters * 4 /*sizeof(float)*/);
        model_file.getChannel().read(params_memory);
        params_memory.order(ByteOrder.LITTLE_ENDIAN);
        params_memory.flip();
        this.params_memory = params_memory.asFloatBuffer();
        //for (int i = 0; i < num_parameters; i++) {
        //    params_memory.put(i, Float.intBitsToFloat(Integer.reverseBytes(model_file. readInt())));
        //}
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

        for ( int i=0 ; i < 16 ; i++) {
            System.out.println(this.params_memory.get(i));
        }

    }
}

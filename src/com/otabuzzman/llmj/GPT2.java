package com.otabuzzman.llmj;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

public class GPT2 {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    int param_sizes[ParameterTensors.NUM_PARAMETER_TENSORS];
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
    int act_sizes[ActivationTensors.NUM_ACTIVATION_TENSORS];
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
}

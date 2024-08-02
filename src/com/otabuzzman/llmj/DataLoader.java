/*
 Implements:
 - DataLoader for model training. Reads and serves data shards.
 - EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag.
 */

package com.otabuzzman.llmj;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.rmi.UnexpectedException;

public class DataLoader {

    private final static int HEADER_SIZE = 256;

        // variables related to distributed training
    // each process/worker has to access different parts of the data
    int process_rank = 0;
    int num_processes = 0;
    // batch and token information
    int B = 0; // batch size
    int T = 0; // sequence length
    public long num_tokens = 0; // total number of tokens
    int shard_num_samples = 0;  // total number of samples in the current shard per process
    // shards and current position
    Glob glob_result; // stores the result of glob, for all shards we want to iterate
    int current_shard_idx = 0; // the current shard we are reading from
    int current_sample_idx = 0; // the current sample we are reading from
    // file handle
    RandomAccessFile tokens_file = null;
    // data buffers
    // we fread data from file into this buffer
    IntBuffer buffer; // actually unsigned shorts
    // input tokens into transformer
    public IntBuffer inputs;
    // target tokens for the transformer
    public IntBuffer targets;
    // random shuffle related variables
    Mt19937 shuffle_rng;
    boolean should_shuffle = false;
    int[] shard_indices;
    int[] intra_shard_indices = null;
    // sizes in bytes
    int total_batch_size_bytes = 0;
    int local_batch_offset_bytes = 0;
    int header_bytes = 0;
    long file_size_bytes = 0;

    public DataLoader(String filename_pattern, int B, int T, int process_rank, int num_processes, boolean should_shuffle) throws IOException, UnexpectedException {
        this.process_rank = process_rank;
        this.num_processes = num_processes;
        this.B = B;
        this.T = T;
        tokens_file = null;
        this.should_shuffle = should_shuffle;
        header_bytes = HEADER_SIZE * 4; // sizeof(int)
        total_batch_size_bytes = ((num_processes * (B * T)) * 2); // sizeof(short)
        local_batch_offset_bytes = process_rank * B * T * 2; // sizeof(short)

        // glob to get the list of files matching the pattern, these are our data shards
        glob_result = new Glob(filename_pattern, ".");
        if (glob_result.gl_pathc() == 0) {
            throw new UnexpectedException(null);
        }

        if (should_shuffle) {
            shuffle_rng = new Mt19937();
            shuffle_rng.manual_seed(42 + process_rank);
            shard_indices = new int[glob_result.gl_pathc()];
            shuffle_rng.init_identity_permutation(shard_indices, glob_result.gl_pathc());
            intra_shard_indices = null; // dynamically allocated allowing different shard sizes
        }

        // inspect and validate all shards so we don't get any runtime errors later
        // if too slow / too many shards, may wish to revisit later
        long ntok_total = 0;
        for (int shard_index = 0 ; shard_index < glob_result.gl_pathc() ; shard_index++) {
            long shard_ntok = load_shard(shard_index);
            // we need at least one batch/shard, the way things are written right now.
            // can be relaxed a lot later.
            if (shard_ntok < num_processes * B * T + 1) { throw new UnexpectedException(null); } // at least one batch per shard needed
            ntok_total += shard_ntok;
        }
        // debugging prints
        // printf("DataLoader: filename_pattern: %s\n", filename_pattern);
        // printf("DataLoader: Found %ld tokens across %zu shards\n", ntok_total, loader->glob_result.gl_pathc);

        // allocate all the space we'll need
        buffer = IntBuffer.allocate(B * T + 1);
        inputs = IntBuffer.allocate(B * T);
        targets = IntBuffer.allocate(B * T);
        num_tokens = ntok_total;

        // reset the loader, to initialize it
        reset();
    }

    private long load_shard(int shard_index) throws FileNotFoundException, IOException, UnexpectedException {
        int file_index = shard_index;
        if (should_shuffle) {
            file_index = shard_indices[shard_index];
        }
        // use the first glob match as the filename for now
        String filename = glob_result.gl_pathv(file_index);
        // open the input file for reading. also only a single file can be opened at a time
        if (tokens_file != null) {
            tokens_file.close();
        }
        tokens_file = new RandomAccessFile(filename, "r");
        // validate the header
        int[] header = new int[DataLoader.HEADER_SIZE];
        for (int i = 0; i < 256; i++) {
            header[i] = Integer.reverseBytes(tokens_file.readInt()); // convert little-endians in file to JVM big-endians
        }
        assert(header[0] == 20240520) : "Bad magic in data file (retry preprocessing or refer to README)";
        assert(header[1] == 1 || header[1] == 2) : "Wrong version in data file (retry preprocessing or refer to README)";
        int ntok = header[2]; // number of tokens in the file
        if (ntok == 0) { throw new UnexpectedException(null); } // we expect some tokens in the file. this should never trip, right?
        // determine the file size and make sure it is consistent with the number of tokens
        file_size_bytes = tokens_file.length();
        // we expect ntok in the file to be consistent with filesize, assert that is the case
        long expected_file_size = HEADER_SIZE * 4 /*sizeof(int)*/ + ntok * 2 /*sizeof(short)*/;
        if (file_size_bytes != expected_file_size) { throw new UnexpectedException(null); }
        // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
        shard_num_samples = (ntok * 2 - 2 /*sizeof(short) - sizeof(short)*/) / total_batch_size_bytes;
        return ntok;
    }

    private void prepare_intra_shard_indices() {
        // shuffle the examples inside the shards
        if (intra_shard_indices != null) {
            // in case shards have different number of samples / sizes
            intra_shard_indices = null; // GC
        }
        intra_shard_indices = new int[shard_num_samples];
        shuffle_rng.init_identity_permutation(intra_shard_indices, shard_num_samples);
        shuffle_rng.random_permutation(intra_shard_indices, shard_num_samples);
    }

    public void reset() throws FileNotFoundException, IOException, UnexpectedException {
        current_shard_idx = 0;
        current_sample_idx = 0;

        if (should_shuffle) { // shuffle the shards
            shuffle_rng.random_permutation(shard_indices, glob_result.gl_pathc());
        }

        load_shard(current_shard_idx);

        if (should_shuffle) {
            prepare_intra_shard_indices();
        }
    }

    private void advance() throws FileNotFoundException, IOException, UnexpectedException {
        if (current_shard_idx == glob_result.gl_pathc() - 1) {
            // if we are at the last shard, we reset the loader and start a new epoch
            reset();
            return;
        }

        // advance the loader by loading the next data shard and resetting the position
        current_shard_idx = (current_shard_idx + 1) % glob_result.gl_pathc();
        current_sample_idx = 0;
        load_shard(current_shard_idx);

        if (should_shuffle) {
            prepare_intra_shard_indices();
        }
    }

    public void load_batch() throws IOException, UnexpectedException {
        if (should_shuffle && intra_shard_indices == null) { throw new UnexpectedException(null); } // no shards to shuffle
        if (current_sample_idx >= shard_num_samples) { throw new UnexpectedException(null); } // sample index out of bounds
        int idx = should_shuffle ? intra_shard_indices[current_sample_idx] : current_sample_idx;
        int global_batch_offset_bytes = idx * total_batch_size_bytes;
        long current_offset = header_bytes + global_batch_offset_bytes + local_batch_offset_bytes;

        // read B*T+1 uint16_t tokens from the file into buffer
        tokens_file.seek(current_offset);
        ByteBuffer token_bytes = ByteBuffer.allocate((B * T + 1) * 2 /*sizeof(short)*/);
        tokens_file.getChannel().read(token_bytes);
        token_bytes.order(ByteOrder.LITTLE_ENDIAN);
        token_bytes.flip();
        ShortBuffer token_shorts = token_bytes.asShortBuffer();
        for (int i = 0 ; i < B * T + 1 ; i++) {
            int t = token_shorts.get(i) & 0xffff;
            buffer.put(i, t);
        }
        // decode the buffer into inputs and targets (cast to int)
        for (int i = 0; i < B * T; i++) {
            inputs.put(i, buffer.get(i));
            targets.put(i, buffer.get(i + 1));
        }
    }

    public void next_batch() throws IOException, UnexpectedException {
        // if the next batch would go past the end of the file, advance the loader
        if (current_sample_idx >= shard_num_samples) {
            advance();
        }
        load_batch();
        current_sample_idx += 1;
    }

    public void resume(int current_shard_idx, int current_sample_idx) throws FileNotFoundException, IOException, UnexpectedException {
        // used during model resumption (-y 1) flag
        this.current_shard_idx = current_shard_idx;
        this.current_sample_idx = current_sample_idx;
        load_shard(current_shard_idx);
    }       
}

// ----------------------------------------------------------------------------
// Distributed Eval Loader
// Java port saved for later...

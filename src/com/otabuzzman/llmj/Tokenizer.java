/*
 Defines the GPT-2 Tokenizer.
 Only supports decoding, i.e.: tokens (integers) -> strings
 This is all we need for unconditional generation.
 If we wanted to later prompt the model, we'd have to add decoding.
 Which could be tricky in C because of the regex involved, to look into later.
 */

package com.otabuzzman.llmj;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.rmi.UnexpectedException;

public class Tokenizer {

    int vocab_size = 0;
    String[] token_table;
    public boolean init_ok = false;
    public int eot_token = 0; // <|endoftext|> token id

    public Tokenizer(String filename) throws FileNotFoundException, IOException, UnexpectedException  {
        RandomAccessFile file = new RandomAccessFile(filename, "r");
        int[] header = new int[256];
        for (int i = 0; i < 256; i++) {
            header[i] = Integer.reverseBytes(file.readInt()); // convert little-endians in file to JVM big-endians
        }
        assert(header[0] == 20240328) : "Bad magic in tokenizer file";
        assert(header[1] == 1 || header[1] == 2) : "Wrong version in tokenizer file";
        int version = header[1];
        vocab_size = header[2];
        if (version == 1) {
            // version 1 didn't include the EOT token id
            // so we assume it is 50256, the EOT in GPT-2
            if (vocab_size != 50257) { throw new UnexpectedException(null); } // let's be defensive here
            eot_token = 50256;
        } else { // version == 2
            eot_token = header[3];
        }
        int length;
        token_table = new String[vocab_size];
        for (int i = 0 ; i<vocab_size ; i++) {
            length = file.readByte() & 0xFF; // convert signed byte to unsigned int
            if (length == 0) { throw new UnexpectedException(null); }
            byte[] token_bytes = new byte[length];
            file.read(token_bytes, 0, length);
            token_table[i] = new String(token_bytes, StandardCharsets.UTF_8);
        }
        file.close();
        init_ok = true;
    }

    public String decode(int token_id) {
        if (!init_ok) { return null; }
        if (token_id < vocab_size) {
            return token_table[token_id];
        } else {
            return null;
        }
    }

    public boolean isprint(String piece) {
        // the tokens are raw bytes, and we we only want to print the printable ones
        // many bytes can be various control codes, backspace, etc.
        if (piece == null) { return false; }
        if (piece.length() == 0) { return false; }
        // handle individual byte tokens
        // every token is asserted to be at least one byte so doing piece[1] is ok
        if (piece.length() == 1) {
            char byte_val = piece.charAt(0);
            if (Character.isISOControl(byte_val)) {
                return false; // weird byte, don't print it
            }
        }
        return true;
    }
}

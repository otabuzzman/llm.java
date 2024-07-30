/*
 Mersenne Twisters implementation, numerically identical to torch.

 Example usage:

    Mt19937 state = new Mt19937();
    state.manual_seed(137);
    System.out.println(state.randint32());
    System.out.println(state.randint32());
    System.out.println(state.randint32());
    System.out.println(state.randint32());
    System.out.println(state.randint32());

    float t8[] = new float[8];
    state.normal(t8, 8, 0, 1);
    for (int i = 0; i < 8; i++) {
        System.out.println(t8[i]);
    }
    System.out.println(state.randint32());

    float t16[] = new float[16];
    state.normal(t16, 16, 0, 1);
    for (int i = 0; i < 16; i++) {
        System.out.println(t16[i]);
    }
    System.out.println(state.randint32());

 PyTorch reference (producing identical results):

    import torch
    torch.manual_seed(137)
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    t = torch.zeros(8);
    t.normal_()
    for i in range(len(t)) :
        print(t[i].item())
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    t = torch.zeros(16);
    t.normal_()
    for i in range(len(t)) :
        print(t[i].item())
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())

 Both output:

    4053805790
    2173880614
    380293709
    1237255315
    2986595568
    0.7947664260864258
    1.4369317293167114
    - 0.2292192131280899
    0.47556325793266296
    - 0.6334410905838013
    - 0.5791953802108765
    - 0.0925704762339592
    - 0.8659197092056274
    2186503452
    - 1.2813878059387207
    - 2.646395683288574
    - 0.06569503247737885
    0.2180829495191574
    - 0.46536165475845337
    - 0.33108410239219666
    2.5485482215881348
    0.10425379872322083
    0.8460659980773926
    0.9462448358535767
    - 0.2913765013217926
    0.34313806891441345
    - 1.1186704635620117
    - 0.18305328488349915
    - 2.3153159618377686
    0.3961987793445587
    2756748748
 */

 package com.otabuzzman.llmj;

public class Mt19937 {

    final static int MERSENNE_STATE_M = 397;
    final static int MERSENNE_STATE_N = 624;

    final static int LMASK = 0x7fffffff;
    final static int UMASK = 0x80000000;

    // private long seed;
    int left; 
    int next; 
    int[] state = new int[MERSENNE_STATE_N];
    int[] MATRIX_A = new int[2];

    public void manual_seed(int seed) {
         MATRIX_A[0] = 0x0;
         MATRIX_A[1] = 0x9908b0df;
         state[0] = seed & 0xffffffff;
        for (int j = 1 ; j < MERSENNE_STATE_N ; j++) {
            state[j] = 1812433253 * (state[j - 1] ^ (state[j - 1] >>> 30)) + j;
            state[j] &= 0xffffffff;
        }
        left = 1;
        next = 0;
    }

    public void next_state() {
        left = MERSENNE_STATE_N;
        next = 0;
        int y, j;
        for (j = 0 ; j < MERSENNE_STATE_N -  MERSENNE_STATE_M; j++) {
            y = (state[j] & UMASK) | (state[j + 1] & LMASK);
            state[j] = state[j + MERSENNE_STATE_M] ^ (y >>> 1) ^ MATRIX_A[y & 0x1];
        }
        for ( ; j < MERSENNE_STATE_N - 1 ; j++) {
            y = (state[j] & UMASK) | (state[j + 1] & LMASK);
            state[j] = state[j + (MERSENNE_STATE_M - MERSENNE_STATE_N)] ^ (y >>> 1) ^ MATRIX_A[y & 0x1];
        }
        y = (state[MERSENNE_STATE_N - 1] & UMASK) | (state[0] & LMASK);
        state[MERSENNE_STATE_N - 1] = state[MERSENNE_STATE_M - 1] ^ (y >>> 1) ^ MATRIX_A[y & 0x1];
    }

    public long randint32() { // return long due to Java's lack of unsigned int
        if (MATRIX_A[0] != 0 || MATRIX_A[1] != 0x9908b0df) manual_seed(5489); // auto-initialize
        if (--left <= 0) {
            next_state();
        }
        int y = state[next++];
        y ^= y >>> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >>> 18;
       return (long) ((y <= 0) ? y + 0x100000000L : y);
    }

    public long randint64() {
        return (((randint32()) << 32) | randint32());
    }

    public float randfloat32() {
        return (randint32() & ((1l << 24) - 1)) * (1.0f / (1l << 24));
    }

    public double randfloat64() {
        return (randint64() & ((1l << 53) - 1)) * (1.0 / (1l << 53));
    }

    private void uniform(float[] data, int numel, float from, float to) {
        for (int t = 0 ; t < numel ; t++) {
            data[t] = randfloat32() * (to - from) + from;
        }
    }

    // Box-Muller transform: maps uniform random numbers to Gaussian distributed numbers
    // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    public void normal_fill_16(float[] data, float mean, float std) {
        double EPSILONE = 1e-12;
        for (int t = 0 ; t < 8 ; t++) {
            double u1 = 1 - data[t];
            double u2 = data[t + 8];
            double radius = Math.sqrt(-2 * Math.log(u1 + EPSILONE));
            double theta = (float) (2.0 * Math.PI * u2);
            data[t] = (float) (radius * Math.cos(theta) * (double) std + (double) mean);
            data[t + 8] = (float) (radius * Math.sin(theta) * (double) std + (double) mean);
        }
    }

    public void normal_fill(float[] data, int numel, float mean, float std) {
        for (int t = 0 ; t < numel ; t++) {
            data[t] = randfloat32();
        }
        for (int i = 0 ; i < numel - 15 ; i += 16) {
            normal_fill_16(data, mean, std);
        }
        if (numel % 16 != 0) {
            // recompute the last 16 values
            int last16 = numel - 16;
            for (int i = 0 ; i < 16 ; i++) {
                data[last16 + i] = randfloat32();
            }
            normal_fill_16(data, mean, std);
        }
    }

    private void normal(float[] data, int numel, float mean, float std) {
        double EPSILONE = 1e-12;
        if (numel >= 16) {
            normal_fill(data, numel, mean, std);
        }
        else {
            double next_double_normal_sample = 0.0; // make compiler warning happy, won't be used
            boolean has_next_double_normal_sample = false;
            for (int t = 0 ; t < numel ; t++) {
                if (has_next_double_normal_sample) {
                    data[t] = (float)(next_double_normal_sample * std + mean);
                    has_next_double_normal_sample = false;
                    continue;
                }
                // for numel < 16 we draw a double (float64)
                double u1 = (double) randfloat64();
                double u2 = (double) randfloat64();
                double radius = Math.sqrt(-2 * Math.log(1 - u2 + EPSILONE));
                double theta = 2.0 * Math.PI * u1;
                next_double_normal_sample = radius * Math.sin(theta);
                has_next_double_normal_sample = true;
                data[t] = (float) (radius * Math.cos(theta) * (double) std + (double) mean);
            }
        }
    }

    public void init_identity_permutation(int[] data, int numel) {
        for (int i = 0 ; i < numel ; i++) {
            data[i] = i;
        }
    }

    public void random_permutation(int[] data, int numel) {
        for (int i = numel - 1 ; i > 0 ; i--) {
            // pick an index j in [0, i] with equal probability
            int j = (int) (randint32() % (i + 1));
            // swap i <-> j
            int tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }

    public static void test_Mt19937() {
        Mt19937 state = new Mt19937();
        state.manual_seed(137);
        System.out.println(state.randint32());
        System.out.println(state.randint32());
        System.out.println(state.randint32());
        System.out.println(state.randint32());
        System.out.println(state.randint32());

        float t8[] = new float[8];
        state.normal(t8, 8, 0, 1);
        for (int i = 0; i < 8; i++) {
            System.out.println(t8[i]);
        }
        System.out.println(state.randint32());

        float t16[] = new float[16];
        state.normal(t16, 16, 0, 1);
        for (int i = 0; i < 16; i++) {
            System.out.println(t16[i]);
        }
        System.out.println(state.randint32());
    }
}

# llm.java
A Java port of Andrej Karpathy‘s llm.c that uses TornadoVM for parallelization on accelarators.

## Quick start (Windows)
- Clone [llm.c](https://github.com/karpathy/llm.c) and follow instructions given there in README, section [quick start (CPU)](https://github.com/otabuzzman/llm.c/blob/2346cdac931f544d63ce816f7e3f5479a917eef5/README.md#quick-start-cpu). This will get you the dataset, the tokens, the small GPT-2 model (124M) released by OpenAI, and two executables for testing and training.

- Clone this repository, open in VS Code, adjust path to TornadoVM's JARs in `launch.json`, and build the executables for testing and training.

- Copy `*.bin` files from `llm.c` into this directory and run the test class in a terminal window.

```
# assuming `llm.c´ is a sibling folder of `llm.java´ (CWD)
copy ..\llm.c\*.bin .

# assuming TornadoVM build in sibling folder took
..\TornadoVM\setvars.cmd

python %TORNADO_SDK%\bin\tornado --jvm="-Ds0.device=0:0" --classpath bin com.otabuzzman.llmj.TestGpt2
```

## Output samples
Output of Java's `TrainGpt2`:
```
PS C:\Users\iuerg\lab\llm.java>  & 'C:\Program Files\Java\graalvm-jdk-21.0.1+12.1\bin\java.exe' '-XX:+ShowCodeDetailsInExceptionMessages' '-cp' 'C:\Users\iuerg\lab\llm.java\bin' 'com.otabuzzman.llmj.TrainGpt2'
[GPT-2]
max_seq_len: 1024
vocab_size: 50257
padded_vocab_size: 50304
num_layers: 12
num_heads:12
channels: 768
num_parameters: 124475904
train dataset num_batches: 1192
val dataset num_batches: 128
num_activations: 73347840
val loss 5.325414
step 0: train loss 4,677650 (took 22101 ms)
step 1: train loss 5,190830 (took 21196 ms)
step 2: train loss 4,437739 (took 30378 ms)
step 3: train loss 4,138213 (took 26414 ms)
step 4: train loss 4,144530 (took 24980 ms)
step 5: train loss 3,834245 (took 20758 ms)
step 6: train loss 4,297797 (took 20040 ms)
step 7: train loss 4,280396 (took 19912 ms)
step 8: train loss 4,249562 (took 20158 ms)
step 9: train loss 4,392187 (took 19768 ms)
val loss 4.415881
step 10: train loss 3,911200 (took 19623 ms)
step 11: train loss 3,738629 (took 21031 ms)
step 12: train loss 3,840885 (took 21411 ms)
step 13: train loss 4,367395 (took 22501 ms)
step 14: train loss 4,130939 (took 20762 ms)
step 15: train loss 4,013472 (took 19802 ms)
step 16: train loss 3,796320 (took 20225 ms)
step 17: train loss 4,357059 (took 21337 ms)
step 18: train loss 3,766620 (took 19560 ms)
step 19: train loss 4,552443 (took 21122 ms)
val loss 4.331538
generating:
---
I am Rouset:As for my brother, my brother,I hope be pursued.Dearest, hear me speak.

<|endoftext|>JOINED ART:One's addressed as 'Governor Chapel,'Nay, I weary our NHS asButhow could feminine Enter
---
step 20: train loss 4,530054 (took 20144 ms)
step 21: train loss 4,067400 (took 20003 ms)
step 22: train loss 3,969011 (took 20434 ms)
step 23: train loss 3,450137 (took 19910 ms)
step 24: train loss 4,493859 (took 19825 ms)
step 25: train loss 4,036771 (took 20276 ms)
step 26: train loss 3,442927 (took 20059 ms)
step 27: train loss 3,993678 (took 21665 ms)
step 28: train loss 4,201677 (took 21948 ms)
step 29: train loss 4,542441 (took 21189 ms)
val loss 4.302901
step 30: train loss 4,306012 (took 20070 ms)
step 31: train loss 4,854148 (took 19888 ms)
step 32: train loss 4,583712 (took 19720 ms)
step 33: train loss 4,122869 (took 19956 ms)
step 34: train loss 4,334546 (took 19969 ms)
step 35: train loss 3,401595 (took 20166 ms)
step 36: train loss 3,661730 (took 19804 ms)
step 37: train loss 3,331205 (took 20283 ms)
step 38: train loss 3,570896 (took 19882 ms)
step 39: train loss 3,904977 (took 20224 ms)
val loss 4.2940817
generating:
---
Maduki Kenya:Good, sir!

<|endoftext|>DESONATION:Thus said you, madam;And now.<|endoftext|>RENU:As distinguished as we may choose to speak. Romeo and Juliet,Dost thou say, children, I mean, of charms?With her
---
step 40: train loss 3,953481 (took 19861 ms)
PS C:\Users\iuerg\lab\llm.java>
```

Output of Java's `TestGpt2`:
```
PS C:\Users\iuerg\lab\llm.java>  & 'C:\Program Files\Java\graalvm-jdk-21.0.1+12.1\bin\java.exe' '-XX:+ShowCodeDetailsInExceptionMessages' '-cp' 'C:\Users\iuerg\lab\llm.java\bin' 'com.otabuzzman.llmj.TestGpt2'
[GPT-2]
max_seq_len: 1024
vocab_size: 50257
padded_vocab_size: 50304
num_layers: 12
num_heads:12
channels: 768
num_parameters: 124475904
[State]
batch_size: 4
seq_len: 64
num_activations: 73347840
-43,431618, -43,431736
-39,836346, -39,836449
-43,065910, -43,066032
-42,828045, -42,828136
-43,529541, -43,529652
-44,318398, -44,318516
-41,227425, -41,227539
-41,270760, -41,270870
-42,541393, -42,541531
-42,394997, -42,395123
OK (LOGITS), max_diff = 1,953125e-03
LOSS OK: 5,269891 5,270009
dwte
OK -0,002320 -0,002320
OK 0,002072 0,002072
OK 0,003716 0,003717
OK 0,001307 0,001307
OK 0,000631 0,000632
TENSOR OK, maxdiff = 1,361847e-03
dwpe
OK -0,005118 -0,005110
OK -0,000001 -0,000012
OK -0,003267 -0,003262
OK 0,009909 0,009909
OK 0,002155 0,002145
TENSOR OK, maxdiff = 5,414989e-05
dln1w
OK -0,007520 -0,007523
OK 0,008624 0,008643
OK 0,005004 0,005029
OK -0,011098 -0,011095
OK -0,001666 -0,001664
TENSOR OK, maxdiff = 3,606715e-03
dln1b
OK -0,038494 -0,038458
OK -0,030547 -0,030600
OK 0,010189 0,010223
OK 0,080134 0,080176
OK -0,060990 -0,060901
TENSOR OK, maxdiff = 1,532087e-03
dqkvw
OK -0,000031 -0,000031
OK -0,000026 -0,000025
OK -0,000064 -0,000064
OK 0,000074 0,000074
OK 0,000020 0,000020
TENSOR OK, maxdiff = 5,576834e-04
dqkvb
OK -0,000414 -0,000411
OK -0,000410 -0,000412
OK 0,000113 0,000113
OK -0,000564 -0,000565
OK 0,000574 0,000570
TENSOR OK, maxdiff = 3,139013e-04
dattprojw
OK 0,000081 0,000080
OK -0,000005 -0,000005
OK -0,000019 -0,000019
OK 0,000005 0,000004
OK 0,000031 0,000031
TENSOR OK, maxdiff = 2,251565e-04
dattprojb
OK 0,000456 0,000470
OK -0,009969 -0,009979
OK -0,001794 -0,001804
OK 0,037638 0,037584
OK -0,031287 -0,031239
TENSOR OK, maxdiff = 2,020858e-04
dln2w
OK -0,018372 -0,018312
OK 0,004811 0,004813
OK 0,008084 0,008091
OK -0,001465 -0,001470
OK -0,002740 -0,002737
TENSOR OK, maxdiff = 1,153713e-02
dln2b
OK -0,026405 -0,026368
OK -0,016712 -0,016695
OK 0,001067 0,001074
OK 0,034754 0,034711
OK -0,028630 -0,028584
TENSOR OK, maxdiff = 9,741783e-04
dfcw
OK 0,000438 0,000440
OK -0,000000 -0,000000
OK -0,000153 -0,000154
OK -0,000165 -0,000165
OK 0,000404 0,000405
TENSOR OK, maxdiff = 9,582713e-04
dfcb
OK 0,003282 0,003293
OK 0,002038 0,002043
OK -0,001386 -0,001386
OK 0,000381 0,000386
OK 0,001602 0,001604
TENSOR OK, maxdiff = 2,334290e-04
dfcprojw
OK 0,000678 0,000681
OK 0,000073 0,000073
OK -0,000415 -0,000416
OK -0,000059 -0,000061
OK -0,000603 -0,000604
TENSOR OK, maxdiff = 4,582697e-04
dfcprojb
OK 0,003573 0,003584
OK -0,007148 -0,007158
OK -0,001955 -0,001964
OK 0,001466 0,001462
OK 0,001219 0,001217
TENSOR OK, maxdiff = 1,408615e-04
dlnfw
OK -0,000022 -0,000022
OK 0,000811 0,000811
OK 0,001161 0,001161
OK -0,002956 -0,002957
OK 0,001146 0,001145
TENSOR OK, maxdiff = 3,448352e-04
dlnfb
OK -0,011101 -0,011101
OK 0,008007 0,008007
OK -0,004763 -0,004769
OK -0,002110 -0,002113
OK -0,005903 -0,005905
TENSOR OK, maxdiff = 6,372365e-05
step 0: loss 5,269891 (took 16854 ms) OK = true
step 1: loss 4,059389 (took 18153 ms) OK = true
step 2: loss 3,374211 (took 18515 ms) OK = true
step 3: loss 2,800129 (took 18642 ms) OK = true
step 4: loss 2,315314 (took 18633 ms) OK = true
step 5: loss 1,849349 (took 18574 ms) OK = true
step 6: loss 1,395219 (took 19108 ms) OK = true
step 7: loss 0,998618 (took 18891 ms) OK = true
step 8: loss 0,625539 (took 18674 ms) OK = true
step 9: loss 0,378014 (took 18692 ms) OK = true
overall okay: true
PS C:\Users\iuerg\lab\llm.java>
```


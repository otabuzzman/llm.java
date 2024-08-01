# llm.java
A Java port of Andrej Karpathy‘s llm.c.

## Quick start
- Clone [llm.c](https://github.com/karpathy/llm.c) and follow instructions given there in README, section [quick start (CPU)](https://github.com/otabuzzman/llm.c/blob/2346cdac931f544d63ce816f7e3f5479a917eef5/README.md#quick-start-cpu). This will get you the dataset, the tokens, the small GPT-2 model (124M) released by OpenAI, and two executables for testing and training.

- Clone this repository, open in VS Code, build and run the executables for testing and training.

## Output samples

Output of Java's `TrainGpt2`:
```
```

Output of Java's `TestGpt2`:
```
PS C:\Users\SchuckJürgen(Group)\lab\llm.java>  & 'C:\Program Files\Java\Zulu\JDK-21\bin\java.exe' '-agentlib:jdwp=transport=dt_socket,server=n,suspend=y,address=localhost:64371' '-XX:+ShowCodeDetailsInExceptionMessages' '-cp' 'C:\Users\SchuckJürgen(Group)\lab\llm.java\bin' 'com.otabuzzman.llmj.TestGpt2'
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
step 0: loss 5,269891 (took 107280 ms) OK = true
step 1: loss 4,059389 (took 95589 ms) OK = true
step 2: loss 3,374211 (took 90826 ms) OK = true
step 3: loss 2,800129 (took 90924 ms) OK = true
step 4: loss 2,315314 (took 89474 ms) OK = true
step 5: loss 1,849349 (took 80289 ms) OK = true
step 6: loss 1,395219 (took 93374 ms) OK = true
step 7: loss 0,998618 (took 79436 ms) OK = true
step 8: loss 0,625539 (took 78807 ms) OK = true
step 9: loss 0,378014 (took 77934 ms) OK = true
overall okay: true
PS C:\Users\SchuckJürgen(Group)\lab\llm.java>
```

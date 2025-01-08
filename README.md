# llm.java
A Java port of Andrej Karpathy‘s llm.c.

## Quick start
- Clone [llm.c](https://github.com/karpathy/llm.c) and follow instructions given there in README, section [quick start (CPU)](https://github.com/otabuzzman/llm.c/blob/2346cdac931f544d63ce816f7e3f5479a917eef5/README.md#quick-start-cpu). This will get you the dataset, the tokens, the small GPT-2 model (124M) released by OpenAI, and two executables for testing and training.

- Clone this repository, open in VS Code, build and run the executables for testing and training.

The [samples.md](samples-md) file provides the output of llm.java captured from the first working version with Java Stream parallelization on a Lenovo T15p notebook. There is a [blog](https://otabuzzman.com/posts/tornado-llmc/) on parallelization with [TornadoVM](https://www.tornadovm.org/).

## Acknowledgements

Andrej Karpathy - (llm.c)
<br>Copyright (c) 2024 Andrej Karpathy - MIT License

**Java implementation**

Harry Jackson - (Java implementation)
<br>Copyright (c) 2024 Harry Jackson - MIT License

Adopted file and endian handling from [llm.java](https://github.com/harryjackson/llm.java) shared by [@harryjackson](https://github.com/harryjackson).

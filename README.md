# llm.java
A Java port of Andrej Karpathy‘s [llm.c](https://github.com/karpathy/llm.c) that uses TornadoVM for parallelization on accelarators.

## Quick start
- Clone, build and test TornadoVM according to [installation & configuration guide](https://tornadovm.readthedocs.io/en/latest/installation.html)

- Clone and build this repository in sibling of TornadoVM folder
  ```
  git clone https://github.com/otabuzzman/llm.java.git
  cd llm.java
  
  git checkout tornado
  
  make compile
  ```

- Check and adjust TornadoVM devices in `Makefile`

- Run `make test`

## Output samples

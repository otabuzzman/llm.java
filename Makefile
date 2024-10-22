ifeq ($(OS),Windows_NT)
	winos := 1
else
	linos := 1
endif

GPT2BINS = \
	gpt2_124M.bin \
	gpt2_124M_debug_state.bin \
	gpt2_tokenizer.bin \

.PHONY: compile run

compile:
ifdef winos
	javac \
		--enable-preview \
		--add-modules jdk.incubator.vector \
		-target 21 -source 21 \
		-Xlint:preview -proc:full \
		-d bin \
		src\com\otabuzzman\llmj\*.java
else
	javac \
		--enable-preview \
		--add-modules jdk.incubator.vector \
		-target 21 -source 21 \
		-Xlint:preview -proc:full \
		-d bin \
		src/com/otabuzzman/llmj/*.java
endif

run: $(GPT2BINS)
	java --enable-preview --add-modules jdk.incubator.vector -DUseVectorAPI=true --classpath bin com.otabuzzman.llmj.TestGpt2



gpt2_124M.bin:
	wget -q -O $@ https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_124M.bin
gpt2_124M_debug_state.bin:
	wget -q -O $@ https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_124M_debug_state.bin
gpt2_tokenizer.bin:
	wget -q -O $@ https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_tokenizer.bin

ifeq ($(OS),Windows_NT)
	winos := 1
else
	linos := 1
endif

GPT2BINS = \
	gpt2_124M.bin \
	gpt2_124M_debug_state.bin \
	gpt2_tokenizer.bin \

.PHONY: compile test

compile:
ifdef winos
	javac \
		-classpath "..\TornadoVM\bin\sdk\share\java\tornado\*" \
		--enable-preview \
		--add-modules jdk.incubator.vector \
		-target 21 -source 21 \
		-Xlint:preview -proc:full \
		-d bin \
		src\com\otabuzzman\llmj\*.java
else
	javac \
		-classpath "../TornadoVM/bin/sdk/share/java/tornado/*" \
		--enable-preview \
		--add-modules jdk.incubator.vector \
		-target 21 -source 21 \
		-Xlint:preview -proc:full \
		-d bin \
		src/com/otabuzzman/llmj/*.java
endif

test: $(GPT2BINS)
ifdef winos
	python %TORNADO_SDK%\bin\tornado --jvm="-Dtb.device=1:0 -Dol.device=2:0 -DUseVectorAPI=true -Dtornado.device.memory=2GB" --classpath bin com.otabuzzman.llmj.TestGpt2
else
	tornado --jvm="-Dtb.device=1:0 -Dol.device=2:0 -DUseVectorAPI=true -Dtornado.device.memory=2GB" --classpath bin com.otabuzzman.llmj.TestGpt2
endif



gpt2_124M.bin:
	wget -q -O $@ https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_124M.bin
gpt2_124M_debug_state.bin:
	wget -q -O $@ https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_124M_debug_state.bin
gpt2_tokenizer.bin:
	wget -q -O $@ https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_tokenizer.bin

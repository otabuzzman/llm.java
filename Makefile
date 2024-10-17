.PHONY: compile

compile:
	javac \
		-classpath "..\TornadoVM\bin\sdk\share\java\tornado\*" \
		--enable-preview \
		--add-modules jdk.incubator.vector \
		-target 21 -source 21 \
		-Xlint:preview -proc:full \
		-d bin \
		src\com\otabuzzman\llmj\*.java

#!/bin/bash

# Set the classpath
export CLASSPATH=./src

# Compile the MyCompiler class
javac src/MyCompiler.java

# Compile the MyJRE class
javac src/MyJRE.java

# Compile the HelloWorld class
javac src/HelloWorld.java

# Run the HelloWorld class
java -cp ./src HelloWorld

# INTRO basic understanding

This Code serves as an introduction of the CUDA toolkit

## Device Query

The code for the first section is given in `device_query.cu`. This code will query the device and print out some information about it. You will need to compile and ru.

## Vector Addition

### Generating the data
 A good way to first undertstand GPU vs CPU would be to first implement a CPU version of the algorithm and then compare that output with your GPU implementation.Both the CPU and GPU implementation is already provided.
To generate the data for this code, you will need to run the following command:
```
make datagen
./build/datagen
```

### Completing the main function
A starting template has been provided in `main.cu`. This outlines the basic process of allocating memory, copying data, executing the kernel, and reading the result. You can build the final executable with `make`. The following command will run the executable and compare the result to the expected output:

```
./build/main -e data/0/output.raw -i data/0/input0.raw,data/0/input1.raw -t vector
```

You can also compare the test Data. You can do this by changing the number in the `data/0` directory to the appropriate number.

The following command will also run the data in background:
```
sbatch run.sh
```

## Thank you
* I would like to thank Professor Dillhoff, for teaching class of Spring 2025, learning GPU was very fun and worth studying.
NOTE: Some of the code were given by the professor and implemented by me.


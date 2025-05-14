#include <gputk.h>
#include <cublas.h>

#include "device_query.cuh"
#include "vec_add.cuh"

int main(int argc, char **argv) {
    gpuTKArg_t args;
    int inputLength;
    float *hostInput1; //Self reminder, Host = CPU , Device = GPU
    float *hostInput2;
    float *hostOutput;
    float *deviceInput1;
    float *deviceInput2;
    float *deviceOutput;

    args = gpuTKArg_read(argc, argv);

    // TODO: Query and print the device properties here. This will help you answer the questions in Questions.md
    device_query();    


    gpuTKTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 =
        (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &inputLength);
    hostInput2 =
        (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *)malloc(inputLength * sizeof(float));
    gpuTKTime_stop(Generic, "Importing data and creating memory on host");

    gpuTKLog(TRACE, "The input length is ", inputLength);

    gpuTKTime_start(GPU, "Allocating GPU memory.");
    // TODO: Allocate GPU memory here
    
    cudaMalloc(&deviceInput1, inputLength*sizeof(float));
    cudaMalloc(&deviceInput2, inputLength*sizeof(float));
    cudaMalloc(&deviceOutput, inputLength*sizeof(float));

    gpuTKTime_stop(GPU, "Allocating GPU memory.");

    gpuTKTime_start(GPU, "Copying input memory to the GPU.");
    // TODO: Copy memory to the GPU here

    cudaMemcpy(deviceInput1, hostInput1, inputLength*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength*sizeof(float), cudaMemcpyHostToDevice);
    //=> Might not need => cudaMemcpy(deviceOutput, hostOutput, inputLength*sizeof(float), cudaMemcpyHostToDevice);

    gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

    // TODO: Initialize the grid and block dimensions here
    dim3 threadsPerBlock(64, 1, 1); //CHANGE this ... not sure what size I should use, 16, 128, 256??
    dim3 blocksPerGrid(1024, 1, 1); // Change this.
//32, 16

    gpuTKTime_start(Compute, "Performing CUDA computation");
    // TODO: Launch the GPU Kernel here

    vec_add<<<blocksPerGrid, threadsPerBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength); 


    gpuTKTime_stop(Compute, "Performing CUDA computation");

    gpuTKTime_start(Copy, "Copying output memory to the CPU");
    // TODO: Copy the GPU memory back to the CPU here

    cudaMemcpy(hostOutput, deviceOutput, inputLength*sizeof(float), cudaMemcpyDeviceToHost);


    gpuTKTime_stop(Copy, "Copying output memory to the CPU");

    gpuTKTime_start(GPU, "Freeing GPU Memory");
    // TODO: Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);


    gpuTKTime_stop(GPU, "Freeing GPU Memory");

    gpuTKSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}

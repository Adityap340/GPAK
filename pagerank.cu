#include <cuda_runtime.h>
#include <iostream>

extern "C" __global__ void pagerank_kernel(float *d_graph, float *d_pagerank, float *d_temp, int num_nodes, float d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        d_temp[idx] = 0.0f;
        for (int j = 0; j < num_nodes; j++) {
            if (d_graph[idx * num_nodes + j] != 0.0f) {
                if (d_graph[j * num_nodes + idx] != 0.0f) {
                    d_temp[idx] += d_pagerank[j] / d_graph[j * num_nodes + idx];
                }
            }
        }
        d_temp[idx] = (1.0f - d) + d * d_temp[idx];
    }
}

extern "C" __declspec(dllexport) void run_pagerank(float *graph, int num_nodes, float d, int max_iterations, float *pagerank_scores) {
    float *d_graph, *d_pagerank, *d_temp;

    // Allocate memory on the GPU with error checking
    if (cudaMalloc((void**)&d_graph, num_nodes * num_nodes * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**)&d_pagerank, num_nodes * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**)&d_temp, num_nodes * sizeof(float)) != cudaSuccess) {
        std::cerr << "Error allocating memory on the GPU." << std::endl;
        return;
    }

    // Copy graph and initial PageRank scores to GPU
    cudaMemcpy(d_graph, graph, num_nodes * num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pagerank, pagerank_scores, num_nodes * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (num_nodes + block_size - 1) / block_size;

    // PageRank iterations
    for (int i = 0; i < max_iterations; i++) {
        pagerank_kernel<<<grid_size, block_size>>>(d_graph, d_pagerank, d_temp, num_nodes, d);
        
        // Synchronize and check for errors
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) {
            std::cerr << "Error in kernel execution." << std::endl;
            cudaFree(d_graph);
            cudaFree(d_pagerank);
            cudaFree(d_temp);
            return;
        }

        // Copy result back to pagerank_scores after each iteration
        cudaMemcpy(pagerank_scores, d_temp, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);

        // Normalize scores to avoid overflow
        float sum = 0.0f;
        for (int j = 0; j < num_nodes; j++) {
            sum += pagerank_scores[j];
        }
        for (int j = 0; j < num_nodes; j++) {
            pagerank_scores[j] /= sum;
        }
    }

    // Free GPU memory
    cudaFree(d_graph);
    cudaFree(d_pagerank);
    cudaFree(d_temp);
}
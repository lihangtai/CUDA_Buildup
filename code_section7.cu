inline cudaError_t cudaMallocMapped( void** ptr, size_t size, bool clear=true )
{
	void* cpu = NULL;
	void* gpu = NULL;

	if( !ptr || size == 0 )
		return cudaErrorInvalidValue;

	//CUDA_ASSERT(cudaSetDeviceFlags(cudaDeviceMapHost));

    CUDA_ASSERT(cudaHostAlloc(&cpu, size, cudaHostAllocMapped));
    CUDA_ASSERT(cudaHostGetDevicePointer(&gpu, cpu, 0));

    if( cpu != gpu )
    {
        LogError(LOG_CUDA "cudaMallocMapped() - addresses of CPU and GPU pointers don't match (CPU=%p GPU=%p)\n", cpu, gpu);
        return cudaErrorInvalidDevicePointer;
    }
    
    if( clear )
	    memset(cpu, 0, size);

    *ptr = cpu;
	return cudaSuccess;
}

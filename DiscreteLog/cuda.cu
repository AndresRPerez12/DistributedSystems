// nvcc cuda.cu -o cuda
#include <bits/stdc++.h>
#include <sys/time.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

using namespace std;
typedef __int128 i128;
#define MAX_NODES 10000

string print(i128 x) {
    string ret = "";
    if( x >= (i128)10 ) ret += print(x / (i128)10);
    ret += char(x % (i128)10 + '0');
    return ret;
}

i128 ceil_division(i128 num, i128 den){
    return (num + den - (i128)1) / den;
}

i128 hostFastExpo( i128 base, i128 expo, i128 m ){
    if( expo == 0 ) return 1;
    i128 prv_expo = expo/(i128)2;
    i128 ret = hostFastExpo(base, prv_expo, m);
    ret = (ret * ret)%m;
    if( expo%(i128)2 ) ret = (ret * base)%m;
    return ret;
}

__device__ i128 fastExpo( i128 base, i128 expo, i128 m ){
    if( expo == 0 ) return 1;
    i128 prv_expo = expo/(i128)2;
    i128 ret = fastExpo(base, prv_expo, m);
    ret = (ret * ret)%m;
    if( expo%(i128)2 ) ret = (ret * base)%m;
    return ret;
}

__device__ i128 function_1( i128 a, i128 n, i128 p, i128 m ){
    i128 expo = (n * p)%m;
    return fastExpo(a, expo, m);
}

__device__ i128 function_2( i128 a, i128 b, i128 q, i128 m ){
    i128 ret = fastExpo(a, q, m);
    ret = (ret * b)%m;
    return ret;
}

__global__ void calculateFunction1(long long *a, long long *m, long long *n, long long *limit, long long *step, 
                                    long long *results, long long *keys){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = gridDim.x * blockDim.x;
    // printf("IN THREAD %d\n", threadId);

    i128 low = (*step) * threadId + (i128)1;
    i128 high = low + (*step) - (i128)1;
    if( threadId + 1 == numThreads ) high = (*limit);
    long long results_index = threadId * (*step);
    // printf("MIDDLE THREAD %d\n", threadId);
    for(i128 p = low ; p <= high ; p ++) {
        i128 value = function_1((*a), (*n), p, (*m));
        // INSERT TO RESULTS
        //printf("\tPut %d -> %d in %d\n", (int) p, (int) value, (int) results_index);
        results[results_index] = (long long) p;
        keys[results_index] = (long long) value;
        results_index ++;
    }
    // printf("OUT THREAD %d\n", threadId);
}

__device__ long long getEqualResult(long long limit, long long *results, long long *keys, long long target){
    int low = 0;
    int high = limit-1;
    int middle;
    while( low < high ){
        middle = (low+high+1)/2;
        if( keys[middle] <= target ) low = middle;
        else high = middle-1;
    }
    if(keys[low] == target) return results[low];
    return -1;
}

__global__ void calculateFunction2(long long *a, long long *b, long long *m, long long *n,
                                    long long *limit, long long *array_limit, long long *step, 
                                    long long *results, long long *keys, long long* x){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = gridDim.x * blockDim.x;
    // printf("IN F2 THREAD %d\n", threadId);

    i128 low = (*step) * threadId;
    i128 high = low + (*step) - (i128)1;
    if( threadId + 1 == numThreads ) high = (*limit);
    for(i128 q = low ; q <= high ; q ++) {
        long long value = (long long)function_2((*a), (*b), q, (*m));
        long long findP = getEqualResult((*array_limit), results, keys, value);
        // printf("\tTry %d -> %d :: %d\n", (int) q, (int) value, (int) findP);
        if( findP == -1 ) continue;
        i128 currentX = ((i128)(*n) * (i128)findP)%( (i128)(*m) );
        currentX = (currentX - q + (i128)(*m))%( (i128)(*m) );
        *x = (long long) currentX;
        // printf("\t FOUND X :: p=%d q=%d x=%d\n",(int)findP, (int)q, (int)currentX);
    }
    // printf("OUT F2 THREAD %d\n", threadId);
}

__global__ void print_arrays(long long *limit, long long *results, long long *keys){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    // int numThreads = gridDim.x * blockDim.x;
    printf("IN PRINT THREAD %d\n", threadId);
    for(int i = 0 ; i < (*limit) ; i ++){
        printf("\tpos %d -> %d :: %d\n",i, (int)keys[i], (int)results[i]);
    }
    printf("OUT PRINT THREAD %d\n", threadId);
}

int main(){

    i128 a = 5;
    i128 gen_x = 14;
    i128 m = 37;
    i128 b = hostFastExpo(a, gen_x, m);

    printf("TEST\n");

    int blocks = 2;
    int threadsPerBlock = 2;
    int numThreads = blocks * threadsPerBlock;

    // Host variables
    long long host_a = a;
    long long host_b = b;
    long long host_m = m;
    long long host_n = sqrt((long double) m);
    long long host_limit = ceil_division(host_m,host_n);
    long long host_step = host_limit / (long long) numThreads;
    long long host_x = -1;

    int var_size = sizeof( long long );
    int array_value_size = host_n + 5;
    int array_size = array_value_size * sizeof( long long );

    // Device variables
    long long *device_a;
    long long *device_b;
    long long *device_m;
    long long *device_n;
    long long *device_limit;
    long long *device_step;
    long long *device_results;
    long long *device_keys;
    long long *device_x;
    long long *device_array_limit;

    cudaMalloc( (void**)&device_a , var_size );
    cudaMalloc( (void**)&device_b , var_size );
    cudaMalloc( (void**)&device_m , var_size );
    cudaMalloc( (void**)&device_n , var_size );
    cudaMalloc( (void**)&device_limit , var_size );
    cudaMalloc( (void**)&device_step , var_size );
    cudaMalloc( (void**)&device_results , array_size );
    cudaMalloc( (void**)&device_keys , array_size );
    cudaMalloc( (void**)&device_x , var_size );
    cudaMalloc( (void**)&device_array_limit , var_size );

    cudaMemcpy( device_a , &host_a , var_size , cudaMemcpyHostToDevice );
    cudaMemcpy( device_b , &host_b , var_size , cudaMemcpyHostToDevice );
    cudaMemcpy( device_m , &host_m , var_size , cudaMemcpyHostToDevice );
    cudaMemcpy( device_n , &host_n , var_size , cudaMemcpyHostToDevice );
    cudaMemcpy( device_limit , &host_limit , var_size , cudaMemcpyHostToDevice );
    cudaMemcpy( device_step , &host_step , var_size , cudaMemcpyHostToDevice );
    cudaMemcpy( device_x , &host_x , var_size , cudaMemcpyHostToDevice );

    // Call the Function 1 kernel
    calculateFunction1<<<blocks,threadsPerBlock>>>(
        device_a, device_m, device_n, device_limit, device_step,
        device_results, device_keys);
    cudaDeviceSynchronize();

    // Sort results and keys
    thrust::device_ptr<long long> thrust_keys(device_keys);
    thrust::device_ptr<long long> thrust_results(device_results);
    thrust::sort_by_key(thrust::device, thrust_keys, thrust_keys + host_limit, thrust_results);
    cudaDeviceSynchronize();

    // Check sort process
    // print_arrays<<<1,1>>>(device_limit, device_results, device_keys);
    // cudaDeviceSynchronize();

    // Call the Function 2 kernel
    cudaMemcpy( device_array_limit , &host_limit , var_size , cudaMemcpyHostToDevice );
    host_limit = host_n;
    host_step = host_limit / (long long) numThreads;
    cudaMemcpy( device_limit , &host_limit , var_size , cudaMemcpyHostToDevice );
    cudaMemcpy( device_step , &host_step , var_size , cudaMemcpyHostToDevice );

    calculateFunction2<<<blocks,threadsPerBlock>>>(
        device_a, device_b, device_m, device_n, device_limit, device_array_limit, device_step,
        device_results, device_keys, device_x);
    cudaDeviceSynchronize();

    cudaMemcpy( &host_x , device_x , var_size , cudaMemcpyDeviceToHost );
    cout << "FOUND X " << host_x << " :: " << host_a  << " ^ " << host_x << " =? " << host_b << " mod " << host_m << "\n" ;
    cout << host_a  << " ^ " << host_x << " = " << (long long)hostFastExpo(a,host_x,m) << "\n" ;

    // Free device memory
    cudaFree( device_a );
    cudaFree( device_b );
    cudaFree( device_m );
    cudaFree( device_n );
    cudaFree( device_limit );
    cudaFree( device_step );
    cudaFree( device_results );
    cudaFree( device_keys );
    cudaFree( device_x );
    cudaFree( device_array_limit );

    printf("POST TEST\n");
    
}
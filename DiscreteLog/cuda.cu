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

i128 hostFastExpo( i128 &base, i128 &expo, i128 &m ){
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

// __global__ void calculateFunction1(thrust::device_vector<long long> &results){
__global__ void calculateFunction1(){
    i128 var = fastExpo(5,3,37);
    printf("%d\n", (int) var);
}

__global__ void calculateFunction2(){
    long long var = fastExpo(5,3,37);
}

int main(){

    // i128 a = 56439;
    // i128 gen_x = 15432465;
    // i128 m = 29996224275833;
    // i128 b = hostFastExpo(a, gen_x, m);

    int var_a = 22;
    printf("TEST %d\n",var_a);

    calculateFunction1<<<2,2>>>();
    cudaDeviceSynchronize();

    printf("POST TEST\n");
    
}
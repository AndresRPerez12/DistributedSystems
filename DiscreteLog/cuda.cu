// nvcc cuda.cu -o cuda
#include <bits/stdc++.h>
#include <sys/time.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

using namespace std;
typedef unsigned __int128 i128;

int main(){

    // generate random numbers serially
    thrust::host_vector<int> h_vec(50);
    std::generate(h_vec.begin(), h_vec.end(), rand);

    // transfer data to the device
    thrust::device_vector<int> d_vec = h_vec;
    // sort data on the device (846M keys per second on GeForce GTX 480)
    thrust::sort(d_vec.begin(), d_vec.end());

    thrust::host_vector<int> h_vec_b(50,0);
    // transfer data back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec_b.begin());

    for(int i = 0 ; i < h_vec_b.size() ; i ++){
        cout << h_vec_b[i] << " " ;
        if(i) assert(h_vec_b[i] > h_vec_b[i-1]);
    }
    cout << endl ;
    
}
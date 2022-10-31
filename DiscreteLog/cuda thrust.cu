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
    thrust::host_vector<int> h_vec( (int) sqrt(29996224275833) );
    std::generate(h_vec.begin(), h_vec.end(), rand);
    std::cout << "generate " << time(NULL) << endl;

    // transfer data to the device
    thrust::device_vector<int> d_vec = h_vec;
    cout << "copy to device " << time(NULL) << endl;

    // sort data on the device
    thrust::sort(d_vec.begin(), d_vec.end());
    std::cout << "sort in device " << time(NULL) << endl;

    // transfer data back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    std::cout << "copy to host " << time(NULL) << endl;

    // gen local
    vector<int> test( (int) sqrt(29996224275833) );
    std::generate(test.begin(), test.end(), rand);
    std::cout << "generate test " << time(NULL) << endl;

    // sort local
    sort(test.begin(),test.end());
    std::cout << "sort test " << time(NULL) << endl;
    
}
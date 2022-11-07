// g++ sequential.cpp -o secuential
#include <bits/stdc++.h>
#include <sys/time.h>

using namespace std;
typedef unsigned __int128 i128;

i128 fastExpo( i128 &base, i128 &expo, i128 &m ){
    if( expo == 0 ) return 1;
    i128 prv_expo = expo/(i128)2;
    i128 ret = fastExpo(base, prv_expo, m);
    ret = (ret * ret)%m;
    if( expo%(i128)2 ) ret = (ret * base)%m;
    return ret;
}

i128 function_1( i128 &a, i128 &n, i128 &p, i128 &m ){
    i128 expo = (n * p)%m;
    return fastExpo(a, expo, m);
}

i128 function_2( i128 &a, i128 &b, i128 &q, i128 &m ){
    i128 ret = fastExpo(a, q, m);
    ret = (ret * b)%m;
    return ret;
}

i128 ceil_division(i128 num, i128 den){
    return (num + den - (i128)1) / den;
}

i128 discreteLog( i128 a , i128 b , i128 m ){ // a^x = b mod m
    i128 n = sqrt((long double) m), x;
    bool finished = false;
    unordered_map<i128,i128> f1_results;
    
    for(i128 p = 1 ; p <= ceil_division(m,n) ; p ++) {
        i128 value = function_1(a, n, p, m);
        if(!f1_results.count(value)) f1_results[value] = p;
    }
    
    for(i128 q = 0 ; q <= n and !finished ; q ++) {
        i128 value = function_2(a, b, q, m);
        if( f1_results.count(value) ){
            i128 p = f1_results[value];
            //finished = true;
            x = (n * p)%m;
            x = (x - q + m)%m;
        }
    }

    return x;
}

int main(int argc, char* argv[]){
    i128 a, b, m;
    long long read_a, read_b, read_m;
    bool verbose_flag = 0;

    stringstream read_a_SS(argv[1]);
    read_a_SS >> read_a;

    stringstream read_b_SS(argv[2]);
    read_b_SS >> read_b;

    stringstream read_m_SS(argv[3]);
    read_m_SS >> read_m;

    stringstream read_flag_SS(argv[4]);
    read_flag_SS >> verbose_flag;

    a = read_a;
    b = read_b;
    m = read_m;

    if(verbose_flag)
        cout << "Solve " << (long long)a << "^x" << " = " << (long long)b
            << " mod " << (long long)m << endl ;
 
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);
    
    i128 x = discreteLog(a, b, m);

    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);

    if(verbose_flag)
        cout << (long long)a << "^" << (long long)x << " = " << (long long)b
            << " mod " << (long long)m << endl ;

    assert(fastExpo(a,x,m) == b);
    printf("%ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
}
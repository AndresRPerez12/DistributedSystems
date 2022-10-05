// g++ open_mp.cpp -o openMP -fopenmp
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
typedef unsigned __int128 i128;

string print(i128 x) {
    string ret = "";
    if( x >= (i128)10 ) ret += print(x / (i128)10);
    ret += char(x % (i128)10 + '0');
    return ret;
}

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

i128 calculate_x(i128 n, i128 p, i128 q, i128 m){
    i128 x = (n * p)%m;
    x = (x - q + m)%m;
    return x;
}

i128 discreteLog( i128 a , i128 b , i128 m ){ // a^x = b mod m
    i128 n = sqrt(m), x;
    bool finished = false;
    unordered_map<i128,i128> f1_results;
    cout << "n = " << print(n) << endl ;
    
    #pragma omp parallel for
    for(i128 p = 1 ; p <= ceil_division(m,n) ; p ++) {
        i128 value = function_1(a, n, p, m);
        if(!f1_results.count(value)) f1_results[value] = p;
    }

    cout << "END PRECALC" << endl ;
    
    #pragma omp parallel for
    for(i128 q = 0 ; q <= n ; q ++) {
        i128 value = function_2(a, b, q, m);
        if( f1_results.count(value) ){
            i128 p = f1_results[value];
            cout << "p = " << print(p) << " q = " << print(q) << endl ;
            finished = true;
            #pragma omp critical
            x = calculate_x(n, p, q, m);
        }
    }

    return x;
}

int main(){
    i128 a = 56439;
    i128 gen_x = 15432465;
    i128 m = 1000000009;
    i128 b = fastExpo(a, gen_x, m);

    cout << "Solve " << print(a) << "^x" << " = " << print(b) << " mod " << print(m) << endl ;
    
    i128 x = discreteLog(a, b, m);
    assert(fastExpo(a,x,m) == b);
    //cout << "AAA" << endl ;
    cout << print(a) << "^" << print(x) << " = " << print(b) << " mod " << print(m) << endl ;
}
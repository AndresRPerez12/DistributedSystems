// mpicxx -o log-mpi log-mpi.cpp
// mpirun -np 4 log-mpi
#include <bits/stdc++.h>
#include <sys/time.h>

#include "mpi.h"

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

int pRank, size;
const int array_size = 5476880 + 100;
long long f1_values[array_size];
long long f1_keys[array_size];
long long local_f1_values[array_size];
long long local_f1_keys[array_size];
i128 a, b, m, n;
long long x, proc_x;

void calculateFunction1(long long low, long long high){
    int position = 0;
    for(i128 p = low ; p <= high ; p ++) {
        long long value = function_1(a, b, p, m);
        local_f1_values[position] = value;
        local_f1_keys[position] = (long long) p;
        position ++;
    }
}

long long getEqualResult(long long target){
    int low = 0;
    int high = ceil_division(m,n)-1;
    int middle;
    while( low < high ){
        middle = (low+high+1)/2;
        if( f1_values[middle] <= target ) low = middle;
        else high = middle-1;
    }
    if(f1_values[low] == target) return f1_keys[low];
    return -1;
}

void calculateFunction2(long long low, long long high){
    proc_x = -1;
    for(i128 q = low ; q <= high and proc_x == -1 ; q ++) {
        long long value = function_2(a, b, q, m);
        long long findP = getEqualResult(value);
        if( findP == -1 ) continue;
        i128 currentX = (n * (i128)findP)%( m );
        currentX = (currentX - q + m)%( m );
        proc_x = (long long) currentX;
    }
}

bool sort_by_value( int a , int b ){
    return f1_values[a] < f1_values[b];
}

int indexes[array_size];
long long f1_values_copy[array_size];
long long f1_keys_copy[array_size];

void sort_arrays(){
    const int limit = ceil_division(m,n);
    printf("Process %d enters sort_arrays with %d\n",pRank, limit);
    printf("Process %d before copy and indexes\n",pRank);
    for( int i = 0 ; i < limit ; i ++ ){
        indexes[i] = i;
        f1_values_copy[i] = f1_values[i];
        f1_keys_copy[i] = f1_keys[i];
        printf("\t %d -> %lld with p=%lld\n", i, f1_values[i], f1_keys[i]);
    }
    printf("Process %d after copy and indexes\n",pRank);
    sort(indexes, indexes+limit, sort_by_value);
    printf("Process %d after sort call\n",pRank);
    for( int i = 0 ; i < limit ; i ++ ){
        f1_values[i] = f1_values_copy[indexes[i]];
        f1_values[i] = f1_keys_copy[indexes[i]];
    }
}

int main(int argc, char* argv[]){

    MPI_Init( &argc, &argv );
        int root = 0;
        MPI_Comm_rank( MPI_COMM_WORLD, &pRank );
        MPI_Comm_size( MPI_COMM_WORLD, &size );

        a = 5;
        b = 14;
        m = 37;
        n = sqrt((long double) m);
    
        struct timeval tval_before, tval_after, tval_result;

        if( pRank == root ){
            gettimeofday(&tval_before, NULL);
        }

        long long limit = ceil_division(m,n);
        long long step = limit/(long long)size;
        long long low = (long long)1 + ((long long) pRank * step);
        long long high = low + step - (long long)1;
        if( pRank + 1 == size ) high = limit;

        for(int i = 0 ; i < step+2 ; i ++){
            local_f1_values[i] = LLONG_MAX;
            local_f1_keys[i] = LLONG_MAX;
        }

        printf("Process %d before f1\n",pRank);

        calculateFunction1(low, high);
        printf("Process %d after f1\n",pRank);
        MPI_Barrier( MPI_COMM_WORLD );

        MPI_Gather ( local_f1_values, (int) step+2, MPI_LONG_LONG_INT, f1_values, (int)(step+2)*size,
                    MPI_LONG_LONG_INT, root, MPI_COMM_WORLD );
        MPI_Gather ( local_f1_keys, (int) step+2, MPI_LONG_LONG_INT, f1_keys, (int)(step+2)*size,
                    MPI_LONG_LONG_INT, root, MPI_COMM_WORLD );
        MPI_Barrier( MPI_COMM_WORLD );

        printf("Process %d after gather\n",pRank);

        if( pRank == root ){
            sort_arrays();
        }

        printf("Process %d after sort\n",pRank);
        MPI_Barrier( MPI_COMM_WORLD );

        MPI_Bcast( f1_values, array_size, MPI_LONG_LONG_INT, root, MPI_COMM_WORLD);
        MPI_Bcast( f1_keys, array_size, MPI_LONG_LONG_INT, root, MPI_COMM_WORLD);
        MPI_Barrier( MPI_COMM_WORLD );

        printf("Process %d after bcast\n",pRank);

        limit = n;
        step = limit/(long long)size;
        low = (long long) pRank * step;
        high = low + step - (long long)1;
        if( pRank + 1 == size ) high = limit;

        printf("Process %d before f2\n",pRank);

        calculateFunction2(low, high);
        printf("Process %d after f2\n",pRank);
        printf("Process %d FOUND X=%lld\n",pRank, proc_x);
        MPI_Barrier( MPI_COMM_WORLD );

        MPI_Reduce(&proc_x, &x, 1, MPI_LONG_LONG_INT, MPI_MAX, root, MPI_COMM_WORLD);
        MPI_Barrier( MPI_COMM_WORLD );

        if( pRank == root ){
            printf("FOUND X=%lld\n", x);
            gettimeofday(&tval_after, NULL);
            timersub(&tval_after, &tval_before, &tval_result);
            printf("%ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
        }
    MPI_Finalize( );
    
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);

    i128 testX = x;
    assert(fastExpo(a,testX,m) == b);
}
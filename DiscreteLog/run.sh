#!/bin/bash

# a^x = b mod m
# Sample values
#   a = 56439
#   b = 27341644544150
#   m = 29996224275833
#   result: x = 15432465
a=$1
b=$2
m=$3

# Secuential
echo "Start sequential"
g++ sequential.cpp -o sequential
total=0
for ((j=1; j<=5; j++))
do
    result=$(./sequential $a $b $m 0)
    total=`echo $total+$result | bc -l`
done
echo "Total" $total
avg=`echo $total/5.0 | bc -l`
echo "Average" $avg

# OpenMP
echo "Start OpenMP"
g++ open_mp.cpp -o open_mp -fopenmp -ltbb
for ((threads=1; threads<=16; threads*=2))
do
    echo "Threads:" $threads
    total=0
    for ((j=1; j<=5; j++))
    do
        result=$(./open_mp $a $b $m 0 $threads)
        total=`echo $total+$result | bc -l`
    done
    echo "Total" $total
    avg=`echo $total/5.0 | bc -l`
    echo "Average" $avg
done

#CUDA
echo "Start CUDA"
nvcc cuda.cu -o cuda
blocks=5
for ((threads=32; threads<=256; threads*=2))
do
    echo "Threads:" $threads
    total=0
    for ((j=1; j<=5; j++))
    do
        result=$(./cuda $a $b $m 0 $blocks $threads)
        total=`echo $total+$result | bc -l`
    done
    echo "Total" $total
    avg=`echo $total/5.0 | bc -l`
    echo "Average" $avg
done
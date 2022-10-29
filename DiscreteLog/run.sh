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
g++ sequential.cpp -o secuential
echo "Start sequential"
for ((j=1; j<=10; j++))
do
    result=$(./secuencial $a $b $m)
    total=$(python -c "print ($total + $result)")
done
echo "Total" $total
average=$(python -c "print ($total / 10)")
echo "Average" $average

# OpenMP
g++ open_mp.cpp -o open_mp -fopenmp -ltbb
echo "Start OpenMP"
for ((threads=1; threads<=16; threads*=2))
do
    echo "Threads:" $threads 
    for ((j=1; j<=10; j++))
    do
        result=$(./open_mp $threads $a $b $m)
        total=$(python -c "print ($total + $result)")
    done
    echo "Total" $total
    average=$(python -c "print ($total / 10)")
    echo "Average" $average
done
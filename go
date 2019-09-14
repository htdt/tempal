#!/bin/bash
rm -rf runs
for i in `seq 1 $1`; do
    python -m ppo.train $2
done 

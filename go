#!/bin/bash
rm -rf runs
for i in `seq 1 $1`; do
    python -m ppo.train --cfg $2 --env $3
done 

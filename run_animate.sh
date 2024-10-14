#!/bin/bash

MODEL_PATH="outputs/run/train/ffhq_30/colat.models.NonlinearConditional_colat.projectors.NonlinearProjector/alpha_1__BS_128/2024-09-26"
# MODEL_PATH="outputs/run/train/ffhq_5/colat.models.NonlinearConditional_colat.projectors.NonlinearProjector/2024-08-14"

# Run the Python script with the specified arguments
python animate.py \
    --config-path="$MODEL_PATH/.hydra" \
    --config-name=config \
    checkpoint="$MODEL_PATH/best_model.pt" \
    +n_samples=30 \
    +max_alpha=-20 \
    +alphas_step_size=2 \
    +iterative=False \
    +image_size=128 \
    +direction=20 \
    +desired_class="DMSO"




# [-5,-3,-1,1,3,5]
# [-7,-5,-3,3,5,7]
# [-15,-10,-5,5,10,15]



# [0,1,2,3,4,5,6,7,8,9]
# [10,11,12,13,14,15,16,17,18,19]
# [20,21,22,23,24,25,26,27,28,29]
# [30,31,32,33,34,35,36,37,38,39]
# [40,41,42,43,44,45,46,47,48,49]
# [50,51,52,53,54,55,56,57,58,59]
# [60,61,62,63,64,65,66,67,68,69]
# [70,71,72,73,74,75,76,77,78,79]
# [80,81,82,83,84,85,86,87,88,89]
# [90,91,92,93,94,95,96,97,98,99]
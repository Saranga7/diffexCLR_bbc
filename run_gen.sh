#!/bin/bash

MODEL_PATH="outputs/run/train/ffhq_5/colat.models.NonlinearConditional_colat.projectors.NonlinearProjector/alpha_1__BS_128/2024-10-25"
# MODEL_PATH="outputs/run/train/ffhq_5/colat.models.NonlinearConditional_colat.projectors.NonlinearProjector/2024-08-14"

# Run the Python script with the specified arguments
python gen.py \
    --config-path="$MODEL_PATH/.hydra" \
    --config-name=config \
    checkpoint="$MODEL_PATH/best_model.pt" \
    +n_samples=20 \
    +alphas="[-25,-20,-15,15,20,25]" \
    +iterative=False \
    +image_size=128 \
    +n_dirs=[0,1,2,3,4] \




# [-5,-3,-1,1,3,5]
# [-7,-5,-3,3,5,7]
# [-15,-10,-5,5,10,15]

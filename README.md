# Diff-Ex CLR on BBBC021 

## Setup

```
git clone https://github.com/Saranga7/diffexCLR_bbc.git
cd diffexCLR_bbc

conda env create -f environment.yml
conda activate diffexCLR
```

## Training

### Changes to be made in the `diffae` directory

1. Modify the `PretrainedClassifier` in `model/unet_autoenc.py` by copying the code of the `PretrainedClassifier` architecture from the [`diffex` repo](https://github.com/Saranga7/diffex_bbc). Also set the correct `classifier_path` in the `BeatGANsAutoencModel` class.

2. Modify the `templates.py` and `dataset.py` as per your dataset (copying from the diffex repo). 

3. Modify the `data_paths` dictionary and `make_dataset` method in `config.py`

4. Modify `conf = bbc_autoenc()` accordingly in `colate/runner.py` in the ``train``, ``generate``, ``att_find``, and ``animate`` functions.


<br>

You can set in `conf/train.yaml` what kind of direction model and projection you want to use (by default nonlinear).

In `conf/misc/train_misc.yaml` set the `diffae_ckpt_path`, the number of `epochs` and the index of the gpu in `device` (remember hydra config created by the training is saved and reused during generation, attfind, animation etc).

The `colat/runner.py` is the base control base from where the different tasks take place. Modify the wandb key and the project name as needed.


NOTE :

A lot of changes have been made in the `colat/trainer.py` and `colat/visualizer.py` scripts w.r.t to the original LatentCLR version. However, the `colat/attFinder.py` and `colat/animator.py` scripts have been written from scratch. The `colat/evaluator.py` script is untouched and unused. The other scripts inside `colat` are not modified, but used by the other scripts. The `colat/loss.py` contains the Contrastive Loss. Any modifcations in the loss must be done here (and perhaps also inside `colat/trainer.py` because that is where the loss is used).


Now, to train:

```
python train.py k=10 hparams.batch_size=128 model.alpha=1 
```

I use the following to store the training logs (however, a train.log is also created in the directory where the training is saved internally)
```
python train.py k=5 hparams.batch_size=128 model.alpha=1> output_logs/train_k5_diffaeclr_alpha1_BS128.log 2>&1
```

Trainigns are saved in the `outputs/run` directory.

<hr>

## Generating

In `run_gen.sh`

- It is crucial to set the correct `MODEL_PATH` after training is complete.

- `n_samples` means the number of samples you want to generate.

- `alphas` are the different magnitude shifts

- `n_dirs` is a list of the indices of the different direction models along which you want to shift. The indices should not exceed the total number of direction models, `k`, that were used to train with in the first place.

Then simply

```
bash run_gen.sh
```

<hr>

## AttFind

### Changes to be made in the `diffae` directory

1. Create the "Selective dataset" in `dataset.py`. 

2. Modify the `data_paths` dictionary and `make_dataset` method in `config.py` accordingly.

3. Modify `conf.data_name` accordingly in `colate/runner.py` in the ``train``, ``generate``, ``att_find``, and ``animate`` functions.

<br>

In `colat/attFinder.py`, modify the `if self.desired_class` condition as per the dataset you're adapting the code to. Do the same in `colat/animator.py`.

In `run_AttFind.sh`

- It is crucial to set the correct `MODEL_PATH` after training is complete.

- `threshold`- to start AttFind with. It later decrements with each iteration.

- `M` - the number of good directions (indices of direction models) desired.

- `desired_class` is the class, in this case (either ``DMSO`` or ``latrunculin_B_high_conc``) on which AttFind is applied.

Then simply


```
bash run_AttFind.sh
```

<hr>

## Animations


In `run_AttFind.sh`

- It is crucial to set the correct `MODEL_PATH` after training is complete.

- `n_samples`- the number of samples you want to animate.

- `max_alpha`- upto which alpha you want to shift.

- `alphas_step_size` - step size for incraseing alphas (both positively and negatively).

- `direction` - index of the direction model.

- `desired_class` is the class, in this case (either ``DMSO`` or ``latrunculin_B_high_conc``) on which animations will be generated. For eg. if ``DMSO``, then images of ``latrunculin_B_high_conc`` will be used, as they are shifted to increase ``DMSO`` probability.


Then simply

```
bash run_animate.sh
```
<br>

NOTE:

Each time you generate, attfind, or animate, the results will be stored inside the `outputs/<<date>>/<<time>>` directory. 


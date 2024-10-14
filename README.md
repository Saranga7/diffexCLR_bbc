To run training:

Change train_misc, train, wandb projectname
```
python train.py k=30 hparams.batch_size=128 model.alpha=1 > output_logs/train_k30_diffaeclr_alpha1_BS128.log 2>&1
```



To generate:

```
bash run_gen.sh
```

For AttFind:

```
bash run_AttFind.sh
```

For Animations:

```
bash run_animate.sh
```



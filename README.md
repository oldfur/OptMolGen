## Evaluate the GeoLDM

To analyze the sample quality of molecules:

```python sample_analyze.py --model_path outputs/$exp_name --n_samples 10_000```

To visualize some molecules:

```python sample_visualize.py --model_path outputs/$exp_name --n_samples 10_000```

## args for sample test in pretrained model 

--model_path: outputs/drugs_latent2.tar.gz
--n_samples: 100

## Evaluate the GeoLDM

To analyze the sample quality of molecules:

```python sample_analyze.py --model_path outputs/$exp_name --n_samples 10_000```

To visualize some molecules:

```python sample_visualize.py --model_path outputs/$exp_name --n_samples 10_000```

To eval the reconstruction quality of VAE in GeoLDM for our fine-tune dataset(BPN, 2-MBA)

```python verify_vae_recon.py```

## args for sample test in pretrained model 

--model_path: outputs/drugs_latent2.tar.gz
--n_samples: 100

## pretrained model download

[download model](https://drive.google.com/drive/folders/1EQ9koVx-GA98kaKBS8MZ_jJ8g4YhdKsL)

## To install the conda environment

Please confirm that you are in **~/OptMolGen**

### install conda omg

```conda env update -f conda_environment.yml```

### install torch

```pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall```

### install other pip pkgs

```pip install annotated-types==0.7.0 certifi==2026.2.25 cffi==2.0.0 click==8.3.1 cryptography==46.0.5 cuda-bindings==12.9.4 cuda-pathfinder==1.2.2 filelock==3.20.0 fsspec==2025.12.0 gitdb==4.0.12 gitpython==3.1.46 idna==3.11 imageio==2.37.2 jinja2==3.1.6 markupsafe==3.0.2 mpmath==1.3.0 msgpack==1.1.2 msoffcrypto-tool==6.0.0 networkx==3.6.1 nvidia-cublas-cu12==12.6.4.1 nvidia-cuda-cupti-cu12==12.6.80 nvidia-cuda-runtime-cu12==12.6.77 nvidia-cudnn-cu12==9.10.2.21 nvidia-cufft-cu12==11.3.0.4 nvidia-cufile-cu12==1.11.1.6 nvidia-curand-cu12==10.3.7.77 nvidia-cusolver-cu12==11.7.1.2 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.7.1 nvidia-nccl-cu12==2.27.5 nvidia-nvjitlink-cu12==12.6.85 nvidia-nvshmem-cu12==3.4.5 nvidia-nvtx-cu12==12.6.77 olefile==0.47 platformdirs==4.9.2 protobuf==6.33.5 pycparser==3.0 pydantic==2.12.5 pydantic-core==2.41.5 pyyaml==6.0.3 requests==2.32.5 scipy==1.17.1 sentry-sdk==2.53.0 smmap==5.0.2 sympy==1.14.0 tqdm==4.67.3 triton==3.6.0 typing-inspection==0.4.2 urllib3==2.6.3 wandb==0.25.0```


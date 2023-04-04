## Environment configuration

###### **The codes are based on python3.8+, CUDA version 11.0+. The specific configuration steps are as follows:**

1. Create conda environment
   
   ```shell
   conda create -n fadm python=3.8
   conda activate fadm
   ```

2. Install pytorch
   
   ```shell
   conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
   ```

3. Installation profile
   
   ```shell
   pip install -r requirements.txt
   ```

## Pre-trained checkpoint

Checkpoints can be found under the following link: [link](https://1drv.ms/f/s!As9j28x7cXDfav147bznxlt-Jmg).

+ Download the `data.zip`, and unzip to the path `modules/deca`.
+ Download the `FOMM.pth`, `Facevid.pth`, and `diffusion.pth` to the path `modules/FOMM/ckpt/`, `modules/Facevid/ckpt/`, and `modules/diffusion/ckpt/` respectively.

## Demo testing

To run a reenactment demo, download checkpoint and run the following command:

```shell
python demo.py
```

And run the following command for training:

```shell
sh run.sh
```

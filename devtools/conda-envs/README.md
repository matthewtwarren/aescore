1. Create an environment and install dependencies using `environment.yml` file with Mamba (Conda installation is very slow, so Mamba is recommended):

    ```
    mamba env create -f environment.yml
    ```

2. Install PyTorch with CUDA for your platform using Conda/pip with the install command listed at https://pytorch.org/get-started/locally. E.g. for Linux with CUDA 11.8, I used:

    ```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

3. Verify PyTorch is installed with CUDA support by running the following command in a Python session:

    ```
    import torch
    torch.cuda.is_available()
    ```

    This should return True if correctly installed.

4. Download the AEScore repository from https://github.com/matthewtwarren/aescore/, using the master branch.

5. Verify that AEScore is working with simple training/inference examples using the `train.sh` and `predict.sh` scripts in `aescore/examples`. Make sure to run the scripts from the AEScore root directory (i.e. `path/to/aescore/`).

    ```
    bash examples/train.sh
    bash examples/predict.sh
    ```


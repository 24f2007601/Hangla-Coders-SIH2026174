nvidia-smi to check gpu info.
1. use conda/uv to create a virtual environment.
 conda create --name mlops python=3.14
 conda activate mlops
(preferably use uv, i didnt have option so conda)
2. install cuda toolkit & pytorch (https://pytorch.org/get-started/locally/#windows-verification)
 pip3 install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu132

 check proper installation -> python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

3. download the dataset & yolo mid sized model for accuracy https://youtu.be/A1V8yYlGEkI
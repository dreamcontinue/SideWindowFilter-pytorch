# Side Window Filter on PyTorch
The PyTorch implementation for [Side Window Filter](https://arxiv.org/abs/1905.07177) by sharing convolution filter kernel parameters.

From this idea, We also designed a **[Side Window Convolution](SideWindowConv.py)** that can be used for back propagation. 
Unfortunately, Our experiments have found that this Side Window Convolution is not as good as the original convolution.


## Usage
All codes are in [SideWindowFilter.py](SideWindowFilter.py) file.

You can run `main.py` or modify by yourself.

## Demo
#### Original image
<div align="center">
 <img src="img/ori.jpg" width = "500" height = "500" alt="Original image" />
 </div>

#### Image after 20 iterations with 3x3 Gaussian filtering
<div align="center">
 <img src="img/process.jpg" width = "500" height = "500" alt="Image after 5 iterations with 5x5 Box filtering" />
 </div>

#### Image after 5 iterations with 5x5 Box filtering
<div align="center">
 <img src="img/process_5x5box.jpg" width = "500" height = "500" alt="Image after 5 iterations with 5x5 Box filtering" />
 </div>

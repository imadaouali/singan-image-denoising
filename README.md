# SinGAN for Image Denoising (Abdellah RAHMANI - Imad AOUALI)

[Project](https://tamarott.github.io/SinGAN.htm) | [Arxiv](https://arxiv.org/pdf/1905.01164.pdf) | [CVF](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.pdf) 
### Reference : Official pytorch implementation of the paper: "SinGAN: Learning a Generative Model from a Single Natural Image" : https://github.com/tamarott/SinGAN

###  SinGAN Denoising
To perform image denoising over a noisy image using SinGAN (Please see SinGAN.ipynb for examples):

```
python denoising.py --input_name <training_image_file_name> --ref_name <noisy_image_file_name> --paint_start_scale <scale to inject>

```
Different injection scale will produce different denoising effects. A good choice of injection scale would be arround 2. 

Note that this will automatically train a SinGAN model on the image with a reconstruction factor that is specific to Image Denoising task.

### Existing Deep Learning Approahes for Image Denoising
We provied To Deep_Learning_Image_Denoising.ipynb that contains a benchmark of existing deep learning approaches for Image Denoising  

### Additional Notebooks
For traditional Image Denoising approaches, we use IPOL Demos. SinGAN_Reproduce.ipynb contains some code to reproduce the results of the official Github Repo cited above.

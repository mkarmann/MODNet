mkdir out_ppm_100
PYTHONPATH=$(pwd) python demo/image_matting/colab/inference.py\
 --input-path /datasets/portrait_matting_datasets/PPM-100/image\
 --output-path ./out_ppm_100\
 --ckpt-path ./pretrained/modnet_photographic_portrait_matting.ckpt
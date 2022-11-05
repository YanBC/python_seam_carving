# python seam-carving
Seam Carving in python, and it comes with flask application! Original paper: [Seam Carving for Content-Aware Image Resizing](http://www.faculty.idc.ac.il/arik/SCWeb/imret/index.html)

## What does it do?
Seam carving is an image processing algorithm for content-aware image resizing. Simply put, it can resize images without distorting important objects.

Simple demonstration:
| image | original (1428x968) | resized (968x968)  |
| --- | --- | --- |
| simple resize |![](images/Broadway_tower_edit.jpg) | ![](images/Broadway_tower_resized.jpg) |
| seam carving | ![](images/Broadway_tower_edit.jpg) | ![](images/Broadway_tower_seamcarved.jpg) |

## How does it work?
Consider resizing the above image from 1428x968 to 1427x968 for the moment. We first calculate a floating-point score for each pixel in the image. It representing the importance of that pixel as the shown below. The brighter the pixel, the more important it is.

<div align="centre">
![](images/Broadway_tower_enery.jpg)
</div>

And then, all we have to do is to find a path or a **seam** from top to bottom which accumulates to the mininum score and remove every pixel along that seam.

The most important contribution of the seam carving paper is that it provides a way to evaluate the "importance" of each pixel in the image.

## Quick start
```bash
# install python packages
python3 -m pip install -r requirements.txt
```


## Run in docker (recommended)
For those of you who are familiar with docker, you can use the following docker images to run the web app.
- `yanbc/seam_carving_cuda:latest`: for cuda version
- `yanbc/seam_carving_cpu:latest`: for cpu version

```bash
# cuda version
docker run --init -d --rm \
    -v /etc/localtime:/etc/localtime:ro \
    -v /etc/timezone:/etc/timezone:ro \
    -p 10800:10800 \
    --gpus all \
    yanbc/seam_carving_cuda:latest
# cpu version
docker run --init -d --rm \
    -v /etc/localtime:/etc/localtime:ro \
    -v /etc/timezone:/etc/timezone:ro \
    -p 10800:10800 \
    yanbc/seam_carving_cpu:latest
```

Or you can use the following commands to build the docker images:

```bash
# make cuda image
bash make_image.sh build_cuda
# make cpu image
bash make_image.sh build_cpu
```


## Usage
```bash
python cml_ui.py -h
```

for the flask web app, simply run
```bash
python web_ui.py
```

and then go to `http://127.0.0.1:10800/` in your favorite browser

![this is how the webpage looks like](images/demo.png)


## Comparison of speed
I use Linux `time` program to measure the time consumed. Each version is tested with the following command:

```bash
time python cml_ui.py images/rem.jpeg -r 400 -c 600
```

My hardware setup:
- CPU: `AMD Ryzen 7 5800H (16-core)`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

Results:
### GPU (cuda)
```
real    0m1.763s
user    0m1.702s
sys     0m4.742s
```

### CPU fast (numba)
```
real    0m10.509s
user    0m14.076s
sys     0m9.695s
```

### CPU slow
```
real    0m42.481s
user    0m45.133s
sys     0m4.929s
```

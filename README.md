# python seam-carving
Seam Carving in python, and it comes with flask application! For further information on seam carving, please refer to the original paper [Seam Carving for Content-Aware Image Resizing](http://www.faculty.idc.ac.il/arik/SCWeb/imret/index.html)

## requirements
```bash
# for cpu version
python3 -m pip install -r requirements.txt
# additionally, for gpu version
python3 -m pip install pycuda
```

## usage
```bash
python cml_ui.py -h
```

for the flask web app, simply run
```bash
python web_ui.py
```

and then go to `http://127.0.0.1:10800/` in your favorite browser

![this is how the webpage looks like](images/demo.png)


## using docker

### cpu runtime image
```bash
# make image
bash make_image.sh build
# start container
docker run --init -d --rm \
    -v /etc/localtime:/etc/localtime:ro \
    -v /etc/timezone:/etc/timezone:ro \
    -p 10800:10800 \
    seam_carving:latest
```

### cuda runtime image
```bash
# make image
bash make_image.sh build_gpu
# start container
docker run --init -d --rm \
    -v /etc/localtime:/etc/localtime:ro \
    -v /etc/timezone:/etc/timezone:ro \
    -p 10800:10800 \
    --gpus all \
    seam_carving_cuda:latest
```

## Comparison of speed
I use Linux `time` program to measure the time consumed. Each version is tested with the following command:

```bash
time python cml_ui.py images/rem.jpeg -r 400 -c 600
```

My hardware setup:
- CPU: `AMD Ryzen 7 5800H (16-core)`
- GPU: `NVIDIA GeForce RTX 3080 Ti`

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

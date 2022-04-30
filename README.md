# python seam-carving
Seam Carving in python, and it comes with flask application! For more infos on seam carving, please refer to the original paper [Seam Carving for Content-Aware Image Resizing](http://www.faculty.idc.ac.il/arik/SCWeb/imret/index.html)

## requirements
- python3
- and all packages in `requirements.txt`

## usage
```bash
python cml_ui.py -h
```

for flask web app, simply run
```bash
python web_ui.py
```

and then go to `http://127.0.0.1:10800/` in your favorite browser

![](images/demo.png)


## using docker
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

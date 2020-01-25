from flask import Flask, request, render_template
from flask import redirect, url_for, flash
from flask import send_from_directory
import os
from werkzeug.utils import secure_filename
from imageio import imread, imwrite
import argparse

from cml_ui import Engine




UPLOAD_FOLDER = './uploads'
IMAGE_FOLDER = './results'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

carver = Engine()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        srcPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(srcPath)

        target_width = int(request.form['width'])
        target_height = int(request.form['height'])
        image = imread(srcPath)

        output = carver.run(image, target_width, target_height)

        if output is not None:
            desPath = os.path.join(app.config['IMAGE_FOLDER'], filename)
            imwrite(desPath, output)
            return redirect(url_for('show_file', filename=filename))
        else:
            return "Sorry, something went wrong"


    return render_template('upload.html')


@app.route('/images/<filename>')
def show_file(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--ip', default='', help='ip address. Default to wild card')
    p.add_argument('--port', default=10800, type=int, help='port number. Default to 10800')
    args = p.parse_args()
    return args

if __name__ == '__main__':
    opts = parser()

    if not os.path.isdir(app.config['UPLOAD_FOLDER']):
        os.mkdir(app.config['UPLOAD_FOLDER'])
    if not os.path.isdir(app.config['IMAGE_FOLDER']):
        os.mkdir(app.config['IMAGE_FOLDER'])

    # app.config['ENV'] = 'development'
    app.config['MAX_CONTENT_LENGTH'] = 124 * 1024 * 1024
    app.run(host=opts.ip, port=opts.port)


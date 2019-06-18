import os
#import magic
import urllib.request
#from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = '/Users/officialbiznas/Documents/GitHub/ImageSimilarity/src'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_image_sims(fn_image_to_compare):
    #1. vectorize image
    #2. Get Sims
    #3. Return urls



    return 

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Processing File')

        #process file
        #save output in an templates/___.html
        #render_template(___.html)




            return redirect('/')
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)

if __name__ == "__main__":
    #app.run()
    app.run(host='0.0.0.0', port=5001, debug=True)
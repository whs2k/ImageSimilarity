import os
import sys

# import magic
# import urllib.request
# from app import app
from flask import Flask, flash, request, redirect, render_template, url_for, jsonify
from werkzeug.utils import secure_filename

sys.path.append(os.getcwd())
import helper
from Config import UPLOAD_FOLDER, FN_DF_TRANSFORMED

app = Flask(__name__, static_folder="static")
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'gif'])

trained_densenet_model = helper.load_model()
trained_yolo_model = helper.load_yolo_model()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload2.html')


@app.route('/', methods=['POST'])
def upload_file():
    print("-------------------------------------------------")
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
            fn_to_save = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(fn_to_save)
            print(fn_to_save)
            print('file: ', file)

            # flash('Processing File')
            try:
                fn_html_export = 'export.html'
            except:
                fn_html_export = 'export.html'
                pass
            print(fn_html_export)
            df = helper.get_image_sims(fn_image_to_compare=fn_to_save,
                                       trained_model=trained_densenet_model,
                                       trained_yolo_model=trained_yolo_model,
                                       fn_df_save=FN_DF_TRANSFORMED).sort_values('cosim', ascending=False)

            helper.createResultsHTML(df_html=df[['fn', 'cosim']],
                                     upload_image=fn_to_save,
                                     result_one=df.fn.loc[0],
                                     # list_of_topX_links=df.fn.tolist()[0:3],
                                     # list_of_perfect_links=df[df['cosim'] > 0.8].fn.tolist()[0:3],
                                     fn_to_export_template=os.path.join(os.getcwd(), 'templates', fn_html_export))

            print('df.fn.loc[0]: ', df.fn.loc[0])
            print('df.fn.loc[1]: ', df.fn.loc[1])
            print('df.fn.loc[2]: ', df.fn.loc[2])
            return render_template(fn_html_export,
                                   img_org=url_for('static', filename=fn_to_save.split('/')[-1]),
                                   img_res1=url_for('static', filename=df.fn.loc[0].split('data/')[-1]),
                                   img_res2=url_for('static', filename=df.fn.loc[1].split('data/')[-1]),
                                   img_res3=url_for('static', filename=df.fn.loc[2].split('data/')[
                                       -1]), )  # , upload_image_url=url_for('static', filename=fn_to_save.split('/')[-1]), #result_one=url_for('static', filename=df.fn.loc[0].split('/')[-1]))

            # process file
            # save output in an templates/___.html
            # render_template(___.html)

            return redirect('/')
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)


@app.route('/detect', methods=['POST'])
def detect_files():
    # check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({
            "status": 400,
            "msg": 'No file part'
        })

    file = request.files['file']
    if file.filename == '':
        print("filename is blank")
        return jsonify({
            "status": 400,
            "msg": 'No file selected for uploading'
        })

    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)
        fn_to_save = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(fn_to_save)

        df = helper.get_image_sims(
            fn_image_to_compare=fn_to_save,
            trained_model=trained_densenet_model,
            trained_yolo_model=trained_yolo_model,
            fn_df_save=FN_DF_TRANSFORMED
        ).sort_values('cosim', ascending=False)
        fn_html_export = 'export.html'
        response = [
            "static/" + fn_to_save.split('/')[-1],
            "static/" + df.fn.loc[0].split('data/')[-1],
            "static/" + df.fn.loc[1].split('data/')[-1],
            "static/" + df.fn.loc[2].split('data/')[-1]
        ]
        return jsonify({
            "status": 200,
            "data": response
        })
    else:
        return jsonify({

            "status": 400,
            "msg": 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'
        })


if __name__ == "__main__":
    # app.run()
    app.run(host='0.0.0.0', port=5001, debug=True)

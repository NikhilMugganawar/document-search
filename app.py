from flask import Flask, request, render_template,flash,Markup,url_for,jsonify
from PyPDF2 import PdfFileReader, PdfFileWriter
from PyPDF2 import PdfFileMerger 
import string, random, hashlib, os, json,sqlite3 as lite
from werkzeug.utils import secure_filename
import glob
import fitz
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__)
# Get current path
path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')

# Make directory if uploads is not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['pdf'])
file_mb_max = 10
app.config['MAX_CONTENT_LENGTH'] = file_mb_max * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
@app.route("/home",methods=['POST'])
def home():
    #return "Hello, Flask!"
    return render_template('index.html')

@app.route("/processdocuments",methods=['POST'])
def processdocuments():
    if request.method == 'POST':
        mypath = request.form['textboxpath']
        mypath = mypath.replace('\\','//')
        # text_box_value = "**/*.pdf"
        # files = glob.glob('/home/geeks/Desktop/gfg/**/*.pdf', recursive = True)
        documents = []
        number_of_files = 1
        #for file in glob.glob(mypath + "/*.pdf"):
        for file in glob.iglob(mypath + "/*.pdf"):
            number_of_files = number_of_files + 1
            with fitz.open(file) as doc:
                text = "";
                #documents =[]
                res = []
                for page in doc:
                    text += page.getText().replace("\n", "").replace("\\","")
                    res.append(text)
                    documents.append(' '.join(res))
        vectorizer = TfidfVectorizer()
        # It fits the data and transform it as a vector
        X = vectorizer.fit_transform(documents)
        # Convert the X as transposed matrix
        X = X.T.toarray()
        # Create a DataFrame and set the vocabulary as the index
        df = pd.DataFrame(X, index=vectorizer.get_feature_names())
        query = request.form['querybox']
        q = [query]
        q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
        sim = {}
        # Calculate the similarity
        for i in range(number_of_files):
            sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
        sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
        search_results ={}
        for k, v in sim_sorted:
            if v > 0.0:
               search_results["Search Relevance "+str(v)]=documents[k]
        return render_template('results.html',search_results=search_results)
        #return jsonify(search_results)                    
        # for file in glob.iglob(mypath + "/*.pdf"):    
        #     if file.endswith('.pdf'):
        #         fileReader = PdfFileReader(open(file, "rb"))
        #         count = 0
        #         count = fileReader.numPages
        #         while count >= 0:
        #             count -= 1
        #             pageObj = fileReader.getPage(count)
        #             text = pageObj.extractText()
        #             res.append(text)
        # if 'files[]' not in request.files:
        #     flash('No files found, try again.')
        #     return render_template('index.html')
        # res = []
        # for f in request.files.getlist('files[]'):
        #     #data = request.files[f].read()
        #     #data = f.read()
        #     #filename = f.name
        #     #file = request.FILES[filename]
        #     #res.append(filename)
        #     pdfReader = PdfFileReader(request.files['files[]'])
        #     for page in range(pdfReader.numPages):
        #         page1 = pdfReader.getPage(page)
        #         res.append(page1.extractText())
            #res.append(data)
    #return "Done"
    # return jsonify({
    #     "api_stuff": res,
    # })     



if __name__ == 'main':
    app.debug = True
    app.run(host='localhost',port=5000)
from flask import Flask, request, render_template,flash,Markup,url_for,jsonify
from PyPDF2 import PdfFileReader, PdfFileWriter
from PyPDF2 import PdfFileMerger 
import string, random, hashlib, os, json,sqlite3 as lite
from werkzeug.utils import secure_filename
import glob
import fitz
import pandas as pd
import numpy as np
from flask_caching import Cache
#import textract
import os
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__)
app.secret_key = "abc"
app.config['UPLOAD_EXTENSIONS'] = ['.pdf']
#app.config['SESSION_TYPE'] = 'filesystem'
config = {
    "DEBUG":True,
    "CACHE_TYPE":"simple",
    "CACHE_DEFAULT_TIMEOUT":300,
}
app.config.from_mapping(config)
cache1 = Cache(app)
# Get current path
path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')

# Make directory if uploads is not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['pdf'])
file_mb_max = 3
app.config['MAX_CONTENT_LENGTH'] = file_mb_max * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(413)
def too_large(e):
    error = 'File Uploaded is too large'
    return render_template('index.html',error=error)

@app.route("/")
@app.route("/home",methods=['POST'])
def home():
    #return "Hello, Flask!"
    cache1.clear()
    return render_template('index.html')

@app.route("/processdocuments",methods=['POST'])
def processdocuments():
    cache1.clear()
    if request.method == 'POST':
        file_names=[]
        curr_path=os.getcwd() 
        files_in_dir=os.listdir()
        for file in file_names:
            if file.split('.')[-1] in ['pdf']:
                    os.remove(file)
            if file not in request.files:
                    error = 'No file attached in request'
                    return render_template('index.html',error=error)             
        uploaded_files=request.files.getlist("files[]") 
        for file in uploaded_files:
            if file.filename =="":
                error = 'File Name of one of the files selected is empty.'
                return render_template('index.html',error=error)
            if file.filename.split('.')[-1] not in ['pdf']:
                error = 'Only PDF files allowed'
                return render_template('index.html',error=error)      
            if file.filename.split('.')[-1] in ['pdf']:
                file.save(file.filename)
        files_in_dir=os.listdir()
        curr_path=os.getcwd()
        conventions=['pdf']
        documents = []
        number_of_files = 1
        for file in files_in_dir:
            ext=file.split('.')[-1]
            if ext in conventions:
                number_of_files = number_of_files + 1
                doc = fitz.open(file)
                #with fitz.open(file) as doc:
                text = ""
                res = []
                for page in doc:
                    text += page.getText().replace("\n", "").replace("\\","")
                    res.append(text)
                    documents.append(' '.join(res))
        # mypath = request.form['textboxpath']
        # mypath = mypath.replace('\\','//')
        # documents = []
        # number_of_files = 1
        # for file in glob.iglob(mypath + "/*.pdf"):
        #     number_of_files = number_of_files + 1
        #     with fitz.open(file) as doc:
        #         text = "";
        #         res = []
        #         for page in doc:
        #             text += page.getText().replace("\n", "").replace("\\","")
        #             res.append(text)
        #             documents.append(' '.join(res))
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
        if not search_results:
            error = 'No search results retrieved , please try rephrasing the query or try a search on different set of documents'
            return render_template('index.html',error=error)     
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
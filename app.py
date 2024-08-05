# app.py
import urllib.request

import os
import numpy as np
import cv2
import keras

# Keras
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

####
from flask import Flask, request, send_file
from flask_mail import Mail, Message
from moviepy.editor import VideoFileClip, AudioFileClip
####

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from pathlib import Path
import shutil


from PIL import Image
#DataBASE#
import sqlite3

from models import model1, model2

app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'Big Model.keras'

# Load your trained model
model = load_model(MODEL_PATH)

UPLOAD_FOLDER = 'static/uploads/'

#Name_uploaded_video =''

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['mp4'])

######################################models
Model_1=model1()
Model_2=model2()

######################################mail
# Create a Mail instance

app.config['MAIL_SERVER']= 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = 'moviefilteration@gmail.com'
app.config['MAIL_PASSWORD'] = 'wwyechfosczohrtc'
mail_forget = Mail(app)
#######################################

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/page')
def page():
    # Main page
    return render_template('index.html')


@app.route('/check', methods=['POST'])
def check():
    mail = request.form.get("mail")
    global user_id
    passw = request.form.get("pass")
    con = sqlite3.connect("graduate.db")
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT password FROM USERS WHERE gmail = ?", (mail,))
    row = cur.fetchone()
    con.close()
    
    if row:
        password = row['password']
        if password== passw:
            con = sqlite3.connect("graduate.db")
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute("SELECT userid FROM USERS WHERE gmail = ?", (mail,))
            row = cur.fetchone()
            con.close()
            user_id=row['userid']
            return render_template('index1.html')
        else:
            return render_template('index.html')
    else:
        print("No password found for the provided email.")
        return render_template('index.html')
        

@app.route('/reg', methods=['POST'])
def reg():
    try:
        username = request.form.get("reguser")
        mail = request.form.get("regmail")
        passw = request.form.get("regpass")
        with sqlite3.connect("graduate.db") as con:
            cur = con.cursor()
            cur.execute("INSERT INTO USERS (userName, gmail, password) VALUES (?, ?, ?)", (username, mail, passw))
            con.commit()
            msg = "Record successfully added"
            print(msg)
    except:
        con.rollback()
        msg = "Error in insert operation"
        print(msg)
    finally:
        con.close()
        return render_template('index.html')
        
@app.route('/send')
def send():
    return render_template('index3.html')

@app.route('/forget_mail', methods=['POST'])
def forget_mail():
    email = request.form.get("Smail")
    con = sqlite3.connect("graduate.db")
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT password FROM USERS WHERE gmail = ?", (email,))
    row = cur.fetchone()
    con.close()
    if row:
        password = row['password']
        msg = Message(subject='Your Password',sender='moviefilteration@gmail.com', recipients=[email])
        msg.body = f"Your password is: {password}"
        mail_forget.send(msg)
        return render_template('index.html')
    else:
        print("No password found for the provided email.")
        return render_template('index3.html', forget="ok")

@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            global uploaded_video
            uploaded_video=filename
            with sqlite3.connect("graduate.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO moveFiltered (movieName, userid) VALUES (?, ?)", (filename, user_id))
                con.commit()
                msg = "Record successfully added"
                print(msg)
    except:
        con.rollback()
        msg = "Error in insert operation"
        print(msg)
    finally:
        con.close()
        return render_template('index2.html', filename="")
    
    
@app.route('/model1', methods=['POST'])
def model1():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_video);
    out_path = os.path.join(app.config['UPLOAD_FOLDER'],'filtered_video.mp4')
    if os.path.exists(out_path):
        os.remove(out_path)
    global filtered_vid
    filtered_vid=Model_1.filter_violent_scenes_with_audio(file_path,out_path,model)
    return render_template('index2.html', filename=filtered_vid)

@app.route('/model2', methods=['POST'])
def model2():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_video)
    out_path = os.path.join(app.config['UPLOAD_FOLDER'],'filtered_video.mp4')
    if os.path.exists(out_path):
        os.remove(out_path)
    global filtered_vid
    filtered_vid=Model_2.finalVideo(file_path, out_path)
    print(filtered_vid)
    return render_template('index2.html', filename=filtered_vid)

@app.route('/model1_2', methods=['POST'])
def model1_2():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_video)
    out_path = os.path.join(app.config['UPLOAD_FOLDER'],'update_video.mp4')
    if os.path.exists(out_path):
        os.remove(out_path)
    global filtered_vid
    filtered_vid=Model_1.filter_violent_scenes_with_audio(file_path,out_path,model)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'update_video.mp4')
    out_path = os.path.join(app.config['UPLOAD_FOLDER'],'filtered_video.mp4')
    if os.path.exists(out_path):
        os.remove(out_path)
    
    filtered_vid=Model_2.finalVideo(file_path, out_path)
    print(filtered_vid)
    return render_template('index2.html', filename=filtered_vid)
    

@app.route('/download', methods=['POST'])
def download():
    home_directory = Path.home()
    downloads_directory = home_directory / "Downloads"
    downloads_directory.mkdir(parents=True, exist_ok=True)
    source_path = os.path.join(app.config['UPLOAD_FOLDER'],'filtered_video.mp4')
    destination_path = downloads_directory / 'filtered_video.mp4'
    print(f'File found in {source_path}')
    print(f'File successfully moved to {destination_path}')
    #Move the file to the Downloads folder
    shutil.move(str(source_path), str(destination_path))
    print(f'File successfully moved to {destination_path}')
    return render_template('index2.html', filename='filtered_video.mp4')

@app.route('/display/<filename>')
def display_video(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run()
  

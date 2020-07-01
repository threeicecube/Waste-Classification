'''
import numpy as np
from flask import Flask, request, render_template
import pickle

from fastai.vision import *
from fastai import *

import os

cwd = os.getcwd()
path = Path()

application = Flask(__name__)

model = load_learner(path, 'model/export.pkl')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #labels = ['grizzly','black','teddy']

    file = request.files['file']

    #Store the uploaded images in a temporary folder
    if file:
        filename = file.filename
        file.save(os.path.join("resources/tmp", filename))

    to_predict = "resources/tmp/"+filename
    img = open_image(to_predict)

    #Getting the prediction from the model
    prediction = model.predict(img)[0]

    #Render the result in the html template
    return render_template('index.html', prediction_text='Your Prediction :  {} '.format(prediction))
    
    '''
## 
import aiohttp
import asyncio
## 

import numpy as np
from flask import Flask, request, render_template
import pickle

from fastai.vision import *
from fastai import *

import os

## --
export_file_url = 'https://www.googleapis.com/drive/v3/files/13OoIJwqGZlQg6f5SVpKuNSLzORTkbGzs?alt=media&key=AIzaSyCX8hkh1IeoNTb0Mv8kbqcp2PGM8CMQcf4'
export_file_name = 'model/export.pkl'
## --

cwd = os.getcwd()
path = Path()

application = Flask(__name__)

## --
async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)

async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()
## --

model = load_learner(path, 'model/export.pkl')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #labels = ['grizzly','black','teddy']

    file = request.files['file']

    #Store the uploaded images in a temporary folder
    if file:
        filename = file.filename
        file.save(os.path.join("resources/tmp", filename))

    to_predict = "resources/tmp/"+filename
    img = open_image(to_predict)

    #Getting the prediction from the model
    prediction = model.predict(img)[0]

    #Render the result in the html template
    return render_template('index.html', prediction_text='Your Prediction :  {} '.format(prediction))

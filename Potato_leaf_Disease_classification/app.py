from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os, io, base64
from PIL import Image
import matplotlib.pyplot as plt
#from keras.models import load_model
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
#Python 3.9.19
UPLOAD_FOLDER='/static/uploads'
app=Flask(__name__)

app.config['SECRET_KEY']='secret_key'
app.config['UPLOADED_PHOTOS_DEST']=UPLOAD_FOLDER
#To serve static files
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


class webForm(FlaskForm):
    File=FileField('Browse and upload an image')
    submit=SubmitField('submit')
    
#Loding a model
model=load_model('D:\SUCHITRA\VSCode\FLASK_CODES\Potato_leaf_Disease_classification\potato_leaf_disease_detection_model.keras')
    
    
#Define classnames
class_names=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@app.route('/', methods=['GET','POST'])
def predict():
    form=webForm()
    image_url=None
    if form.validate_on_submit():
        image=form.File.data
        image_name=secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], image_name)
        
        #Save the image:
        image.save(image_path)
        
        #Read image using pillow
        img=Image.open(image_path)
        
         # Preprocess the image for the model
        img = img.resize((224, 224))  # Resize to the expected input size
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        
       
        # Make predictions
        batch_prediction = model.predict(img_array)
        # Pass a boolean to the template if needed
        prediction_exists = batch_prediction.any()  # This checks if any value in the prediction is true or non-zero
        Max_prob= np.argmax(batch_prediction[0])
        predicted_class=class_names[Max_prob]
        confidence=round(100*(np.max(batch_prediction[0])),2)

        
        
        # Convert the image to PNG format and save to a buffer
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")  # Saving as PNG even if the input was JPG or PNG
        buffer.seek(0)

        # Encode the image to a base64 string
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Optional: Remove the temporary file after processing
        #pip install --upgrade tensorflow keras
        os.remove(image_path)

        return render_template('index.html',form=form, img_data=img_str, batch_prediction=batch_prediction[0],prediction_exists=prediction_exists, Max_prob=Max_prob, predicted_class=predicted_class,confidence=confidence)
    return render_template('index.html',form=form)
        
if __name__=='__main__':
    app.run(debug=True)
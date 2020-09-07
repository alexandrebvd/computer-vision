# Digit Recognizer Webapp

This is an OpenCV based Flask web application. Once the user scribbles a digit and press upload button, the web application reads the scribbled digit as an image, applies a CNN based model, and finally displays the recognized digit with a confidence score as shown in the following image.

## Project Instructions

1. Clone the repository and navigate to the downloaded folder.


2. Create and activate a new environment.

```
conda create -n flask_app python=3.6
conda activate flask_app
pip install -r requirements.txt
```

3. Run cv_web_app.py
# Customvision-mask-classifier
The code is based on Python SDK for Custom Vision API and does the follwing:
* Makes a custom vision project
* Creates classification tags and uploads image dataset with the tags to the custom vision portal
* Performs training
* Makes prediction call to our trained model

## Changes
You need to update the following into the code from your Azure portal:
* Training Key
* API Endpoint
* Prediction EndPoint
* Prediction Resource id

## Usage
    pip install azure-cognitiveservices-vision-customvision
On successful installation,
    
    python CreateClassifier.py
    
## Result 
    Creating project... 
    Adding images... 
    Training... 
    Training status: Training 
    Training status: Completed Done!         
        With Mask: -%         
        Without Mask : -%
 
Follow my blog for more details:

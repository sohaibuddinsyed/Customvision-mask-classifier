from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials
import time
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
 
ENDPOINT = "https://testcvp.cognitiveservices.azure.com/"

# Replace with a valid key
training_key = "0c64291277c8425399d3873b8b2a35b0"
prediction_key = "21dcf849cf844333a0c1f45f6e5937aa"
prediction_resource_id = "/subscriptions/4199da6c-8f1e-4bfb-a8c1-62ba5bc76d73/resourceGroups/AzurePy/providers/Microsoft.CognitiveServices/accounts/TestCVP-Prediction"

publish_iteration_name = "classifyModel"

credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

# Create a new project
# print ("Creating project...")
# project = trainer.create_project("Y")

# withMask_tag = trainer.create_tag(project.id, "With Mask")
# withoutMask_tag = trainer.create_tag(project.id, "Without Mask")

base_image_url = "./images/"

# print("Adding images...")

# image_list = []

# for image_num in range(1, 6):
#     file_name = "withmask_{}.jpg".format(image_num)
#     with open(base_image_url + "withmask/" + file_name, "rb") as image_contents:
#         image_list.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), tag_ids=[withMask_tag.id]))

# for image_num in range(1, 6):
#     file_name = "withoutmask_{}.jpg".format(image_num)
#     with open(base_image_url + "withoutmask/" + file_name, "rb") as image_contents:
#         image_list.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), tag_ids=[withoutMask_tag.id]))

# upload_result = trainer.create_images_from_files(project.id, ImageFileCreateBatch(images=image_list))
# if not upload_result.is_batch_successful:
#     print("Image batch upload failed.")
#     for image in upload_result.images:
#         print("Image status: ", image.status)
#     exit(-1)

# print ("Training...")
# iteration = trainer.train_project(project.id)
# while (iteration.status != "Completed"):
#     iteration = trainer.get_iteration(project.id, iteration.id)
#     print ("Training status: " + iteration.status)
#     time.sleep(1)

# # The iteration is now trained. Publish it to the project endpoint
# trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
# print ("Done!")

# Now there is a trained endpoint that can be used to make a prediction
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

with open(base_image_url + "withmask/testImg.jpg", "rb") as image_contents:
    results = predictor.classify_image(
        "dc248edb-8a7d-4314-90a7-0494403301a3", publish_iteration_name, image_contents.read())

    # Display the results.
    for prediction in results.predictions:
        print("\t" + prediction.tag_name +
              ": {0:.2f}%".format(prediction.probability * 100))
    # print(project.id)
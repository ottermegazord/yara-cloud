
'''
***************************************************************************************
*
*                   Yara Cloud Segmentation
*
*
*  Name : Idaly Ali
*
*  Designation : Data Scientist
*
*  Description : Classification of stitched sky images using Bayesian inference
*
*
***************************************************************************************

'''
########################

""" Import Libraries """

import skyweatherCloud

########################

""" Parameters """

file1 = "images/img1.png"
file2 = "images/img2.png"
file3 = "images/img3.png"
file4 = "images/img4.png"


ls_of_image = [file1, file2, file3, file4]

model_file = "model/yaraCloudNet_v1.pb"
label_file = "model/yaraCloudNet_v1.txt"


""" Create Skyweather Cloud Object"""

clouds = skyweatherCloud.NormalizedCloud(ls_of_image, model_file, label_file)

""" Prediction """

print(clouds.pred())

""" Calculate cloud coverage"""

print(clouds.cloud_coverage())
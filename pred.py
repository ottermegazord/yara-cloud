
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
*  Description : Sample SkyWeather
*
*
***************************************************************************************

'''
########################

""" Import Libraries """

import skyweatherCloud

########################

""" Parameters """

file_name = "images/img5.png"
model_file = "model/retrained_graph.pb"
label_file = "model/retrained_labels.txt"


""" Create Skyweather Cloud Object"""

cloud = skyweatherCloud.Cloud(file_name, model_file, label_file)

""" Print cloud classification and % """

print(cloud.pred())
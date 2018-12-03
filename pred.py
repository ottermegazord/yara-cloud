
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
import argparse

########################

""" Parameters """

parser = argparse.ArgumentParser()
parser.add_argument('--image_file', type=str, help='specify weather image filename')
parser.add_argument('--model_file', type=str, help='specify weather model filename')
parser.add_argument('--label_file', type=str, help='specify weather label filename')
args = parser.parse_args()

file_name = args.image_file
model_file = args.model_file
label_file = args.label_file

if file_name == None:
	print('[ERROR] missing argument --image_file=filename')
	quit()

if model_file == None:
	print('[ERROR] missing argument --model_file=filename')
	quit()

if label_file  == None:
	print('[ERROR] missing argument --label_file=filename')
	quit()

print(file_name, model_file, label_file)
quit()

""" Create Skyweather Cloud Object """

cloud = skyweatherCloud.Cloud(file_name, model_file, label_file)

""" Print cloud classification and % """

print(cloud.pred())

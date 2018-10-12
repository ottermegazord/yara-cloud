# Yara Cloud Classification and Segmentation Tool

Tool to classify cloud into distinct categories, and segment them relative to background (sky)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

For Python 3

```
sudo dnf install python3-numpy
sudo dnf install python3-opencv
pip3 install tensorflow
pip3 install opencv-contrib-python
```

### Installing

Import skyweatherCloud library

```
import skyweatherCloud
```

Define paths to image, model (Tensorflow Frozen Graph) and label. Example...

```
file_name = "images/img3.png"
model_file = "model/yaraCloudNet_v1.pb"
label_file = "model/yaraCloudNet_v1.txt"
```

## Running the program

Create skyweatherCloud object

```
cloud = skyweatherCloud.Cloud(file_name, model_file, label_file)
```

Print cloud classification and segmentation index

```
print(cloud.pred())

```
Sample output

```
['c thick dark', '> 90%']

```

## Authors

* **Idaly Ali** - [GitHub](https://github.com/ottermegazord)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Tensorflow


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
*  Description : Cloud classifier for SkyWeather
*
*
***************************************************************************************

'''

########################

"""Import Libraries"""

import skyweather_prediction
import tensorflow as tf
import time
import numpy as np
from cloudsegmentation import Cloud as CloudSeg

########################


class Cloud():

    def __init__(self, file_name, model_file, label_file, input_height=224, input_width=224, input_mean=128,
                 input_std=128, input_layer="input", output_layer="final_result"):

        """

        :param file_name: PATH TO IMAGE
        :param model_file: PATH TO TENSORFLOW FROZEN MODEL
        :param label_file: PATH TO TENSORFLOW FROZEN LABEL
        :param input_height: INPUT IMAGE HEIGHT
        :param input_width: INPUT IMAGE WIDTH
        :param input_mean: INPUT IMAGE MEAN
        :param input_std: INPUT IMAGE STD
        :param input_layer: THE NAME OF INPUT OPERATION TO RETURN
        :param output_layer: THE NAME OF OUTPUT OPERATION TO RETURN
        """

        self.file_name = file_name
        self.model_file = model_file
        self.label_file = label_file
        self.input_height = input_height
        self.input_width = input_width
        self.input_mean = input_mean
        self.input_std = input_std
        self.input_layer = input_layer
        self.output_layer = output_layer

    def pred(self):

        """

        :return: ARRAY CONSISTING OF PREDICTED LABEL, CLOUD SEGMENTATION
        """

        """ Load Graph """

        graph = skyweather_prediction.load_graph(self.model_file)

        """ Read Tensor from Image"""

        t = skyweather_prediction.read_tensor_from_image_file(self.file_name,
                                        input_height=self.input_height,
                                        input_width=self.input_width,
                                        input_mean=self.input_mean,
                                        input_std= self.input_std)

        """ Set IO Operations """

        input_name = "import/" + self.input_layer
        output_name = "import/" + self.output_layer
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)

        """ Launch the graph in session"""

        with tf.Session(graph=graph) as sess:
            start = time.time()
            results = sess.run(output_operation.outputs[0],
                               {input_operation.outputs[0]: t})
            end = time.time()
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = skyweather_prediction.load_labels(self.label_file)

        score = []

        for i in top_k:
            score.append([labels[i], results[i]])

        """ Predict cloud/sky type and % """

        cloudSeg = CloudSeg(self.file_name)
        cloudSeg.segmentation()

        def switch_cloud(argument):
            switcher = {
                labels[0]: "< 10%",
                labels[1]: cloudSeg.percent(),
                labels[2]: "> 90%",
                labels[3]: cloudSeg.percent(),
                labels[4]: cloudSeg.percent(),
            }
            return switcher.get(argument, "Invalid label")

        percipitation = switch_cloud(score[0][0])

        return [score[0][0], percipitation]


    def pred_cat(self):

        """

        :return: ARRAY CONSISTING OF PREDICTED LABEL, CLOUD SEGMENTATION
        """

        """ Load Graph """

        graph = skyweather_prediction.load_graph(self.model_file)

        """ Read Tensor from Image"""

        t = skyweather_prediction.read_tensor_from_image_file(self.file_name,
                                        input_height=self.input_height,
                                        input_width=self.input_width,
                                        input_mean=self.input_mean,
                                        input_std= self.input_std)

        """ Set IO Operations """

        input_name = "import/" + self.input_layer
        output_name = "import/" + self.output_layer
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)

        """ Launch the graph in session"""

        with tf.Session(graph=graph) as sess:
            start = time.time()
            results = sess.run(output_operation.outputs[0],
                               {input_operation.outputs[0]: t})
            end = time.time()
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = skyweather_prediction.load_labels(self.label_file)

        score = []

        for i in top_k:
            score.append([labels[i], results[i]])

        """ Predict cloud/sky type and % """

        cloudSeg = CloudSeg(self.file_name)
        cloudSeg.segmentation()

        return score


class NormalizedCloud():

    def __init__(self, ls_of_image, model_file, label_file):
        """

        :param ls_of_image: PATH TO 4 IMAGES OF SKY
        :param model_file: PATH TO TENSORFLOW FROZEN MODEL
        :param label_file: PATH TO TENSORFLOW LABEL
        """

        self.ls_of_image = ls_of_image
        self.model_file = model_file
        self.label_file = label_file
        self.clouds = {}

        for i, path in enumerate(ls_of_image):
            self.clouds[i] = Cloud(path, self.model_file, self.label_file)

    def pred(self):
        """

        :return: list of predictions
        """
        categories = {}
        predictions = []
        stitchPredictions = {}

        for i, cloud in self.clouds.items():
            predictions.append(cloud.pred_cat())

        for i, cat in enumerate(self.clouds[0].pred_cat()):

            categories[i] = cat[0]


        for idx, cat in categories.items():

            prob = 0

            for x, prediction in enumerate(predictions):

                for each_category, each_prediction in prediction:

                    if cat == each_category:
                        prob += each_prediction * 0.25

            stitchPredictions[cat] = prob

        return(stitchPredictions)

    def cloud_coverage(self):
        """

        :return: Cloud cover index
        """

        percent_cloud = 0

        for i, cloud in self.clouds.items():

            each_cover = cloud.pred()[1]

            # print(each_cover)

            if each_cover == '< 10%':
                percent_cloud += 0.1 * 0.25

            elif each_cover == '> 90%':
                percent_cloud += 0.9 * 0.25

            else:
                percent_cloud += each_cover * 0.25

        return percent_cloud









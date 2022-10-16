import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as k_backend
from keras.preprocessing import image
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.compat.v1.logging import INFO, set_verbosity

random.seed(a = None, version = 2)

set_verbosity(INFO)


def get_mean_standard_deviation_per_batch(image_path, dataframe, Height = 320, Width = 320):
    sample_data = []
    for idx, img in enumerate(dataframe.sample(100)["Image"].values):
        # path = image_dir + img
        sample_data.append(np.array(image.load_img(image_path, target_size = (Height, Width))))

    mean = np.mean(sample_data[0])
    standard_deviation = np.std(sample_data[0])
    return mean, standard_deviation


def load_image(img, image_files_dir, dataframe, preprocess = True, Height = 320, Width = 320):
    # Load and preprocess image
    img_path = image_files_dir + img
    mean, standard_deviation = get_mean_standard_deviation_per_batch(img_path, dataframe, Height = Height, Width = Width)
    x = image.load_img(img_path, target_size = (Height, Width))
    if preprocess:
        x = x - mean
        x = x / standard_deviation
        x = np.expand_dims(x, axis = 0)
    return x


def grad_cam(input_model, image, cls, layer_name, Height = 320, Width = 320):
    # GradCAM method for visualizing input saliency
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = k_backend.gradients(y_c, conv_output)[0]

    gradient_function = k_backend.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (Width, Height), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam


def compute_grad_cam(model, img, image_files_dir, dataframe, labels, selected_labels, layer_name = "bn"):
    preprocessed_input = load_image(img, image_files_dir, dataframe)
    predictions = model.predict(preprocessed_input)

    print("Loading original image")
    plt.figure(figsize=(15, 10))
    plt.subplot(151)
    plt.title("Original\nImage")
    plt.axis("off")
    plt.imshow(load_image(img, image_files_dir, dataframe, preprocess = False), cmap = "gray")

    j = 1
    for i in range(len(labels)):
        if labels[i] in selected_labels:
            print(f"Generating Grad-CAM for class {labels[i]}")
            gradcam = grad_cam(model, preprocessed_input, i, layer_name)
            plt.subplot(151 + j)
            plt.title(f"{labels[i]}:\n Probability = {predictions[0][i]:.3f}")
            plt.axis("off")
            plt.imshow(load_image(img, image_files_dir, dataframe, preprocess = False), cmap = "gray")
            plt.imshow(gradcam, cmap = "jet", alpha = min(0.5, predictions[0][i]))
            j = j + 1


def get_roc_curve(labels, prediction_test_set, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gnrt = generator.labels[:, i]
            predicted = prediction_test_set[:, i]
            auc_roc = roc_auc_score(gnrt, predicted)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gnrt, predicted)
            plt.figure(1, figsize = (10, 10))
            plt.plot([0, 1], [0, 1], "k--")
            plt.plot(fpr_rf, tpr_rf, label = labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc = "best")
        except:
            print(f"Error in generating ROC curve for {labels[i]}. "
                  f"Dataset lacks enough examples.")
    plt.show()
    return auc_roc_vals

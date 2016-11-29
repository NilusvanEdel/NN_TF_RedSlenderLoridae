import pickle
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib import gridspec
import random
import collections
import tensorflow as tf

# import the data
fileTrainData = open("/home/nilus/PycharmProjects/untitled/trainData.pickle", 'rb')
data = pickle.load(fileTrainData)
fileLabelData = open("/home/nilus/PycharmProjects/untitled/trainLabels.pickle", 'rb')
dataLabels = pickle.load(fileLabelData)
for i in range(len(dataLabels)):
    if dataLabels[i]==10:
        dataLabels[i] = 0
# print how often the different numbers occure in the whole sample
print(collections.Counter(dataLabels))
# generate a figure with 20 subplots to show random sample images of the data
w, h = 28, 28
fig = plt.figure()
gs = gridspec.GridSpec(4, 5)

# generate and plot the random samples
cnt = 0
for i in range(4):
    for j in range (5):
        ax = fig.add_subplot(gs[i, j])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        imgData = np.zeros((h, w, 3), dtype=np.uint8)
        randomNumber = random.randint(0, len(data))
        imgData[:][:] = data[randomNumber][:][:]
        ax.set_title(dataLabels[randomNumber])
        img = Image.fromarray(imgData, 'RGB')
        plt.imshow(img)
        cnt += 1
plt.show()

# split the data into training, test and validation sets
border1 = len(data)*0.75
border2 = len(data)*0.9
border3 = len(data)
trainData = data[:int(border1)]
trainLabels = dataLabels[:int(border1)]
testData = data[int(border1):int(border2)]
testLabels = dataLabels[int(border1):int(border2)]
validationData = data[int(border2):int(border3)]
validationLabels = dataLabels[int(border2):int(border3)]


def getMiniBatch(size):
    miniBatch = data[:size]
    miniBatchLabels = dataLabels[:size]
    for i in range(size):
        randomNo = random.randint(0, len(trainData)-1)
        miniBatch[i] = trainData[randomNo]
        miniBatchLabels[i] = trainLabels[randomNo]
    returnArray = [miniBatch, miniBatchLabels]
    return returnArray


# only the code of assignment 4 so far
x = tf.placeholder(tf.float32, [None, 784])
desired = tf.placeholder(tf.int64, [None])
weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))
logits = tf.matmul(x, weights) + biases
crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, desired)
crossEntropy = tf.reduce_mean(crossEntropy)
learningRate = 1e-5
gdsOptimizer = tf.train.GradientDescentOptimizer(learningRate)
trainingStep = gdsOptimizer.minimize(crossEntropy)
accuracy = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), desired)
accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
trainingSteps = 700
miniBatchSize = 300
plotStepSize = 25
trainingAccuracy = np.ones(trainingSteps)
validationAccuracy = np.ones(trainingSteps)

accuracyFigure, accuracyAxis = plt.subplots(1,1)
weightFigure, weightAxes = plt.subplots(2,5)

with tf.Session() as session:
    session.run(tf.initialize_all_variables())

    for step in range(trainingSteps):
        images, labels = getMiniBatch(miniBatchSize)
        images = images.reshape([-1, 784])
        trainingAccuracy[step], _ = session.run([accuracy, trainingStep], feed_dict={x: images, desired: labels})

        if step % plotStepSize == 0 or step == trainingSteps - 1:
            images, labels = validationData, validationLabels
            images = images.reshape([-1, 784])

            _accuracy, _weights = session.run([accuracy, weights], feed_dict={x: images, desired: labels})
            if step != trainingSteps - 1:
                validationAccuracy[step:step + plotStepSize] = [_accuracy] * plotStepSize
            accuracyAxis.cla()
            accuracyAxis.plot(trainingAccuracy, color='b')
            accuracyAxis.plot(validationAccuracy, color='r')
            accuracyFigure.canvas.draw()

            for i in range(10):
                weight = _weights[:, i].reshape(28, 28)
                weightAxes[i // 5, i % 5].cla()
                weightAxes[i // 5, i % 5].matshow(weight, cmap=plt.get_cmap('bwr'))
            weightFigure.canvas.draw()

    images, labels = testData, testLabels
    images = images.reshape([-1, 784])
    _accuracy = session.run(accuracy, feed_dict={x: images, desired: labels})
    plt.show()
    print("Test accuracy: ", _accuracy)


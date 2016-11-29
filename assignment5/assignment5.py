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


# get miniBatches with equal size of samples for each digit
def getMiniBatch(size):
    miniBatch = data[:size]
    miniBatchLabels = dataLabels[:size]
    noPerDigit = int(size/10)
    noPerDigitArr = np.zeros(10)
    cnt = 0
    for i in range(size):
        randomNo = random.randint(0, len(trainData)-1)
        while (noPerDigitArr[trainLabels[randomNo]] >= noPerDigit):
            randomNo = random.randint(0, len(trainData) - 1)
        miniBatch[i] = trainData[randomNo]
        miniBatchLabels[i] = trainLabels[randomNo]
        noPerDigitArr[trainLabels[randomNo]] += 1
        cnt += 1
    # in case of size%10 != 0
    while (cnt < size):
        randomNo = random.randint(0, len(trainData) - 1)
        miniBatch[i] = trainData[randomNo]
        miniBatchLabels[i] = trainLabels[randomNo]
        cnt += 1
    returnArray = [miniBatch, miniBatchLabels]
    return returnArray


# basically only the code of assignment 4 so far (added saverclass)
x = tf.placeholder(tf.float32, shape = [None, 28, 28, 1])
desired = tf.placeholder(tf.int64, shape = [None])
# First convolutional layer
initval = tf.truncated_normal([5, 5, 1, 32], stddev = 0.1)
kernels1 = tf.Variable(initval)
bias1 = tf.Variable(tf.constant(0.1, shape = [32]))
conv1 = tf.nn.conv2d(x, kernels1, strides = [1, 1, 1, 1], padding = "SAME")
actv1 = tf.nn.tanh(conv1 + bias1)
pool1 = tf.nn.max_pool(actv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
# Second convolutional layer
kernels2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev = 0.1))
bias2 = tf.Variable(tf.constant(0.1, shape = [64]))
conv2 = tf.nn.conv2d(pool1, kernels2, strides = [1, 1, 1, 1], padding = "SAME")
actv2 = tf.nn.tanh(conv2 + bias2)
pool2 = tf.nn.max_pool(actv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
# Vectorize the output of the second convolutional layer
pool2Flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
# Fully connected feed forward layer
initval = tf.truncated_normal([7 * 7 * 64, 1024], stddev = 0.1)
weights1 = tf.Variable(initval)
ffnnbias1 = tf.Variable(tf.constant(0.1, shape = [1024]))
ffnn1act = tf.nn.tanh(tf.matmul(pool2Flat, weights1) + ffnnbias1)
# Fully connected readout layer
weights2 = tf.Variable(tf.truncated_normal([1024, 10], stddev = 0.1))
ffnnbias2 = tf.Variable(tf.constant(0.1, shape = [10]))
ffnn2act = tf.matmul(ffnn1act, weights2) + ffnnbias2
crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(ffnn2act, desired)
crossEntropy = tf.reduce_mean(crossEntropy)
train_step = tf.train.AdamOptimizer(0.00007).minimize(crossEntropy)
# Calculate the accuracy
accuracy = tf.equal(tf.argmax(tf.nn.softmax(ffnn2act), 1), desired)
accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
trainingSteps = 3
plotStepSize = 1
miniBatchSize = 300
trainingAccurcies = np.ones(trainingSteps)
validationAccuracies = np.ones(trainingSteps)
trainingCrossEntropies = np.zeros(trainingSteps)
validationCrossEntropies = np.zeros(trainingSteps)
accFig, accAx = plt.subplots(1,1)
ceFig, ceAx = plt.subplots(1,1)
actFig, actAx = plt.subplots(10,1)
accuracyFigure, accuracyAxis = plt.subplots(1,1)
weightFigure, weightAxes = plt.subplots(2,5)

with tf.Session() as session:
    # activate if run the first time, deactivate otherwise
    session.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    # activate to restore the old variables
    # saver.restore(session, "./simple-ffnn.ckpt")
    cnt = 0
    for step in range(trainingSteps):
        print (cnt)
        cnt += 1
        if step % 100 == 0 or step == (trainingSteps - 1):
            saver.save(session, "./simple-ffnn.ckpt")
        images, labels = getMiniBatch(miniBatchSize)
        images = images.reshape([-1, 28, 28, 1])
        trainingAccurcies[step], trainingCrossEntropies[step], _ = session.run([accuracy, crossEntropy, train_step],
                                                                               feed_dict={x: images, desired: labels})
        print("Oh my fucking god, it's a dinosaur")
        if step % plotStepSize == 0 or step == 0 or step == trainingSteps - 1:
            images, labels = validationData, validationLabels
            images = images.reshape([-1, 28, 28, 1])
            validationAccuracy, validationcrossEntropy, outputActivation = session.run(
                [accuracy, crossEntropy, ffnn2act], feed_dict={x: images, desired: labels})
            if step != trainingSteps - 1:
                validationAccuracies[step:step + plotStepSize] = [validationAccuracy] * plotStepSize
                validationCrossEntropies[step:step + plotStepSize] = [validationcrossEntropy] * plotStepSize
            print("do we even get here?")
            accAx.cla()
            accAx.plot(trainingAccurcies, color='b')
            accAx.plot(validationAccuracies, color='r')
            accFig.canvas.draw()

            ceAx.cla()
            ceAx.plot(trainingCrossEntropies, color='b')
            ceAx.plot(validationCrossEntropies, color='r')
            ceFig.canvas.draw()
            outputActivation = outputActivation.T
            for i, ax in enumerate(actAx):
                ax.cla()
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.matshow(np.matrix(outputActivation[i]), extent=[0, 6000, 0, 1], aspect='auto',
                           cmap=plt.get_cmap('Blues'))
            actFig.canvas.draw()
            print("just do it")
    images, labels = testData, testLabels
    images = images.reshape([-1, 28, 28, 1])
    testAccuracy = session.run(accuracy, feed_dict={x: images, desired: labels})
    print("Final: ", testAccuracy)
plt.show()

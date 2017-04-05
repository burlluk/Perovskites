import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import xlrd
import math

# num of batches
num_batches = 1
# num_points per feature
num_points = 1000
# LOV
fraction_to_leave_out = 0.00
# number_of_epochs (best > 500)
number_of_epochs = 1000
# num diff training and testing sets
num_iterations = 2
# learning_rate
learning_rate = 0.1
# output_col (0 indexed starting at second col)
output_col = 0
# DATA xlsx file
workbook = xlrd.open_workbook('data.xlsx')
# alias
sheet_names = workbook.sheet_names()
sheet = workbook.sheet_by_name(sheet_names[0])
# num cols to use (must be at least 2)
ncols = 300

# returns entire data set
def initialize_data():

    # data formated [[y],[x1],[x2],...,[xn]]
    data = []
    # first row in spreadsheet
    col_name = []
    # first col in spreadsheet
    row_name = []

    # global num_points

    # traverse every column
    for col_idx in range(ncols):
        #
        col = []
        # traverse every row in col including labels
        for row_idx in range(num_points+1):
            # save and store cell cell
            cell = sheet.cell(row_idx, col_idx)
            # store cell to names if first row
            if (row_idx == 0):
                col_name.append(cell.value)
            # else store cell as data point
            else:
                col.append(cell.value)
        # store col to names if first col
        if (col_idx) == 0:
            row_name.extend(col)
        # else store col as data point
        else:
            data.append(col)

    return data

# Separates data into training and testing
def make_training_data(data):
    # Leave out random 20% of points to function as test data for fit
    random_indices_chosen = []
    # counts num of testing chosen
    count = 0

    # initializes arrays for testing and training data
    # xtests and xtrains get 2
    xTests = [[] for x in range(ncols - 2)]
    xTrains = [[] for x in range(ncols - 2)]
    yTest = []
    yTrain = []

    # Make test data set from total data set
    while (count < fraction_to_leave_out*num_points):
        # choose random point
        random_index = np.random.randint(0, num_points)
        # begin picking out points without replacement
        if (random_index not in random_indices_chosen):
            # look for output col and traverse through all col
            output_found = False
            for col in range (0, len(data)):
                # add output to yTest
                if (col == output_col):
                    yTest.append(data[col][random_index])
                    output_found = True
                else:
                    # xTests must be contiguous when skipping an index
                    if (output_found == False):
                        xTests[col].append(data[col][random_index])
                    else:
                        xTests[col-1].append(data[col][random_index])
            # incr points selected
            count += 1
            random_indices_chosen.append(random_index)
        # choose new point if already selected
        else:
            random_index = np.random.randint(0, num_points)

    # Make training data set from those points not chosen to be in test data set
    # traverse through all points
    for index in range(num_points):
        # traverse through indexes not selected as testing
        if (index not in random_indices_chosen):
            # look for output col and traverse through all col
            skipped = True
            for col in range (0, len(data)):
                # add output to yTrain
                if col == output_col:
                    yTrain.append(data[output_col][index])
                    skipped = True
                else:
                    # xTests must be contiguous when skipping an index
                    if (skipped == False):
                        xTrains[col].append(data[col][index])
                    else:
                        xTrains[col-1].append(data[col][index])

    return xTrains, yTrain, xTests, yTest

# Converts lists to numpy arrays
def make_array_from_list(xTrains, yTrain, xTests, yTest):

    # coverts all xTrains to arrays
    xTrains_array = []
    for train in xTrains:
        xTrains_array.append(np.asarray(train).reshape(1,-1))

    # converts all xTests to arrays
    xTests_array =[]
    for test in xTests:
        xTests_array.append(np.asarray(test).reshape(1,-1))

    # converts yTrains and yTests to arrays
    yTrain_array = np.asarray(yTrain).reshape([1, -1])
    yTest_array = np.asarray(yTest).reshape([1, -1])

    return xTrains_array, yTrain_array, xTests_array, yTest_array

# TODO Main NN
# Performs poorly with training data and testing data
def launch_tensorflow(batched_xTrains, batched_yTrain, xTests, yTest, data, xTrains, yTrain):
    # Setup Tensorflow session

    # Layer 1
    # hold all placeholders in list
    placeholders = []
    for i in range(0, len(batched_xTrains)):
        placeholders.append(tf.placeholder(tf.float32, [1, None]))

    # make weight and bias
    W = tf.Variable(tf.truncated_normal(shape=[10, 1], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[10, 1]))

    # Layer 2 (Hidden layer)
    W2 = tf.Variable(tf.truncated_normal(shape=[1, 10], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[1]))

    # Activation (layer 1 -> layer 2)
    temp = tf.matmul(W,placeholders[0])
    for p in range(1, len(placeholders)):
        temp += tf.matmul(W, placeholders[p])
    hidden_layer = tf.nn.sigmoid(temp)

    # Output from layer 2 (hidden layer)
    y = tf.matmul(W2, hidden_layer) + b2

    # Minimize the squared errors.
    cost = tf.reduce_mean(tf.square(y - batched_yTrain))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.initialize_all_variables()
    # Launch TensorFlow graph
    sess = tf.Session()
    # with tf.Session() as sess:
    sess.run(init)

    # adds all placeholders to NN
    feed_dict = {}
    for index in range(0, len(batched_xTrains)):
        feed_dict[placeholders[index]] = batched_xTrains[index]

    # Do epochs and run NN
    for epoch in range(number_of_epochs + 1):
        sess.run(optimizer, feed_dict=feed_dict)
        # Display NN progress every 100th epoch
        if epoch % 100 == 0:
            print ("Epoch number", epoch, "Training set RMSE:", cost.eval(feed_dict, sess))

    # Calculate yFits for entire data, test data, and training data
    yFit_feed = {}
    test_yFit_feed = {}
    train_yFit_feed = {}

    for i in range (0, len(placeholders)):
        yFit_feed[placeholders[i]] = np.asarray(data[1:][i]).reshape([1, -1])
        test_yFit_feed[placeholders[i]] = np.asarray(xTests[i]).reshape([1, -1])
        train_yFit_feed[placeholders[i]] = np.asarray(xTrains[i]).reshape([1, -1])
    test_yFit =  y.eval(test_yFit_feed, sess)
    train_yFit = y.eval(train_yFit_feed, sess)
    yFit =      np.append(test_yFit, train_yFit)
    residual_test = test_yFit - yTest
    rmse_test = np.sqrt((np.sum(residual_test**2)) / len(test_yFit))

    print ("Testing set RMSE:", rmse_test)

    return yFit, rmse_test, test_yFit, train_yFit

def plot_data(label, xTrains_list, yTrain_list, xTests_list, yTest_list, yFit, data, test_yFit, train_yFit):

    yData = data[output_col]

    x = tf.placeholder(tf.float32, [1, None])
    yFit_list = yFit.transpose().tolist()
    test_yFit_list =  np.asarray(test_yFit.transpose().tolist()).flatten().tolist()
    train_yFit_list = np.asarray(train_yFit.transpose().tolist()).flatten().tolist()

    residual_test = test_yFit -yTest_list
    rmse_test = np.sqrt((np.sum(residual_test**2)) / len(test_yFit))


    fig1 = plt.figure()

    #ACTUAL VS PREDICTED
    # ax2 = fig1.add_subplot(221)

    # build a rectangle in axes coords
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    ax2 = fig1.add_subplot(111)
    plt.plot(train_yFit_list, yTrain_list, 'ro', label='training data')
    plt.plot(test_yFit_list,  yTest_list, 'bo', label='test data')
    ax2.set_xlabel('PredictedY')
    ax2.set_ylabel('ActualY')
    ax2.set_title(label + ' ActualY vs PredictedY')
    textstr1='Testing$r-squared=%.4f$'%(polyfit(yFit_list, yData[:len(yData)], 1)['determination'])
    textstr1 = textstr1, '$RMSE=%.4f$'%(rmse_test)
    ax2.text(right, top, textstr1,
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax2.transAxes)
    props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
    ax2.legend()
    plt.draw()
    plt.savefig(label + '_actual_vs_pred.png')

    # #HISTOGRAM
    fig2 = plt.figure()
    yFit_list = np.asarray(yFit_list).flatten().tolist()
    ax4 = fig2.add_subplot(111)
    n, bins, patches = plt.hist([yData, yFit_list], 10, normed=1, histtype='bar', color=['orange', 'red'],label=['Actual','Predicted'])
    textstr1='Predicted\n$Min=%.10f$\n$Max=%.10f$\n$Mean=%.10f$\n$Std Dev=%.10f$'%(min(yFit_list),max(yFit_list),np.mean(yFit_list),np.std(yFit_list))
    textstr2='Actual\n$Min=%.10f$\n$Max=%.10f$\n$Mean=%.10f$\n$Std Dev=%.10f$'%(min(yData),max(yData),np.mean(yData),np.std(yData))
    props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
    ax4.text(0.05,0.95,textstr1,transform=ax4.transAxes,fontsize=14,verticalalignment='top',bbox=props)
    ax4.text(0.05,0.60,textstr2,transform=ax4.transAxes,fontsize=14,verticalalignment='top',bbox=props)
    ax4.legend()
    ax4.set_title(label + ' Predicted Compared with Actual Histogram')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Frequency')
    plt.legend()
    plt.draw()
    plt.savefig(label + '_histogram.png')

    #Confusion Matrix
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for row in range (0, num_points - 1):
        if (yData[row] < 40 and yFit_list[row] < 40):
            true_positive += 1
        elif(yFit_list[row] < 40):
            false_positive += 1
        elif(yData[row] >=40 and yFit_list[row] >= 40):
            true_negative += 1
        elif(yFit_list[row] > 40):
            false_negative += 1

    conf_arr = [[true_positive,false_positive],
                [true_negative,false_negative]]
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            # tmp_arr.append(float(j)/float(a))
            tmp_arr.append(1.0)
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width, height = np.array(conf_arr).shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
    cb = fig.colorbar(res)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plt.xticks(range(width), ['Actually Stable', 'Actually Not Stable'])
    plt.yticks(range(height), ['Pred Stable','Pred Not Stable'])
    plt.savefig('confusion_matrix.png', format='png')

# For R^2 measurement
def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results

def main():

    # Get Parsed Data
    data = initialize_data()

    # Get x and y from parsed data
    xTrains, yTrain, xTests, yTest = make_training_data(data=data)
    xTrains_array, yTrain_array, xTests_array, yTest_array = make_array_from_list(xTrains=xTrains, yTrain=yTrain, xTests=xTests, yTest=yTest)

    # Used to find min and max rmse for each train and test data
    min_xTrains, min_yTrain, min_xTests, min_yTest = xTrains, yTrain, xTests, yTest
    min_xTrains_array, min_yTrain_array, min_xTests_array, min_yTest_array = xTrains_array, yTrain_array, xTests_array, yTest_array
    min_error = float("inf")
    min_yFit = None
    min_test_yFit = None
    min_train_yFit = None
    max_xTrains, max_yTrain, max_xTests, max_yTest = xTrains, yTrain, xTests, yTest
    max_xTrains_array, max_yTrain_array, max_xTests_array, max_yTest_array = xTrains_array, yTrain_array, xTests_array, yTest_array
    max_error = float("-inf")
    max_yFit = None
    max_test_yFit = None
    max_train_yFit = None

    rmses = []

    print()
    print()
    # Get batched training and testing sets
    for x in range(0,num_iterations):
        print("Iteration Number: ", x)
        batch_length = math.floor(len(yTrain_array[0])/num_batches)
        batch_start_index = 0
        batch_end_index = 0

        # Test NN for every batch and find best and worst fits
        for batch in range(num_batches):
            print("Batch Number: ", batch)
            batch_start_index = batch * batch_length
            if ((batch + 1) * batch_length > len(xTrains[0]) - 1):
                batch_end_index = len(xTrains[0]) - 1
            else:
                batch_end_index = (batch + 1) * batch_length
            batched_xTrains_array = []
            for i in range(len(xTrains)):
                batched_xTrains_array.append([xTrains[i][batch_start_index:batch_end_index]])
            batched_yTrain_array = yTrain_array[0][batch_start_index:batch_end_index]
            yFit, rmse_test, test_yFit, train_yFit = launch_tensorflow(batched_xTrains=batched_xTrains_array, batched_yTrain=batched_yTrain_array, xTests=xTests_array, yTest=yTest_array, data=data, xTrains=xTrains_array, yTrain=yTrain_array)
            rmses.append(rmse_test)
            if rmse_test < min_error:
            	min_error = rmse_test
            	min_yFit = yFit.copy()
            	min_test_yFit = test_yFit.copy()
            	min_train_yFit = train_yFit.copy()
            	min_xTrains, min_yTrain, min_xTests, min_yTest = xTrains.copy(), yTrain.copy(), xTests.copy(), yTest.copy()

            if rmse_test > max_error:
            	max_error = rmse_test
            	max_yFit = yFit.copy()
            	max_test_yFit = test_yFit.copy()
            	max_train_yFit = train_yFit.copy()
            	max_xTrains, max_yTrain, max_xTests, max_yTest = xTrains.copy(), yTrain.copy(), xTests.copy(), yTest.copy()


        # Get new traning and testing data
        xTrains, yTrain, xTests, yTest = make_training_data(data=data)
        xTrains_array, yTrain_array, xTests_array, yTest_array = make_array_from_list(xTrains=xTrains, yTrain=yTrain, xTests=xTests, yTest=yTest)
        print()
        print()

    # Print error data measurements
    for x in range(0,len(rmses)):
    	 'Iteration ', x, ': ', rmses[x]
    print ('Best RMSE: ', min_error)
    print ('Worst RMSE: ', max_error)
    print ("Mean RMSE: ", np.array(rmses).mean())
    print ("Std RMSE: ", np.array(rmses).std())
    print()

    # Plot worst and best fits
    plot_data(label='Best', xTrains_list=min_xTrains, yTrain_list=min_yTrain, xTests_list=min_xTests, yTest_list=min_yTest, yFit=min_yFit, data=data, test_yFit = min_test_yFit, train_yFit = min_train_yFit)
    plot_data(label='Worst', xTrains_list=max_xTrains, yTrain_list=max_yTrain, xTests_list=max_xTests, yTest_list=max_yTest, yFit=max_yFit, data=data, test_yFit = max_test_yFit, train_yFit = max_train_yFit)

main()

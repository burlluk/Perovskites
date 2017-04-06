import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import xlrd

num_points = 100
fraction_to_leave_out = 0.80
input_data_range = 2*np.pi

number_of_epochs = 1000
num_iterations = 2
learning_rate = 0.1
output_col = 2


temp_error = None


workbook = xlrd.open_workbook('TestSheet.xlsx')
sheet_names = workbook.sheet_names()
sheet = workbook.sheet_by_name(sheet_names[0])

def initialize_data():
    x_data = []
    y_data = []

    data = []
    col_name = []
    row_name = []

    for col_idx in range(sheet.ncols):
        temp = []
        for row_idx in range(sheet.nrows):
            cell = sheet.cell(row_idx, col_idx)
            # print cell.value
            if (row_idx == 0):
                col_name.append(cell.value)
            else:
                temp.append(cell.value)
        if col_idx == 0:
            row_name.extend(temp)
        else:
            data.append(temp)

    x_data.append(data[0])
    x_data.append(data[1])
    y_data = data[2]
    num_points = row_idx - 1
    #print x_data
    return data

def make_training_data(data):
    # Leave out random 20% of points to function as test data for fit
    #x_data, y_data = initialize_data()

    # xTest = []
    # x2Test = []
    # yTest = []
    # xTrain = []
    # x2Train = []
    # yTrain = []

    random_indices_chosen = []
    count = 0

    # x_data = data[0]
    # x2_data = data[1]
    # y_data = data[2]


    xTests = [[] for x in range(sheet.ncols - 2)]
    xTrains = [[] for x in range(sheet.ncols - 2)]
    yTest = []
    yTrain = []

    # Make test data set from total data set
    while count < fraction_to_leave_out*num_points:
        random_index = np.random.randint(0, num_points)
        if random_index not in random_indices_chosen:
            #print "The random index is", random_index
            for col in range (0, len(data)):
                if col == output_col:
                    yTest.append(data[col][random_index])
                else:
                    xTests[col].append(data[col][random_index])
            count += 1
            random_indices_chosen.append(random_index)
        else:
            random_index = np.random.randint(0, num_points)

    # Make training data set from those points not chosen to be in test data set
    for index in range(num_points):
        if index not in random_indices_chosen:
            for col in range (0, len(data)):
                if col == output_col:
                    yTrain.append(data[col][index])
                else:
                    xTrains[col].append(data[col][index])
                # x2Test.append(x2_data[random_index])
    # xTrain = xTrains[0]
    # xTest = xTests[0]

    # x2Train = xTrains[1]
    # x2Test = xTests[1]

    # # Make test data set from total data set
    # while count < fraction_to_leave_out*num_points:
    #     random_index = np.random.randint(0, num_points)
    #     if random_index not in random_indices_chosen:
    #         #print "The random index is", random_index
    #         xTest.append(x_data[random_index])
    #         x2Test.append(x2_data[random_index])
    #         yTest.append(y_data[random_index])
    #         count += 1
    #         random_indices_chosen.append(random_index)
    #     else:
    #         random_index = np.random.randint(0, num_points)

    # # Make training data set from those points not chosen to be in test data set
    # for index in range(num_points):
    #     if index not in random_indices_chosen:
    #         xTrain.append(x_data[index])
    #         x2Train.append(x2_data[index])
    #         yTrain.append(y_data[index])

    return xTrains, yTrain, xTests, yTest #xTrain, yTrain, xTest, yTest, x2Train, x2Test

def make_array_from_list(xTrains, yTrain, xTests, yTest):

    xTrain = xTrains[0]
    x2Train = xTrains[1]
    xTest = xTests[0]
    x2Test = xTests[1]

    xTrains_array = []
    xTests_array =[]
    for train in xTrains:
        xTrains_array.append(np.asarray(train).reshape(1,-1))

    for test in xTests:
        xTests_array.append(np.asarray(test).reshape(1,-1))


    xTrain_array = xTrains_array[0]
    x2Train_array = xTrains_array[1]
    yTrain_array = np.asarray(yTrain).reshape([1, -1])
    xTest_array = xTests_array[0]
    x2Test_array = xTests_array[1]
    yTest_array = np.asarray(yTest).reshape([1, -1])

    return xTrains_array, yTrain_array, xTests_array, yTest_array  #xTrain_array, yTrain_array, xTest_array, yTest_array, x2Train_array, x2Test_array

def launch_tensorflow(xTrains, yTrain, xTests, yTest, data):
    # Setup Tensorflow session

    xTrain = xTrains[0]
    x2Train = xTrains[1]
    xTest = xTests[0]
    x2Test = xTests[1]
    xData = data[0]
    x2Data = data[1]

    # Layer 1
    placeholders = []
    for i in range(0, len(xTrains)):
        placeholders.append(tf.placeholder(tf.float32, [1, None]))
    # x = tf.placeholder(tf.float32, [1, None])
    # x2 = tf.placeholder(tf.float32, [1, None])

    x = placeholders[0]
    x2 = placeholders[1]

    W = tf.Variable(tf.truncated_normal(shape=[10, 1], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[10, 1]))

    # Layer 2 (Hidden layer)
    W2 = tf.Variable(tf.truncated_normal(shape=[1, 10], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[1]))

    # Activation (layer 1 -> layer 2)
    temp = b
    # print len(placeholders)
    for p in range(0, len(placeholders)):
        temp += tf.matmul(W, placeholders[p])
    hidden_layer = tf.nn.sigmoid(temp)
    # hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(W, x), tf.matmul(W, x2)) + b)

    # Output from layer 2 (hidden layer)
    y = tf.matmul(W2, hidden_layer) + b2

    # Minimize the squared errors.
    cost = tf.reduce_mean(tf.square(y - yTrain))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.initialize_all_variables()
    # Launch TensorFlow graph
    sess = tf.Session()
    # with tf.Session() as sess:
    sess.run(init)

    for epoch in range(number_of_epochs + 1):

        feed_dict = {}
        # feed_dict[x] = xTrain
        # feed_dict[x2] = x2Train
        for index in range(0, len(xTrains)):
            feed_dict[placeholders[index]] = xTrains[index]
        sess.run(optimizer, feed_dict=feed_dict)
        # optimizer.run({x: xTrain}, sess)
        if epoch % 100 == 0:
            print ("Epoch number", epoch, "Training set RMSE:", cost.eval({x: xTrain, x2: x2Train}, sess))





    xData = np.asarray(xData).reshape([1, -1])
    x2Data = np.asarray(x2Data).reshape([1, -1])
    test_yFit =  y.eval({x: xTest, x2: x2Test}, sess)
    train_yFit = y.eval({x: xTrain, x2: x2Train}, sess)
    yFit =       y.eval({x: xData, x2: x2Data}, sess)

    residual_test = test_yFit - np.sin(xTest)
    rmse_test = np.sqrt((np.sum(residual_test**2)) / len(test_yFit))
    print ("Testing set RMSE:", rmse_test)

    return yFit, rmse_test, test_yFit, train_yFit

def plot_data(label, xTrain_list, yTrain_list, xTest_list, yTest_list, yFit, xData, yData, test_yFit, train_yFit, x2Train_list, x2Test_list, x2Data):

    x = tf.placeholder(tf.float32, [1, None])
    yFit_list = yFit.transpose().tolist()
    test_yFit_list =  np.asarray(test_yFit.transpose().tolist()).flatten().tolist()
    train_yFit_list = np.asarray(train_yFit.transpose().tolist()).flatten().tolist()

    residual_test = test_yFit - np.sin(np.asarray(xTest_list).reshape([1, -1]))
    rmse_test = np.sqrt((np.sum(residual_test**2)) / len(test_yFit))


    fig1 = plt.figure()
    # #RAW DATA
    # ax1 = fig1.add_subplot(221)
    # plt.plot(xData, yFit_list, 'b-', label='model fit')
    # plt.plot(xTrain_list, yTrain_list, 'ro', label='training data')
    # plt.plot(xTest_list, yTest_list, 'bo', label='test data')
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('y')
    # textstr1='Testing\n$RMSE=%.10f$'%(rmse_test)
    # props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
    # ax1.text(0.05,0.95,textstr1,transform=ax1.transAxes,fontsize=14,verticalalignment='top',bbox=props)
    # ax1.set_title(label + ' Training and Testing Data')
    # ax1.legend()
    # plt.draw()

    # #RESIDUALS
    # ax1 = fig1.add_subplot(221)
    # test_residuals = []
    # for x in range(0,len(test_yFit_list)):
    #     test_residuals.append(yTest_list[x]-test_yFit_list[x])
    # train_residuals = []
    # for x in range(0,len(train_yFit_list)):
    #     train_residuals.append(yTrain_list[x]-train_yFit_list[x])
    # plt.plot(xTrain_list, train_residuals, 'ro', label='training data')
    # plt.plot(xTest_list, test_residuals, 'bo', label='test data')
    # ax3.set_xlabel('x1')
    # ax3.set_ylabel('yTest - yPredicted')
    # ax3.set_title(label + ' Residuals')
    # ax3.legend()
    # plt.draw()

    #ACTUAL VS PREDICTED
    ax2 = fig1.add_subplot(221)
    plt.plot(train_yFit_list, yTrain_list, 'ro', label='training data')
    plt.plot(test_yFit_list,  yTest_list, 'bo', label='test data')
    ax2.set_xlabel('PredictedY')
    ax2.set_ylabel('ActualY')
    ax2.set_title(label + ' ActualY vs PredictedY')
    textstr1='Testing\n$RMSE=%.10f$'%(rmse_test)
    props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
    ax2.legend()
    plt.draw()

    # #RESIDUALS
    # ax3 = fig1.add_subplot(223)
    # test_residuals = []
    # for x in range(0,len(test_yFit_list)):
    # 	test_residuals.append(yTest_list[x]-test_yFit_list[x])
    # train_residuals = []
    # for x in range(0,len(train_yFit_list)):
    # 	train_residuals.append(yTrain_list[x]-train_yFit_list[x])
    # plt.plot(x2Train_list, train2_residuals, 'ro', label='training data')
    # plt.plot(x2Test_list, test2_residuals, 'bo', label='test data')
    # ax3.set_xlabel('x2')
    # ax3.set_ylabel('yTest - yPredicted')
    # ax3.set_title(label + ' Residuals')
    # ax3.legend()
    # plt.draw()

    #HISTOGRAM
    yFit_list = np.asarray(yFit_list).flatten().tolist()
    ax4 = fig1.add_subplot(224)
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


def main():

    data = initialize_data()
    x_data = data[0]
    x2_data = data[1]
    y_data = data[2]
    # xTrain, yTrain, xTest, yTest, x2Train, x2Test = make_training_data(data=data)
    #TODO
    xTrains, yTrain, xTests, yTest = make_training_data(data=data)
    xTrain = xTrains[0]
    x2Train = xTrains[1]
    xTest = xTests[0]
    x2Test = xTests[1]

    #xTrain_array, yTrain_array, xTest_array, yTest_array, x2Train_array, x2Test_array = make_array_from_list(xTrains=xTrains, yTrain=yTrain, xTests=xTests, yTest=yTest)
    xTrains_array, yTrain_array, xTests_array, yTest_array = make_array_from_list(xTrains=xTrains, yTrain=yTrain, xTests=xTests, yTest=yTest)

    xTrain_array = xTrains_array[0]
    x2Train_array = xTrains_array[1]
    xTest_array = xTests_array[0]
    x2Test_array = xTests_array[1]

    # min_xTrains, min_yTrain, min_xTests, min_yTest = xTrains, yTrain, xTests, yTests
    # min_xTrains_array, min_yTrain_array, min_xTests_array, min_yTest_array = xTrains_array, yTrain_array, xTests_array, yTests_array


    min_xTrain, min_yTrain, min_xTest, min_yTest, min_x2Train, min_x2Test = xTrain, yTrain, xTest, yTest, x2Train, x2Test
    min_xTrain_array, min_yTrain_array, min_xTest_array, min_yTest_array, min_x2Train_array, min_x2Test_array = xTrain_array, yTrain_array, xTest_array, yTest_array, x2Train_array, x2Test_array

    min_error = float("inf")
    min_yFit = None
    min_test_yFit = None
    min_train_yFit = None

    # max_xTrains, max_yTrain, max_xTests, max_yTest = xTrains, yTrain, xTests, yTests
    # max_xTrains_array, max_yTrain_array, max_xTests_array, max_yTest_array = xTrains_array, yTrain_array, xTests_array, yTests_array

    max_xTrain, max_yTrain, max_xTest, max_yTest, max_x2Train, max_x2Test = xTrain, yTrain, xTest, yTest, x2Train, x2Test
    max_xTrain_array, max_yTrain_array, max_xTest_array, max_yTest_array, max_x2Train_array, max_x2Test_array = xTrain_array, yTrain_array, xTest_array, yTest_array, x2Train_array, x2Test_array

    max_error = float("-inf")
    max_yFit = None
    max_test_yFit = None
    max_train_yFit = None

    rmses = []

    for x in range(0,num_iterations):
        yFit, rmse_test, test_yFit, train_yFit = launch_tensorflow(xTrains=xTrains_array, yTrain=yTrain_array, xTests=xTests_array, yTest=yTest_array, data=data)
        rmses.append(rmse_test)
        if rmse_test < min_error:
        	min_error = rmse_test
        	min_yFit = yFit.copy()
        	min_test_yFit = test_yFit.copy()
        	min_train_yFit = train_yFit.copy()
        	min_xTrain, min_yTrain, min_xTest, min_yTest, min_x2Train, min_x2Test = list(xTrain), list(yTrain), list(xTest), list(yTest), list(x2Train), list(x2Test)

        if rmse_test > max_error:
        	max_error = rmse_test
        	max_yFit = yFit.copy()
        	max_test_yFit = test_yFit.copy()
        	max_train_yFit = train_yFit.copy()
        	max_xTrain, max_yTrain, max_xTest, max_yTest, max_x2Train, max_x2Test = list(xTrain), list(yTrain), list(xTest), list(yTest), list(x2Train), list(x2Test)


        #TODO
        xTrains, yTrain, xTests, yTest = make_training_data(data=data)
        xTrain = xTrains[0]
        x2Train = xTrains[1]
        xTest = xTests[0]
        x2Test = xTests[1]

        # xTrain_array, yTrain_array, xTest_array, yTest_array, x2Train_array, x2Test_array = make_array_from_list(xTrains=xTrains, yTrain=yTrain, xTests=xTests, yTest=yTest)
        xTrains_array, yTrain_array, xTests_array, yTest_array = make_array_from_list(xTrains=xTrains, yTrain=yTrain, xTests=xTests, yTest=yTest)

        xTrain_array = xTrains_array[0]
        x2Train_array = xTrains_array[1]
        xTest_array = xTests_array[0]
        x2Test_array = xTests_array[1]

    for x in range(0,len(rmses)):
    	print ('Iteration ', x, ': ', rmses[x])
    print ('Best RMSE: ', min_error)
    print ('Worst RMSE: ', max_error)
    print ("Mean RMSE: ", np.array(rmses).mean())
    print ("Std RMSE: ", np.array(rmses).std())
    plot_data(label='Best', xTrain_list=min_xTrain, yTrain_list=min_yTrain, xTest_list=min_xTest, yTest_list=min_yTest, yFit=min_yFit, xData = x_data, yData = y_data, test_yFit = min_test_yFit, train_yFit = min_train_yFit, x2Train_list=min_x2Train, x2Test_list=min_x2Test, x2Data=x2_data)
    plot_data(label='Worst', xTrain_list=max_xTrain, yTrain_list=max_yTrain, xTest_list=max_xTest, yTest_list=max_yTest, yFit=max_yFit, xData = x_data, yData = y_data, test_yFit = max_test_yFit, train_yFit = max_train_yFit, x2Train_list=max_x2Train, x2Test_list=max_x2Test, x2Data=x2_data)

    plt.show()
main()

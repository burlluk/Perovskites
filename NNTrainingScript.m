inputs = (0:pi/64:2*pi);
targets = sin(inputs);
allRMSE = zeros(0, 5000);
allTestTargets = zeros(0, 10000);
allOutputs = zeros(0, 10000);
double RMSE;
numRuns = 10;
RMSESum = 0;
numBins = 30;
num = 0;
trainPct = .80;
valPct = .10;
testPct = 1-valPct-trainPct;

disp('Standard Deviation of all sin(x):');
disp(std(targets));
disp('Mean of all sin(x)');
disp(mean(targets));
disp('Rsquared for sin(x)');
mdl = fitlm(inputs, targets);
disp(mdl.Rsquared.Ordinary);
%Values Histogram
DispHistogram(targets, 50, 'Histogram of all sin(x) values', 'Value', 'Occurence');

for j=0:2
    minRMSE = 1;
    maxRMSE = 0;
    for i=0:numRuns-1
        %Calls function to set parameters as desired
        network = netParams(trainPct, testPct, valPct, 10);

        %Hides the NNtraintool window for "faster" training
        %net20.trainParam.showWindow = false;

        %Training the network with those set parameters
        [network, tr] = train(network, inputs, targets);

        outputs = network(inputs);
        testOutputs = outputs(tr.testInd);
        allOutputs = [allOutputs, testOutputs];
        testedTargets = targets(tr.testInd);
        %allTestTargets = [allTestTargets, testedTargets];
        errors = testOutputs-testedTargets;

        RMSE = sqrt(mean((errors).^2));

        if (RMSE<minRMSE)
            minRMSE=RMSE;
            minRMSEPred = testOutputs;
            minRMSETargets = testedTargets;
            minRMSETO = outputs(tr.trainInd);
            minRMSETT = targets(tr.trainInd);
        end

        if (RMSE>maxRMSE)
            maxRMSE=RMSE;
            maxRMSEPred = testOutputs;
            maxRMSETargets = testedTargets;
            maxRMSETO = outputs(tr.trainInd);
            maxRMSETT = targets(tr.trainInd);
        end

        allRMSE(i+1) = RMSE;
        RMSESum = RMSESum + RMSE;
    end

    %Display which run this is
    disp('--------------------------------------');
    fprintf('Data for training Percentage of %d%%:\n', trainPct*100);
    
    %Display Numerical Results
    %Mean RMSE
    disp('RMSE average: ');
    avgRMSE = RMSESum/numRuns;
    disp(avgRMSE);
    %RMSE Standard Deviation
    disp('RMSE Std. Dev:');
    RMSEdev = std(allRMSE);
    disp(RMSEdev);
    %Smallest RMSE
    disp('Smallest RMSE:');
    disp(minRMSE);
    %Smallest R^2
    disp('Smallest R^2:');
    mdl2 = fitlm(minRMSEPred, minRMSETargets);
    disp(mdl2.Rsquared.Ordinary);
    %Largest RMSE
    disp('Largest RMSE');
    disp(maxRMSE);
    %Largest R^2
    disp('Largest R^2:');
    mdl3 = fitlm(maxRMSEPred, maxRMSETargets);
    disp(mdl3.Rsquared.Ordinary);

    %Mean of Predicted Outputs
    avgOutput = mean(allOutputs);
    disp('Mean Y-value:');
    disp(avgOutput);
    %All Predicted Outputs Standard Deviation
    Dev = std(allOutputs);
    disp('Std Dev. of Outputs:');
    disp(Dev);

    %Graphs
    %RMSE Histogram
    DispHistogram(allRMSE, numBins, sprintf('RMSE Histogram [%.2g%%]', trainPct*100), 'RMSE', 'Occurence');
    %Data Scatterplot
    %DispScatter(allOutputs ,allTestTargets, sprintf('Predicted vs. Actual RMSE(%d)[%.2g%%]', avgRMSE, trainPct*100), 'Predicted sin(x)', 'sin(x)');
    %Max RMSE Scatterplot
    DispScatter(maxRMSEPred, maxRMSETargets, maxRMSETO, maxRMSETT, sprintf('Predicted vs. Actual with Largest RMSE(%d)[%.2g%%]', maxRMSE, trainPct*100), 'Predicted sin(x)', 'sin(x)');
    %Min RMSE Scatterplot
    DispScatter(minRMSEPred, minRMSETargets, minRMSETO, minRMSETT, sprintf('Predicted vs Actual with Smallest RMSE(%d)[%.2g%%]', minRMSE, trainPct*100), 'Predicted sin(x)', 'sin(x)');
    
    switch (num)
        case 0
            trainPct = .20;
            num = 1;
        case 1
            trainPct = .05;
            valPct = .05;
    end
    
    testPct = 1-valPct-trainPct;
end
%heyyyyyyyy

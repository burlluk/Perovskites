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

%Finds the current folder directory to be used in saving the documents
directory = pwd;
fprintf(fopen([directory, '\NumericalFigures.txt'], 'w'), '%s\n', 'Neural Network Data for Run on: ', date);
fclose('all');
%find the ID for the file specified and with an append permission
fileID = fopen(fopen([directory, '\NumericalFigures.txt'], 'a'));

%Writes to file all the data for sin(x) to a file
fprintf(fileID, '%s\n', 'Standard Deviation of all sin(x): ', std(targets));
%Writes the Mean of sin(x)
fprintf(fileID, '%s\n', 'Mean of all sin(x): ', mean(targets));
%Writes the R^2 value
mdl = fitlm(inputs, targets);
fprintf(fileID, '%s\n', 'Rsquared for sin(x): ', mdl.Rsquared.Ordinary);
%Writes the Std Dev of sin(x)
fprintf(fileID, '%s\n', 'Standard Deviation of all sin(x): ', std(targets));
fclose('all');
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

    %get a new File ID
    fileID = fopen([directory, '\NumericalFigures.txt'], 'a');
    
    %Display which run this is
    fprintf(fileID, '%s\n', '--------------------------------------');
    fprintf(fileID, '%s\n', sprintf('Data for training Percentage of %d%%:\n', trainPct*100));
    
    %Display Numerical Results
    %Mean RMSE
    avgRMSE = RMSESum/numRuns;
    fprintf(fileID, '%s\n', 'RMSE average: ', avgRMSE);
    %RMSE Standard Deviation
    RMSEdev = std(allRMSE);
    fprintf(fileID, '%s\n', 'RMSE Std. Dev: ', RMSEdev);
    %Smallest RMSE
    fprintf(fileID, '%s\n', 'Smallest RMSE: ', minRMSE);
    %Smallest R^2
    mdl2 = fitlm(minRMSEPred, minRMSETargets);
    fprintf(fileID, '%s\n', 'Smallest R^2: ', mdl2.Rsquared.Ordinary);
    %Largest RMSE
    fprintf(fileID, '%s\n', 'Largest RMSE: ', minRMSE);
    %Largest R^2
    mdl3 = fitlm(maxRMSEPred, maxRMSETargets);
    fprintf(fileID, '%s\n', 'Largest R^2: ', mdl3.Rsquared.Ordinary);

    %Mean of Predicted Outputs
    avgOutput = mean(allOutputs);
    fprintf(fileID, '%s\n', 'Mean Y-value: ', avgOutput);
    %All Predicted Outputs Standard Deviation
    Dev = std(allOutputs);

    fprintf(fileID, '%s\n', 'Std Dev. of Outputs: ', Dev);
    
    %Normalized data
    minNormal= (minRMSE-avgRMSE)/Dev;
    disp('Normalized Smallest RMSE');
    disp(minNormal);

    maxNormal= (maxRMSE-avgRMSE)/Dev;
    disp('Normalized Largest RMSE');
    disp(maxNormal);

    %Graphs
    %RMSE Histogram
    DispHistogram(allRMSE, numBins, sprintf('RMSE Histogram [%.2g%%]', testPct*100), 'RMSE', 'Occurence');
    %Data Scatterplot
    %DispScatter(allOutputs ,allTestTargets, sprintf('Predicted vs. Actual RMSE(%d)[%.2g%%]', avgRMSE, trainPct*100), 'Predicted sin(x)', 'sin(x)');
    %Max RMSE Scatterplot
    DispScatter(maxRMSEPred, maxRMSETargets, maxRMSETO, maxRMSETT, sprintf('Predicted vs. Actual with Largest RMSE(%d)[%.2g%%]', maxRMSE, testPct*100), 'Predicted sin(x)', 'sin(x)');
    %Min RMSE Scatterplot
    DispScatter(minRMSEPred, minRMSETargets, minRMSETO, minRMSETT, sprintf('Predicted vs Actual with Smallest RMSE(%d)[%.2g%%]', minRMSE, testPct*100), 'Predicted sin(x)', 'sin(x)');
    
    switch (num)
        case 0
            trainPct = .20;
            num = 1;
        case 1
            trainPct = .05;
            valPct = .05;
    end
    
    testPct = 1-valPct-trainPct;
    fclose('all');
end


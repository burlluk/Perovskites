% Set up Function
inputs = 0:pi/64:2*pi
targets = sin(inputs);
numRuns = 2;
numBins = 30;
allRMSE =zeros(0, 5000);
allTestTargets =zeros(0,10000);
allOutputs =zeros;
RMSESum=0;

[Train, Test] = crossvalind('LeaveMOut', numRuns, 1)

% First loop
trainPct= .80;
valPct= .1;
testPct= 1-valPct-trainPct;
num=0;

disp('Standard Deviation of all sin(x):');
disp(std(targets));
disp('Mean of all sin(x)');
disp(mean(targets));
disp('Rsquared for sin(x)');
mdl = fitlm(inputs, targets);
disp(mdl.Rsquared.Ordinary);

%Values Histogram
RMSEhisto(targets, 50, 'Histogram of all sin(x) values', 'Value', 'Occurence');

for i=0:3
minRMSE=1;
maxRMSE=0;
for j=0:numRuns-1
% Create a Fitting Network
% Set up Division of Data for Training, Validation, Testing
network = netParams(trainPct, testPct, valPct, 10);

 % Train the Network
[network,tr] = train(network,inputs,targets);
 % Test the Network
outputs = network(inputs);
testOutputs = outputs(tr.testInd);
allOutputs = [allOutputs, testOutputs];
testedTargets = targets(tr.testInd);
errors = gsubtract(outputs,targets);

saveas(gcf, ['My Beautifull Figure' ,num2str(i), '.pdf']);
 
error = targets-outputs;
number_of_bins = 30;

% Errors
 perror = (outputs - targets);   
% Squared Error
 SPerror =  perror.^2;   
% Mean Squared Error 
MSE = mean((sum(perror)).^2);   
% Root Mean Squared Error 
RMSE = sqrt(MSE);
    
%Saving RMSE Values
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
 
%Seperates values for differnet runs nicely 
disp('--------------------------------------------');
fprintf('Data for training Percentage of %d%%:\n', trainPct*100);

%RMSE values
disp('RMSE average: ');
    avgRMSE = RMSESum/numRuns;
    disp(avgRMSE);
disp('RMSE Std. Dev:');
    RMSEdev = std(allRMSE);
    disp(RMSEdev); 
disp('Smallest RMSE:');
    disp(min(allRMSE));
disp('Largest RMSE');
    disp(max(allRMSE));
    
%Rsquared Values 
disp('Largest R^2:');
    mdl2 = fitlm(minRMSEPred, minRMSETargets);
    disp(mdl2.Rsquared.Ordinary);
disp('Smallest R^2:');
    mdl3 = fitlm(maxRMSEPred, maxRMSETargets);
    disp(mdl3.Rsquared.Ordinary);

%Deviation Values
Dev = std(outputs);
avgOutput = mean(outputs);
disp('Mean Y-value:');
disp(avgOutput);
disp('Std Dev. of Outputs:');
disp(Dev);

%Normalized data
minNormal= (minRMSE-avgRMSE)/Dev;
disp('Normalized Smallest RMSE');
disp(minNormal);

maxNormal= (maxRMSE-avgRMSE)/Dev;
disp('Normalized Largest RMSE');
disp(maxNormal);

%cross validation plots

%Error Histogram 
ploterrhist(error,'bins',number_of_bins);

%Scatterplots
%Scatterplot(targets,outputs, sprintf('Predicted vs. Actual RMSE(%d)[%.2g%%]', avgRMSE, trainPct*100), 'sin(x)', 'Predicted sin(x)');
%Scatterplot(maxRMSETargets, maxRMSEPred, sprintf('Predicted vs. Actual with Largest RMSE(%d)[%.2g%%]', maxRMSEPred, trainPct*100), 'sin(x)', 'Predicted sin(x)');
%Scatterplot(minRMSETargets, minRMSEPred, sprintf('Predicted vs. Actual with Smallest RMSE(%d)[%.2g%%]', minRMSEPred, trainPct*100), 'sin(x)', 'Predicted sin(x)');

%Max RMSE Scatterplot
Scatterplot(maxRMSEPred, maxRMSETargets, maxRMSETO, maxRMSETT, sprintf('Predicted vs. Actual with Largest RMSE(%d)[%.2g%%]', maxRMSE, trainPct*100), 'Predicted sin(x)', 'sin(x)');
%Min RMSE Scatterplot
Scatterplot(minRMSEPred, minRMSETargets, minRMSETO, minRMSETT, sprintf('Predicted vs Actual with Smallest RMSE(%d)[%.2g%%]', minRMSE, trainPct*100), 'Predicted sin(x)', 'sin(x)');

%RMSE Histogram
RMSEhisto(allRMSE, numBins, sprintf('RMSE Histogram [%.2g%%]', trainPct*100), 'RMSE', 'Occurence');


%Switch between different leave out percentages 
switch(num)
    case 0
        trainPct = .20;
        num = 1;
    case 1
        trainPct = .05;
     
end

valPct = trainPct/2;
testPct = 1-valPct-trainPct;

end
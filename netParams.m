function [net] = netParams(trainPct, testPct, valPct, numInputs, nodes)
%UNTITLED2 Summary of this function goes here
%   if the user has not specified a certain number a nodes, sets to 10
    if (nargin<4)
        nodes = 10;
        numInputs = 1;
    end
    
    %Calling a new NN with 10 nodes
    net = fitnet(nodes);
    %setting number of inputs
    net.numInputs = numInputs;
    if (numInputs == 2)
        net.inputConnect = [1 1; 0 0];
    end
    %setting training data percentage to 20%
    net.divideParam.trainRatio = trainPct;
    %Validation percentage is 10%
    net.divideParam.valRatio = valPct;
    %Test percentage is also 70%
    net.divideParam.testRatio = testPct;
end


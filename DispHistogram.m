function [] = DispHistogram(values, numBins, gTitle, xAxisLabel, yAxisLabel)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    %Un-comment this next line if you wish for each graph to be displayed
    %in a separate window in addition to being saved to the folder
    %figure
    histogram(values, numBins);
    title(gTitle);
    xlabel(xAxisLabel);
    ylabel(yAxisLabel);
    directory = pwd;
    saveas(gcf, [fullfile([directory, '\Figures'], gTitle), '.pdf']);
end


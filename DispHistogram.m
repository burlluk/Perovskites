function [] = DispHistogram(values, numBins, gTitle, xAxisLabel, yAxisLabel)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    figure
    histogram(values, numBins);
    title(gTitle);
    xlabel(xAxisLabel);
    ylabel(yAxisLabel);
    directory = pwd;
    saveas(gcf, [fullfile([directory, '\Figures'], gTitle), '.pdf']);
end


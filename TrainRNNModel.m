%-------------------------------------------------------------------
%
%   TrainRNNModel.mat: train RNN model for image quality prediction 
%
%   Input:  XTrain: feature vector sequences
%           YTrain: target MOS values
%   Output: model: trained RNN model
% 
%   Jari Korhonen, Shenzhen University, 2020
%   tested with Matlab R2020a
%

function model = TrainRNNModel(XTrain, YTrain)

    % Define network layers
    numFeatures = size(XTrain{1},1);
    layers = [ sequenceInputLayer(numFeatures)
               fullyConnectedLayer(2048) 
               dropoutLayer(0.25)
               reluLayer  
               gruLayer(256,'OutputMode','sequence') 
               gruLayer(128,'OutputMode','last')
               gruLayer(64,'OutputMode','last')
               gruLayer(32,'OutputMode','last')
               fullyConnectedLayer(1)
               huberRegressionLayer('huber')];

    % Define learning parameters
    maxEpochs = 5;
    miniBatchSize = 32;
    options = trainingOptions('adam', ...
                              'MaxEpochs',maxEpochs, ...
                              'MiniBatchSize',miniBatchSize, ...
                              'LearnRateSchedule','piecewise', ...
                              'LearnRateDropPeriod',1, ...
                              'LearnRateDropFactor',0.5, ...
                              'InitialLearnRate',0.0002, ...
                              'L2Regularization',0.0001, ...
                              'ExecutionEnvironment','cpu', ...
                              'Shuffle','every-epoch', ...
                              'plots','training-progress', ...
                              'Verbose',0);

    % Train network
    model = trainNetwork(XTrain,YTrain',layers,options);
                
end

% EOF
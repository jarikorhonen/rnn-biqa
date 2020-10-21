%-----------------------------------------------------------------------
%
%   KoNIQ_Example.mat: Example how to train and test RNN model
%                      on KoNIQ-10k dataset
%
% 
%   Jari Korhonen, Shenzhen University, 2020
%   tested with Matlab R2020a
%

% Put the path to the KoNIQ dataset here. The images, as well as the
% metadata file, need to be in this directory.
path = './koniq/';

% Read the metadata
[data,datatxt] = xlsread([path 'koniq10k_scores_and_distributions.csv']);

% Get the list of filenames and MOS values
filepaths = {};
mos = [];
for i=1:length(datatxt(:,1))-1
    filepaths{i} = [path datatxt{i+1,1}];
    mos(i) = data(i,9);
end

% Load the pre-trained CNN feature extractor
load('./cnn_model_rnnbiqa.mat');

% Get feature sequences for each image
feature_seqs = MakeRNNFeatures(filepaths, netTransfer, 0, 0, 0);

% Save features for future use
save('KoNIQ_featuresequences.mat','-v7.3','feature_seqs','mos');

% Split feature sequences randomly to training and testing sets (80:20)
rng(10);
ftr_len = length(feature_seqs);
random_seq = randperm(ftr_len);
YTrain = mos(random_seq(1:ceil(0.8*ftr_len)))./100.0;
XTrain = ftr_seq(random_seq(1:ceil(0.8*ftr_len))) ;
YTest = mos(random_seq(ceil(0.8*ftr_len)+1:ftr_len))./100.0;
XTest = ftr_seq(random_seq(ceil(0.8*ftr_len)+1:ftr_len));

% Train RNN model
model = TrainRNNModel(XTrain, YTrain);

% Test RNN model
YPred = predict(model,XTest,'ExecutionEnvironment','cpu')';
fprintf('Test result: SRCC %0.3f PLCC %0.3f RMSE %2.2f\n', ...
           corr(YTest', YPred','type','Spearman'), ...
           corr(YTest', YPred','type','Pearson'), ...
           sqrt(mse(YTest*100.0, YPred*100.0)));

% EOF

% Add noise to TOA dataset
% The mismatches / inaccuracies are modeled by Gaussian noise
clc
clear all

pathDataset = strcat(pwd,'/dataset');

load([pathDataset,'/LocDBDelay.mat'],'matDelayLocReshaped');
TOAs = matDelayLocReshaped;

%----- Add Gaussian noise to TOA
noise_std = 20; % Gaussian noise with 10m std
DIMS = size(TOAs);
noise = noise_std.*randn(DIMS(1), DIMS(2), DIMS(3), DIMS(4));
a = ones(size(TOAs)); a(TOAs==0)=0; % TOA zero indicate buildings

TOAs_Noise = TOAs + a.*noise;
TOAs_Noise(TOAs_Noise<0)=0;

save('LocDBDelay_Noise20m.mat','TOAs_Noise','-v7.3')
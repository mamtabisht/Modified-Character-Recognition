clc;
clear all;
tic;
initime = cputime;
time1   = clock;

% MODEL: Stage-1 for consonant classification
% imds1 is the main dataset 

image_folder1 = fullfile('D:\Data\Z_Resized_consonants'); 
imds1 = imageDatastore(image_folder1, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
tbl = countEachLabel(imds1);
% Determine the smallest amount of images in a category
minSetCount = min(tbl{:,2});

[imds1Train,imds1Validation,imds1Test] = splitEachLabel(imds1,0.7,0.15,'randomize');


 % dataset split for 6-fold cross_validation  
%[imds1Train,imds1Validation,imds1Test] = splitEachLabel(imds1,0.7,0.15);
%[imds1Train,imds1Test,imds1Validation] = splitEachLabel(imds1,0.7,0.15);
%[imds1Validation,imds1Train,imds1Test] = splitEachLabel(imds1,0.15,0.7);
%[imds1Validation,imds1Test,imds1Train] = splitEachLabel(imds1,0.15,0.15);
%[imds1Test,imds1Train,imds1Validation] = splitEachLabel(imds1,0.15,0.7);
%[imds1Test,imds1Validation,imds1Train] = splitEachLabel(imds1,0.15,0.15);

%imageAugmenter = imageDataAugmenter( ...
  %  'RandXTranslation',[-3 3], ...
    %'RandYTranslation',[-3 3])
%imageSize = [300 300 1];
%augimdsTrain1 = augmentedImageDatastore(imageSize,imds1Train,'DataAugmentation',imageAugmenter);

%%CNN 7 layers and 37 FC layer Model.1_7
layers = [
imageInputLayer([300 300 1])

convolution2dLayer(3,8,'Padding','same')
batchNormalizationLayer
reluLayer

maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,16,'Padding','same')
batchNormalizationLayer
reluLayer

maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,32,'Padding','same')
batchNormalizationLayer
reluLayer

maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,64,'Padding','same')
batchNormalizationLayer
reluLayer

maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,128,'Padding','same')
batchNormalizationLayer
reluLayer

maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,256,'Padding','same')
batchNormalizationLayer
reluLayer

maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,256,'Padding','same')
batchNormalizationLayer
reluLayer

fullyConnectedLayer(37)
softmaxLayer
classificationLayer];

options = trainingOptions('sgdm', ...
'InitialLearnRate',0.01, ...
'MaxEpochs',12, ...
'Shuffle','every-epoch', ...
'ValidationData',imds1Validation, ...
'ValidationFrequency',20, ...
'Verbose',false, ...
'Plots','training-progress');

%net1_7 = trainNetwork(augimdsTrain1,layers,options);
net1_7 = trainNetwork(imds1Train,layers,options);

[YPred1_7,YProb1_7] = classify(net1_7,imds1Validation);
YValidation1 = imds1Validation.Labels;
accuracy1_7_Val = sum(YPred1_7 == YValidation1)/numel(YValidation1);
% classify Test images and calculate its probability
[YPred1_Test,Prob1] = classify(net1_7,imds1Test);
YTest1 = imds1Test.Labels;
accuracy_Test1_on_Consonants = sum(YPred1_Test== YTest1)/numel(YTest1);




%Model: Stage-2 for matra classification

image_folder2 = fullfile('D:\Data\Matra_Label'); 
imds2 = imageDatastore(image_folder2, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

% Find the first instance of an image for each category
img2= find(imds2.Labels == 'AA', 1);
figure
imshow(readimage(imds2,img2))
tbl_2 = countEachLabel(imds2);
% Determine the smallest amount of images in a category
minSetCount2 = min(tbl_2{:,2}); 
[imds2Train,imds2Validation,imds2Test] = splitEachLabel(imds2,0.7,0.15,'randomize');

%[imds2Train,imds2Test,imds2Validation] = splitEachLabel(imds2,0.7,0.15);
%[imds2Train,imds2Validation,imds2Test] = splitEachLabel(imds2,0.7,0.15);
%[imds2Validation,imds2Train,imds2Test] = splitEachLabel(imds2,0.15,0.7);
%[imds2Validation,imds2Test,imds2Train] = splitEachLabel(imds2,0.15,0.15);
%[imds2Test,imds2Train,imds2Validation] = splitEachLabel(imds2,0.15,0.7);
%[imds2Test,imds2Validation,imds2Train] = splitEachLabel(imds2,0.15,0.15);

%imageAugmenter = imageDataAugmenter( ...
  %  'RandXTranslation',[-3 3], ...
    %'RandYTranslation',[-3 3])
%imageSize = [300 300 1];
%augimdsTrain1 = augmentedImageDatastore(imageSize,imds1Train,'DataAugmentation',imageAugmenter);

%CNN 7 layers and 13 FC layer
layers2 = [
imageInputLayer([300 300 1])
convolution2dLayer(3,8,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,16,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,32,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,64,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,128,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,256,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,256,'Padding','same')
batchNormalizationLayer
reluLayer
fullyConnectedLayer(13)
softmaxLayer
classificationLayer];

options2 = trainingOptions('sgdm', ...
'InitialLearnRate',0.01, ...
'MaxEpochs',12, ...
'Shuffle','every-epoch', ...
'ValidationData',imds2Validation, ...
'ValidationFrequency',20, ...
'Verbose',false, ...
'Plots','training-progress');

net2_7 = trainNetwork(imds2Train,layers2,options2);
[YPred2,YProb2] = classify(net2_7,imds2Validation);
YValidation2 = imds2Validation.Labels;
accuracy2_7_Val = sum(YPred2 == YValidation2)/numel(YValidation2);
% classify Test images and calculate its probability
%[YPred2_Test,Prob2] = classify(net2_7,imds2Test);
%YTest2 = imds2Test.Labels;
%accuracy_Test2 = sum(YPred2_Test== YTest2)/numel(YTest2)

%check performance of net2_7 (Stage-2) on imds1Test
[YPred2_imds1Test,Prob2] = classify(net2_7,imds1Test);

% Convert predicted categorical label into cellarray
YPred1_Test = cellstr(YPred1_Test);
YPred2_imds1Test = cellstr(YPred2_imds1Test);


ImagesPath= cell(numel(imds1Test.Files),1);
for j=1:numel(imds1Test.Files)
    ImagesPath{j} = imds1Test.Files{j};
end    

xlswrite('D:\new folder\combineResults.xls',YPred1_Test, 'Sheet1', 'B'); % predicted char
xlswrite('D:\new folder\combineResults.xls',YPred2_imds1Test, 'Sheet1', 'C'); % predicted matra
xlswrite('D:\new folder\combineResults.xls',ImagesPath, 'Sheet1', 'A'); % image path

 
fintime = cputime;
elapsed = toc;
time2   = clock;
fprintf('TIC TOC: %g\n', elapsed);
fprintf('CPUTIME: %g\n', fintime - initime);
fprintf('CLOCK:   %g\n', etime(time2, time1));




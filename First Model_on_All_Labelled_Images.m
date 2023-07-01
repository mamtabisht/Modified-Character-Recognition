clc;
clear all;

image_folder1 = fullfile('D:\Data\All_Labelled_Images');
imds1 = imageDatastore(image_folder1, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

% Find the first instance of an image for each category
%img1= find(imds1.Labels == 'BaAA', 1);

%figure
%imshow(readimage(imds1,img1))

%tbl = countEachLabel(imds1);

% Determine the smallest amount of images in a category
%minSetCount = min(tbl{:,2});

% Dataset distribution for cross validation

%[imds1Train,imds1Test,imds1Validation] = splitEachLabel(imds1,0.7,0.15);
[imds1Train,imds1Validation,imds1Test] = splitEachLabel(imds1,0.7,0.15);

%[imds1Validation,imds1Train,imds1Test] = splitEachLabel(imds1,0.15,0.7);
%[imds1Validation,imds1Test,imds1Train] = splitEachLabel(imds1,0.15,0.15);

%[imds1Test,imds1Train,imds1Validation] = splitEachLabel(imds1,0.15,0.7);
%[imds1Test,imds1Validation,imds1Train] = splitEachLabel(imds1,0.15,0.15);

%imageAugmenter = imageDataAugmenter( ...
    %'RandRotation',[-20,20], ...
    %'RandXTranslation',[-3 3], ...
    %'RandYTranslation',[-3 3])
%imageSize = [300 300 1];
%augimdsTrain = augmentedImageDatastore(imageSize,imds1Train,'DataAugmentation',imageAugmenter);

for i=1:10
    tic;
  initime = cputime;
  time1   = clock;
    
  [imds1Train,imds1Validation,imds1Test] = splitEachLabel(imds1,0.7,0.15,'randomize');

    %CNN 7 layers and 435 FC layer

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


fullyConnectedLayer(435)
softmaxLayer
classificationLayer];

  options = trainingOptions('sgdm', ...
'InitialLearnRate',0.01, ...
'MaxEpochs',10, ...
'Shuffle','every-epoch', ...
'ValidationData',imds1Validation, ...
'ValidationFrequency',20, ...
'Verbose',false, ...
'Plots','training-progress');


  net1_7 = trainNetwork(imds1Train,layers,options);
 
  [YPred1_7,YProb1_7] = classify(net1_7,imds1Validation);
  YValidation1 = imds1Validation.Labels;

  accuracy1_7 = sum(YPred1_7 == YValidation1)/numel(YValidation1);

% classify Test images and calculate its probability
  [YPred1_Test,Prob1] = classify(net1_7,imds1Test);


  YTest1 = imds1Test.Labels;
  accuracy_Test1 = sum(YPred1_Test== YTest1)/numel(YTest1);

  fintime = cputime;
  elapsed = toc;
  time2   = clock;
  fprintf('TIC TOC: %g\n', elapsed);
  fprintf('CPUTIME: %g\n', fintime - initime);
  fprintf('CLOCK:   %g\n', etime(time2, time1));

%All_Labelled_CNN7 = net1_7;
%save All_Labelled_CNN7 ;


%code to write file path in excel sheet
%ImagesPath= cell(numel(imds1.Files),1);
%for j=1:numel(imds1.Files)
    %ImagesPath{j} = imds1.Files{j};
%end  
%xlswrite('D:\new folder\All_Labelled.xls',ImagesPath, 'Sheet1', 'A'); 




end

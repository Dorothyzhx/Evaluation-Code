clear all; close all; clc;

% a = './Result/DUTS-TE-mat/ours.mat'
% b = load(a)

%SaliencyMap Path setting
SalMapPath = './SalMap/'; %Put model results in this folder.
Models = {'PAGRN', 'PSPNet'};% You can add other model like: Models = {'PiCAnet', , 'Amulet', 'UCFzzq', 'WSS', 'AFNet', 'BASNet', 'C2SNet', 'CPD', 'MWS', 'RFCN', 'LPS', };
modelNum = length(Models);

%Datasets setting 
DataPath = './Dataset/';
Datasets = {'MSRA-B'};% You may also need other datasets, such as Datasets = {'DUTS-TE', 'DUT-OMRON', 'ECSSD', 'HKU-IS', , , 'SOD'};

%Results setting
ResDir = './Result/';

Thresholds = 1:-1/255:0;
datasetNum = length(Datasets);
for d = 1:datasetNum
    
    tic;
    dataset = Datasets{d};
    fprintf('Processing %d/%d: %s Dataset\n',d,datasetNum,dataset);
    
    ResPath = [ResDir dataset '-mat/']; %The result will be saved in *.mat file so that you can used it for the next time.
    if ~exist(ResPath,'dir')
        mkdir(ResPath);
    end
    resTxt = [ResDir dataset '_result.txt'];  %The evaluation result will be saved in ../Result folder.
    fileID = fopen(resTxt,'w');
    
    for m = 1:modelNum
        model = Models{m};

        gtPath = [DataPath dataset '/GT/'];
          
        salPath = [SalMapPath model '/' dataset '/'];
        
        imgFiles = dir([salPath '*.png']);
        imgNUM = length(imgFiles);
        
        [threshold_Fmeasure, threshold_Emeasure] = deal(zeros(imgNUM,length(Thresholds)));
        
        [threshold_Precion, threshold_Recall] = deal(zeros(imgNUM,length(Thresholds)));
        
        [Smeasure, adpFmeasure, adpEmeasure, MAE] =deal(zeros(1,imgNUM));
        
        for i = 1:imgNUM
            
            fprintf('Evaluating(%s Dataset,%s Model): %d/%d\n',dataset, model, i,imgNUM);
            name =  imgFiles(i).name;
            
            %load gt
            gt = imread([gtPath name]);
            
            if (ndims(gt)>2)
                gt = rgb2gray(gt);
            end
            
            if ~islogical(gt)
                gt = gt(:,:,1) > 128;
            end
            
            %load salency
            sal  = imread([salPath name]);
            
            %check size
            if size(sal, 1) ~= size(gt, 1) || size(sal, 2) ~= size(gt, 2)
                sal = imresize(sal,size(gt));
                imwrite(sal,[salPath name]);
                fprintf('Error occurs in the path: %s!!!\n', [salPath name]);
            end
            
            sal = im2double(sal(:,:,1));
            
            %normalize sal to [0, 1]
            sal = reshape(mapminmax(sal(:)',0,1),size(sal));
            Smeasure(i) = StructureMeasure(sal,logical(gt));
            
            % Using the 2 times of average of sal map as the threshold.
            threshold =  2* mean(sal(:)) ;
            [~,~,adpFmeasure(i)] = Fmeasure_calu(sal,double(gt),size(gt),threshold);
            
            
            Bi_sal = zeros(size(sal));
            Bi_sal(sal>threshold)=1;
            adpEmeasure(i) = Enhancedmeasure(Bi_sal,gt);
            
            [threshold_F, threshold_E]  = deal(zeros(1,length(Thresholds)));
            [threshold_Pr, threshold_Rec]  = deal(zeros(1,length(Thresholds)));
            for t = 1:length(Thresholds)
                threshold = Thresholds(t);
                [threshold_Pr(t), threshold_Rec(t), threshold_F(t)] = Fmeasure_calu(sal,double(gt),size(gt),threshold);
                
                Bi_sal = zeros(size(sal));
                Bi_sal(sal>threshold)=1;
                threshold_E(t) = Enhancedmeasure(Bi_sal,gt);
            end
            
            threshold_Fmeasure(i,:) = threshold_F;
            threshold_Emeasure(i,:) = threshold_E;
            threshold_Precion(i,:) = threshold_Pr;
            threshold_Recall(i,:) = threshold_Rec;
            
            MAE(i) = mean2(abs(double(logical(gt)) - sal));
            
        end
        
        
        column_F = mean(threshold_Fmeasure,1);
        meanFm = mean(column_F);
        maxFm = max(column_F);
        
        column_Pr = mean(threshold_Precion,1);
        column_Rec = mean(threshold_Recall,1);
        
        column_E = mean(threshold_Emeasure,1);
        meanEm = mean(column_E);
        maxEm = max(column_E);
        
        Smeasure = mean2(Smeasure);
        adpFm = mean2(adpFmeasure);
        adpEm = mean2(adpEmeasure);
        mae = mean2(MAE);
        
        save([ResPath model],'Smeasure', 'mae', 'column_Pr', 'column_Rec', 'column_F','adpFm', 'meanFm', 'maxFm', 'column_E', 'adpEm', 'meanEm', 'maxEm'); %   
        fprintf(fileID, '(Dataset:%s; Model:%s) Smeasure:%.4f; MAE:%.4f; adpEm:%.4f; meanEm:%.4f; maxEm:%.4f; adpFm:%.4f; meanFm:%.4f; maxFm:%.4f.\n',dataset,model,Smeasure, mae, adpEm, meanEm, maxEm, adpFm, meanFm, maxFm);       
    end
    toc;
    
end



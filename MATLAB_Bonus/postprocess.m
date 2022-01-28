%% Postprocessing the Predictions:
%% 1 Loading the Predictions
% In this section, we load the predictions of the networks (obtained from considering 
% two random seeds of rng(7) & rng(13)) for training and testing sets. We use 
% the boosted trees algorithm to combine the results of the two random folds (rng(7) 
% & rng(13)) and also adjust the predicted values based on the values of nearest 
% timestamps.
% 
% To further elaborate, for each frame of interest, ten additional timestamps 
% are considered which five of them are from the previous timestamps and five 
% are from the following timestamps. If these timestamps are available before 
% and after the timestamp of interest, their predicted distance values are placed 
% in these additional considered timestamps. Otherwise, the predicted value of 
% the available timestamp is repeated to fill these extra timestamps. The objective 
% in using this idea is that if previous and following timestamps are available 
% for a particular frame of interest, they might help improve the accuracy of 
% the prediction.
% 
% Later, the 22 timestamps from both networks (11 each) are combined in a single 
% set as an input for boosted trees algorithm with the target value of actual 
% prediction for that specific frame. The trained boosted trees model is then 
% applied to the testing set. 

train_labels=readtable('./Data/Tables/train_labels.csv');
train_pred_1=readtable('./Results/Train_pred_R7.csv');
train_pred_2=readtable('./Results/Train_pred_R13.csv');
test_pred_1=readtable('./Results/Test_pred_R7.csv');
test_pred_2=readtable('./Results/Test_pred_R13.csv');
%% 2. Processing the Prediction 
% As explained in section 1, If we have multiple timestamps from one video, 
% we can use the timestamps predictions to adjust the network prediction. Here 
% we used the five previous timestamps and five following timestamps (if available) 
% and provided them to the boosted trees algorithm as input. 

% Train data
X1=process_predictions(train_pred_1,3);
X2=process_predictions(train_pred_2,3);
X=horzcat(X1,X2);
y=train_labels.distance;
%%
% Test data
X_test1=process_predictions(test_pred_1,3);
X_test2=process_predictions(test_pred_2,3);
X_test=horzcat(X_test1,X_test2);
%% 3. Split Training and Validation
% In this section, we split the data to training and validation sets and explore 
% different options for training the boosted trees model. We found the optimal 
% parameters for training the boosted trees model, so we commented this section.

% rng(10) 
% vid_name=unique(train_labels(:,1));
% n_vid= numel(vid_name);
% n_training_id = randperm(n_vid);
% vid_name_tr = vid_name(n_training_id(1:round(0.8*n_vid)),1);
% idx=1:size(train_labels(:,1),1);
% trainingIdx=idx(ismember(train_labels{:,1},vid_name_tr{:,1}));
% validationIdx=idx(~ismember(train_labels{:,1},vid_name_tr{:,1}));
% X_train=X(trainingIdx,:);
% y_train=y(trainingIdx,:);
% X_val=X(validationIdx,:);
% y_val=y(validationIdx,:);
%%
% rmse= (immse(X_val(:,6),y_val))^0.5
% mae=mean(abs(X_val(:,6)-y_val))
%% 4 Training Boosted Trees
% 

 template= templateTree(...
 	'MinLeafSize', 10, ...
 	'NumVariablesToSample', 66);
 Mdl_RF= fitrensemble(X,...
     y, ...
 	'Method', 'LSBoost', ...
 	'NumLearningCycles', 400, ...
 	'Learners', template, ...
 	'LearnRate', 0.1);
%  predict_tree=predict(Mdl_RF,X_val);
%  mae=mean(abs(predict_tree-y_val))
%% 5 Predicting the Testing Set and Generating Final Submission File

train_labels=readtable('./Data/Tables/submission_format.csv');
predict_test=predict(Mdl_RF,X_test);
predict_test=round(predict_test*2)/2;
train_labels.distance=predict_test;
writetable(train_labels, "./Results/"+"submission.csv");
%% 6 Helper Function
% This section postproceses the predictions and prepares them for boosted trees 
% algorithm. 

function mat=process_predictions(train_pred_table,num)
    table0=train_pred_table;
    train_matrix=zeros(size(table0,1),11);
    for ii=1:size(table0,1)
        for jj=0:5
            if jj==0
                train_matrix(ii,6)=table0{ii,num};


            elseif (ii-jj)<=0 
                train_matrix(ii,6-jj)=train_matrix(ii,6-jj+1);

            elseif ~strcmp(table0.video_id(ii),table0.video_id(ii-jj))
                train_matrix(ii,6-jj)=train_matrix(ii,6-jj+1);

            else
                train_matrix(ii,6-jj)=table0{ii-jj,num};
            end
        end

         for kk=1:5

            if (ii+kk)> size(table0,1) 
                train_matrix(ii,6+kk)=train_matrix(ii,6+kk-1);

            elseif ~strcmp(table0.video_id(ii),table0.video_id(ii+kk))
                train_matrix(ii,6+kk)=train_matrix(ii,6+kk-1);
            else

                train_matrix(ii,6+kk)=table0{ii+kk,num};
            end
         end
    end
    mat=train_matrix;

end
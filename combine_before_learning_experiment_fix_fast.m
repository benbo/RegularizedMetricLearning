load('image_segmentation_data2.mat')
load('leaveOneOutModelsTrained.mat')
load('targetTestSetsCreated.mat')

[train_data,train_labels]=extract_training_data_from_cell( segmentation_train_selected , 1);
[test_data , test_labels]=extract_training_data_from_cell( segmentation_test_original , 1 );

if(0) %Everything in here only has to happen once.  It's loaded in 'leaveOneOutMOdelsTrained.mat'
    
    %% Get samples from individual classes to build training data
    mask0=train_labels==0;
    mask1=train_labels==1;
    mask2=train_labels==2;
    mask3=train_labels==3;
    mask4=train_labels==4;
    mask5=train_labels==5;
    mask6=train_labels==6;
    
    %Pull features associated with mask
    pos0_x = train_data(mask0,:);
    pos1_x = train_data(mask1,:);
    pos2_x = train_data(mask2,:);
    pos3_x = train_data(mask3,:);
    pos4_x = train_data(mask4,:);
    pos5_x = train_data(mask5,:);
    pos6_x = train_data(mask6,:);
    
    %Get labels associated with mask (should be uniform for each pos?_y)
    pos0_y = train_labels(mask0,:);
    pos1_y = train_labels(mask1,:);
    pos2_y = train_labels(mask2,:);
    pos3_y = train_labels(mask3,:);
    pos4_y = train_labels(mask4,:);
    pos5_y = train_labels(mask5,:);
    pos6_y = train_labels(mask6,:);
    
    
    %% Build 7 multi-class datasets in a leave one class out style
    trainFeatures_leave0out = [pos1_x;pos2_x;pos3_x;pos4_x;pos5_x;pos6_x]; % i.e. doesn't have class 0
    trainFeatures_leave1out = [pos0_x;pos2_x;pos3_x;pos4_x;pos5_x;pos6_x];
    trainFeatures_leave2out = [pos0_x;pos1_x;pos3_x;pos4_x;pos5_x;pos6_x];
    trainFeatures_leave3out = [pos0_x;pos1_x;pos2_x;pos4_x;pos5_x;pos6_x];
    trainFeatures_leave4out = [pos0_x;pos1_x;pos2_x;pos3_x;pos5_x;pos6_x];
    trainFeatures_leave5out = [pos0_x;pos1_x;pos2_x;pos3_x;pos4_x;pos6_x];
    trainFeatures_leave6out = [pos0_x;pos1_x;pos2_x;pos3_x;pos4_x;pos5_x];
    
    trainLabels_leave0out = [pos1_y;pos2_y;pos3_y;pos4_y;pos5_y;pos6_y];
    trainLabels_leave1out = [pos0_y;pos2_y;pos3_y;pos4_y;pos5_y;pos6_y]; % i.e. doesn't have class 1
    trainLabels_leave2out = [pos0_y;pos1_y;pos3_y;pos4_y;pos5_y;pos6_y];
    trainLabels_leave3out = [pos0_y;pos1_y;pos2_y;pos4_y;pos5_y;pos6_y];
    trainLabels_leave4out = [pos0_y;pos1_y;pos2_y;pos3_y;pos5_y;pos6_y];
    trainLabels_leave5out = [pos0_y;pos1_y;pos2_y;pos3_y;pos4_y;pos6_y];
    trainLabels_leave6out = [pos0_y;pos1_y;pos2_y;pos3_y;pos4_y;pos5_y];
    
    
    %% Get pairwise data labels [-1,1] Used for training.  Features are abs(xi-xj)
    [ leave0out_pfeats , leave0out_plab ] = similarize_data( trainFeatures_leave0out , trainLabels_leave0out );
    [ leave1out_pfeats , leave1out_plab ] = similarize_data( trainFeatures_leave1out , trainLabels_leave1out );
    [ leave2out_pfeats , leave2out_plab ] = similarize_data( trainFeatures_leave2out , trainLabels_leave2out );
    [ leave3out_pfeats , leave3out_plab ] = similarize_data( trainFeatures_leave3out , trainLabels_leave3out );
    [ leave4out_pfeats , leave4out_plab ] = similarize_data( trainFeatures_leave4out , trainLabels_leave4out );
    [ leave5out_pfeats , leave5out_plab ] = similarize_data( trainFeatures_leave5out , trainLabels_leave5out );
    [ leave6out_pfeats , leave6out_plab ] = similarize_data( trainFeatures_leave6out , trainLabels_leave6out );
    
    % Balance the -1 and +1 classes for each class
    leave0out_pfeats_negative = leave0out_pfeats(leave0out_plab==-1,:); %get all -1 samples
    leave0out_pfeats_negative_mask = randsample(sum(leave0out_plab==-1),sum(leave0out_plab==1));% subsample -1 samples to balance classes
    leave0out_pfeats_positive_mask = leave0out_plab==1; %get all +1 samples
    leave0out_pfeats = [ leave0out_pfeats(leave0out_pfeats_positive_mask,:) ; leave0out_pfeats_negative(leave0out_pfeats_negative_mask,:) ];
    leave0out_plab = [ leave0out_plab(leave0out_pfeats_positive_mask,:) ; -1*ones(size(leave0out_pfeats_negative_mask,1),1) ];

     % Balance the -1 and +1 classes for each class
    leave1out_pfeats_negative = leave1out_pfeats(leave1out_plab==-1,:); %get all -1 samples
    leave1out_pfeats_negative_mask = randsample(sum(leave1out_plab==-1),sum(leave1out_plab==1));% subsample -1 samples to balance classes
    leave1out_pfeats_positive_mask = leave1out_plab==1; %get all +1 samples
    leave1out_pfeats = [ leave1out_pfeats(leave1out_pfeats_positive_mask,:) ; leave1out_pfeats_negative(leave1out_pfeats_negative_mask,:) ];
    leave1out_plab = [ leave1out_plab(leave1out_pfeats_positive_mask,:) ; -1*ones(size(leave1out_pfeats_negative_mask,1),1) ];
    
    % Balance the -1 and +1 classes for each class
    leave2out_pfeats_negative = leave2out_pfeats(leave2out_plab==-1,:); %get all -1 samples
    leave2out_pfeats_negative_mask = randsample(sum(leave2out_plab==-1),sum(leave2out_plab==1));% subsample -1 samples to balance classes
    leave2out_pfeats_positive_mask = leave2out_plab==1; %get all +1 samples
    leave2out_pfeats = [ leave2out_pfeats(leave2out_pfeats_positive_mask,:) ; leave2out_pfeats_negative(leave2out_pfeats_negative_mask,:) ];
    leave2out_plab = [ leave2out_plab(leave2out_pfeats_positive_mask,:) ; -1*ones(size(leave2out_pfeats_negative_mask,1),1) ];
    
    % Balance the -1 and +1 classes for each class
    leave3out_pfeats_negative = leave3out_pfeats(leave3out_plab==-1,:); %get all -1 samples
    leave3out_pfeats_negative_mask = randsample(sum(leave3out_plab==-1),sum(leave3out_plab==1));% subsample -1 samples to balance classes
    leave3out_pfeats_positive_mask = leave3out_plab==1; %get all +1 samples
    leave3out_pfeats = [ leave3out_pfeats(leave3out_pfeats_positive_mask,:) ; leave3out_pfeats_negative(leave3out_pfeats_negative_mask,:) ];
    leave3out_plab = [ leave3out_plab(leave3out_pfeats_positive_mask,:) ; -1*ones(size(leave3out_pfeats_negative_mask,1),1) ];
    
    % Balance the -1 and +1 classes for each class
    leave4out_pfeats_negative = leave4out_pfeats(leave4out_plab==-1,:); %get all -1 samples
    leave4out_pfeats_negative_mask = randsample(sum(leave4out_plab==-1),sum(leave4out_plab==1));% subsample -1 samples to balance classes
    leave4out_pfeats_positive_mask = leave4out_plab==1; %get all +1 samples
    leave4out_pfeats = [ leave4out_pfeats(leave4out_pfeats_positive_mask,:) ; leave4out_pfeats_negative(leave4out_pfeats_negative_mask,:) ];
    leave4out_plab = [ leave4out_plab(leave4out_pfeats_positive_mask,:) ; -1*ones(size(leave4out_pfeats_negative_mask,1),1) ];
    
    % Balance the -1 and +1 classes for each class
    leave5out_pfeats_negative = leave5out_pfeats(leave5out_plab==-1,:); %get all -1 samples
    leave5out_pfeats_negative_mask = randsample(sum(leave5out_plab==-1),sum(leave5out_plab==1));% subsample -1 samples to balance classes
    leave5out_pfeats_positive_mask = leave5out_plab==1; %get all +1 samples
    leave5out_pfeats = [ leave5out_pfeats(leave5out_pfeats_positive_mask,:) ; leave5out_pfeats_negative(leave5out_pfeats_negative_mask,:) ];
    leave5out_plab = [ leave5out_plab(leave5out_pfeats_positive_mask,:) ; -1*ones(size(leave4out_pfeats_negative_mask,1),1) ];
    
    % Balance the -1 and +1 classes for each class
    leave6out_pfeats_negative = leave6out_pfeats(leave6out_plab==-1,:); %get all -1 samples
    leave6out_pfeats_negative_mask = randsample(sum(leave6out_plab==-1),sum(leave6out_plab==1));% subsample -1 samples to balance classes
    leave6out_pfeats_positive_mask = leave6out_plab==1; %get all +1 samples
    leave6out_pfeats = [ leave6out_pfeats(leave6out_pfeats_positive_mask,:) ; leave6out_pfeats_negative(leave6out_pfeats_negative_mask,:) ];
    leave6out_plab = [ leave6out_plab(leave6out_pfeats_positive_mask,:) ; -1*ones(size(leave5out_pfeats_negative_mask,1),1) ];
    
    
    %% Learn M
    [n,m] =  size(leave6out_pfeats);%all the class_s are the same size
    coldStep =  10^-6; coldLambda = 100; epsilon = .0001; %.9*10^-7;
    warmStep = .4*10^-8; warmLambda = 100; maxIter = 1000;
    
    [MT0out, ~, ~, ~]       = metric_trace( leave6out_pfeats , leave6out_plab , ones(n) , coldStep , maxIter , coldLambda , epsilon , zeros(m));
    [MT0out, ~, ~, TH0out]  = metric_trace( leave6out_pfeats , leave6out_plab , ones(n) , warmStep , maxIter , warmLambda , epsilon , MT0out );
    [MF0out, ~, ~, ~]       = metric_frob(  leave6out_pfeats , leave6out_plab , ones(n) , coldStep , maxIter , coldLambda , epsilon , zeros(m));
    [MF0out, ~, ~, FH0out]  = metric_frob(  leave6out_pfeats , leave6out_plab , ones(n) , warmStep , maxIter , warmLambda , epsilon , MF0out );
    
    [MT1out, ~, ~, ~]       = metric_trace( leave6out_pfeats , leave6out_plab , ones(n) , coldStep , maxIter , coldLambda , epsilon , zeros(m));
    [MT1out, ~, ~, TH1out]  = metric_trace( leave6out_pfeats , leave6out_plab , ones(n) , warmStep , maxIter , warmLambda , epsilon , MT1out );
    [MF1out, ~, ~, ~]       = metric_frob(  leave6out_pfeats , leave6out_plab , ones(n) , coldStep , maxIter , coldLambda , epsilon , zeros(m));
    [MF1out, ~, ~, FH1out]  = metric_frob(  leave6out_pfeats , leave6out_plab , ones(n) , warmStep , maxIter , warmLambda , epsilon , MF1out );
    
    [MT2out, ~, ~, ~]       = metric_trace( leave6out_pfeats , leave6out_plab , ones(n) , coldStep , maxIter , coldLambda , epsilon , zeros(m));
    [MT2out, ~, ~, TH2out]  = metric_trace( leave6out_pfeats , leave6out_plab , ones(n) , warmStep , maxIter , warmLambda , epsilon , MT2out );
    [MF2out, ~, ~, ~]       = metric_frob(  leave6out_pfeats , leave6out_plab , ones(n) , coldStep , maxIter , coldLambda , epsilon , zeros(m));
    [MF2out, ~, ~, FH2out]  = metric_frob(  leave6out_pfeats , leave6out_plab , ones(n) , warmStep , maxIter , warmLambda , epsilon , MF2out );
    
    [MT3out, ~, ~, ~]       = metric_trace( leave6out_pfeats , leave6out_plab , ones(n) , coldStep , maxIter , coldLambda , epsilon , zeros(m));
    [MT3out, ~, ~, TH3out]  = metric_trace( leave6out_pfeats , leave6out_plab , ones(n) , warmStep , maxIter , warmLambda , epsilon , MT3out );
    [MF3out, ~, ~, ~]       = metric_frob(  leave6out_pfeats , leave6out_plab , ones(n) , coldStep , maxIter , coldLambda , epsilon , zeros(m));
    [MF3out, ~, ~, FH3out]  = metric_frob(  leave6out_pfeats , leave6out_plab , ones(n) , warmStep , maxIter , warmLambda , epsilon , MF3out );
    
    [MT4out, ~, ~, ~]       = metric_trace( leave6out_pfeats , leave6out_plab , ones(n) , coldStep , maxIter , coldLambda , epsilon , zeros(m));
    [MT4out, ~, ~, TH4out]  = metric_trace( leave6out_pfeats , leave6out_plab , ones(n) , warmStep , maxIter , warmLambda , epsilon , MT4out );
    [MF4out, ~, ~, ~]       = metric_frob(  leave6out_pfeats , leave6out_plab , ones(n) , coldStep , maxIter , coldLambda , epsilon , zeros(m));
    [MF4out, ~, ~, FH4out]  = metric_frob(  leave6out_pfeats , leave6out_plab , ones(n) , warmStep , maxIter , warmLambda , epsilon , MF4out );
    
    [MT5out, ~, ~, ~]       = metric_trace( leave6out_pfeats , leave6out_plab , ones(n) , coldStep , maxIter , coldLambda , epsilon , zeros(m));
    [MT5out, ~, ~, TH5out]  = metric_trace( leave6out_pfeats , leave6out_plab , ones(n) , warmStep , maxIter , warmLambda , epsilon , MT5out );
    [MF5out, ~, ~, ~]       = metric_frob(  leave6out_pfeats , leave6out_plab , ones(n) , coldStep , maxIter , coldLambda , epsilon , zeros(m));
    [MF5out, ~, ~, FH5out]  = metric_frob(  leave6out_pfeats , leave6out_plab , ones(n) , warmStep , maxIter , warmLambda , epsilon , MF5out );
    
    [MT6out, ~, ~, ~]       = metric_trace( leave6out_pfeats , leave6out_plab , ones(n) , coldStep , maxIter , coldLambda , epsilon , zeros(m));
    [MT6out, ~, ~, TH6out]  = metric_trace( leave6out_pfeats , leave6out_plab , ones(n) , warmStep , maxIter , warmLambda , epsilon , MT6out );
    [MF6out, ~, ~, ~]       = metric_frob(  leave6out_pfeats , leave6out_plab , ones(n) , coldStep , maxIter , coldLambda , epsilon , zeros(m));
    [MF6out, ~, ~, FH6out]  = metric_frob(  leave6out_pfeats , leave6out_plab , ones(n) , warmStep , maxIter , warmLambda , epsilon , MF6out );
    
    
    save('leaveOneOutModelsTrained.mat');
end




%% Make a random dataset to train the transfer learning task
if(0)
    
    mask0test = test_labels==0;
    mask1test = test_labels==1;
    mask2test = test_labels==2; % Get each class from test data
    mask3test = test_labels==3;
    mask4test = test_labels==4;
    mask5test = test_labels==5;
    mask6test = test_labels==6;
    
    pos0_x_test = test_data(mask0test,:);
    pos1_x_test = test_data(mask1test,:);
    pos2_x_test = test_data(mask2test,:);
    pos3_x_test = test_data(mask3test,:);
    pos4_x_test = test_data(mask4test,:);
    pos5_x_test = test_data(mask5test,:);
    pos6_x_test = test_data(mask6test,:);
    
    pos0_y_test = test_labels(mask0test,:);
    pos1_y_test = test_labels(mask1test,:);
    pos2_y_test = test_labels(mask2test,:);
    pos3_y_test = test_labels(mask3test,:);
    pos4_y_test = test_labels(mask4test,:);
    pos5_y_test = test_labels(mask5test,:);
    pos6_y_test = test_labels(mask6test,:);
    
    n=300; %There are 300 samples from each class, so let's make -1,+1 balanced
    maskRand0_test = randsample(length(test_labels(~mask0test)),n);
    maskRand1_test = randsample(length(test_labels(~mask1test)),n);
    maskRand2_test = randsample(length(test_labels(~mask2test)),n);
    maskRand3_test = randsample(length(test_labels(~mask3test)),n);
    maskRand4_test = randsample(length(test_labels(~mask4test)),n);
    maskRand5_test = randsample(length(test_labels(~mask5test)),n);
    maskRand6_test = randsample(length(test_labels(~mask6test)),n);
    
    rand0neg_x_test = test_data(~mask0test,:);
    rand0neg_x_test = rand0neg_x_test(maskRand0_test,:);
    rand1neg_x_test = test_data(~mask1test,:);
    rand1neg_x_test = rand1neg_x_test(maskRand1_test,:);
    rand2neg_x_test = test_data(~mask2test,:);
    rand2neg_x_test = rand2neg_x_test(maskRand2_test,:);
    rand3neg_x_test = test_data(~mask3test,:);
    rand3neg_x_test = rand3neg_x_test(maskRand3_test,:);
    rand4neg_x_test = test_data(~mask4test,:);
    rand4neg_x_test = rand4neg_x_test(maskRand4_test,:);
    rand5neg_x_test = test_data(~mask5test,:);
    rand5neg_x_test = rand5neg_x_test(maskRand5_test,:);
    rand6neg_x_test = test_data(~mask6test,:);
    rand6neg_x_test = rand6neg_x_test(maskRand6_test,:);
    
    rand0neg_y_test = test_labels(~mask0test,:);
    rand0neg_y_test = rand0neg_y_test(maskRand0_test,:);
    rand1neg_y_test = test_labels(~mask1test,:);
    rand1neg_y_test = rand1neg_y_test(maskRand1_test,:);
    rand2neg_y_test = test_labels(~mask2test,:);
    rand2neg_y_test = rand2neg_y_test(maskRand2_test,:);
    rand3neg_y_test = test_labels(~mask3test,:);
    rand3neg_y_test = rand3neg_y_test(maskRand3_test,:);
    rand4neg_y_test = test_labels(~mask4test,:);
    rand4neg_y_test = rand4neg_y_test(maskRand4_test,:);
    rand5neg_y_test = test_labels(~mask5test,:);
    rand5neg_y_test = rand5neg_y_test(maskRand5_test,:);
    rand6neg_y_test = test_labels(~mask6test,:);
    rand6neg_y_test = rand6neg_y_test(maskRand6_test,:); 
    
    
    class0_x_test = [pos0_x_test ; rand0neg_x_test];
    class1_x_test = [pos1_x_test ; rand1neg_x_test];
    class2_x_test = [pos2_x_test ; rand2neg_x_test];
    class3_x_test = [pos3_x_test ; rand3neg_x_test];
    class4_x_test = [pos4_x_test ; rand4neg_x_test];
    class5_x_test = [pos5_x_test ; rand5neg_x_test];
    class6_x_test = [pos6_x_test ; rand6neg_x_test];
    
    class0_y_test = [ pos0_y_test ; rand0neg_y_test ];
    class1_y_test = [ pos1_y_test ; rand1neg_y_test ];
    class2_y_test = [ pos2_y_test ; rand2neg_y_test ];
    class3_y_test = [ pos3_y_test ; rand3neg_y_test ];
    class4_y_test = [ pos4_y_test ; rand4neg_y_test ];
    class5_y_test = [ pos5_y_test ; rand5neg_y_test ];
    class6_y_test = [ pos6_y_test ; rand6neg_y_test ];
    
    save('targetTestSetsCreated.mat');
end



%% Run these tests with a few different amounts of data in the LAST
%(transfer) learning task

coldSteps = [10e-4,10e-4,10e-4,10e-4,10e-4,10e-4,10e-4,10e-4,10e-4,10e-4,10e-4,10e-4];
warmSteps = [10e-6,10e-6,10e-6,10e-6,10e-6,10e-6,10e-6,10e-6,10e-6,10e-6,10e-6,10e-6];
number_of_data_samples = [6,9,12,15,18,21,24,27,30];

for ii=1:9
    
    N=number_of_data_samples(ii);
    

    
    %Make some test sets, this is coming from code earlier in this script
    %pasted here for ease of referencing:
    % %     %Pull features associated with mask
    % %     pos0_x = train_data(mask0,:);
    % %     pos1_x = train_data(mask1,:);
    % %     pos2_x = train_data(mask2,:);
    % %     pos3_x = train_data(mask3,:);
    % %     pos4_x = train_data(mask4,:);
    % %     pos5_x = train_data(mask5,:);
    % %     pos6_x = train_data(mask6,:);
    
    pos0_mask = randsample(size(pos0_x,1),N);
    pos1_mask = randsample(size(pos1_x,1),N);
    pos2_mask = randsample(size(pos2_x,1),N);
    pos3_mask = randsample(size(pos3_x,1),N);
    pos4_mask = randsample(size(pos4_x,1),N);
    pos5_mask = randsample(size(pos5_x,1),N);
    pos6_mask = randsample(size(pos6_x,1),N);
    
    neg0_mask = randsample(size(rand0neg_x,1),N);
    neg1_mask = randsample(size(rand1neg_x,1),N);
    neg2_mask = randsample(size(rand2neg_x,1),N);
    neg3_mask = randsample(size(rand3neg_x,1),N);
    neg4_mask = randsample(size(rand4neg_x,1),N);
    neg5_mask = randsample(size(rand5neg_x,1),N);
    neg6_mask = randsample(size(rand6neg_x,1),N);
    
    target0_x = [pos0_x(pos0_mask,:) ; rand0neg_x(neg0_mask,:)];
    target1_x = [pos1_x(pos1_mask,:) ; rand1neg_x(neg1_mask,:)];
    target2_x = [pos2_x(pos2_mask,:) ; rand2neg_x(neg2_mask,:)];
    target3_x = [pos3_x(pos3_mask,:) ; rand3neg_x(neg3_mask,:)];
    target4_x = [pos4_x(pos4_mask,:) ; rand4neg_x(neg4_mask,:)];
    target5_x = [pos5_x(pos5_mask,:) ; rand5neg_x(neg5_mask,:)];
    target6_x = [pos6_x(pos6_mask,:) ; rand6neg_x(neg6_mask,:)];
    
    
    
    
    %This is a one class vs all, so overwrite real dissimilar labels to -1
    %Same thing is true for similar labels, which become +1
    target0_y = [ pos0_y(pos0_mask,:) ; rand0neg_y(neg0_mask,:) ];
    target1_y = [ pos1_y(pos1_mask,:) ; rand1neg_y(neg1_mask,:) ];
    target2_y = [ pos2_y(pos2_mask,:) ; rand2neg_y(neg2_mask,:) ];
    target3_y = [ pos3_y(pos3_mask,:) ; rand3neg_y(neg3_mask,:) ];
    target4_y = [ pos4_y(pos4_mask,:) ; rand4neg_y(neg4_mask,:) ];
    target5_y = [ pos5_y(pos5_mask,:) ; rand5neg_y(neg5_mask,:) ];
    target6_y = [ pos6_y(pos6_mask,:) ; rand6neg_y(neg6_mask,:) ];
    

    

    
    % Need to train the transfer metric on pairwise labels
    [ target0_x_bin , target0_y_bin ] = similarize_data( target0_x , target0_y );
    [ target1_x_bin , target1_y_bin ] = similarize_data( target1_x , target1_y );
    [ target2_x_bin , target2_y_bin ] = similarize_data( target2_x , target2_y );
    [ target3_x_bin , target3_y_bin ] = similarize_data( target3_x , target3_y );
    [ target4_x_bin , target4_y_bin ] = similarize_data( target4_x , target4_y );
    [ target5_x_bin , target5_y_bin ] = similarize_data( target5_x , target5_y );
    [ target6_x_bin , target6_y_bin ] = similarize_data( target6_x , target6_y );
    
    
    
    % Now learn the left out metrics by regularizing on the previously
    % learned metrics
    coldStep = coldSteps(ii); coldLambda = 100; epsilon = .0001; coldAlpha = 100;
    warmStep = warmSteps(ii); warmLambda = 100; maxIter = 10000; warmAlpha = 100;
    noRegStep = coldSteps(ii);
    [n,m]=size(target0_x_bin);
    [ MT_transfer0, ~, ~ , t] = metric_trace_transfer_bn(  target0_x_bin , target0_y_bin , ones(n) , coldStep , maxIter , coldLambda , epsilon , coldAlpha , MT0out , zeros(m));
    [ MT_transfer0, ~, ~ , ~] = metric_trace_transfer(     target0_x_bin , target0_y_bin , ones(n) , t/100 , maxIter , warmLambda , epsilon , warmAlpha , MT0out , MT_transfer0 );
    [ MF_transfer0, ~, ~ , t] = metric_frob_transfer_bn(   target0_x_bin , target0_y_bin , ones(n) , coldStep , maxIter , coldLambda , epsilon , coldAlpha , MF0out , zeros(m));
    [ MF_transfer0, ~, ~ , ~] = metric_frob_transfer(      target0_x_bin , target0_y_bin , ones(n) , t/100 , maxIter , warmLambda , epsilon , warmAlpha , MF0out , MF_transfer0 );
    [ ME_transfer0, ~, ~ , t] = metric_trace_transfer_bn(  target0_x_bin , target0_y_bin , ones(n) , noRegStep , maxIter , 0 , epsilon , 0 , MT0out , zeros(m));
    [ ME_transfer0, ~, ~ , ~] = metric_trace_transfer(     target0_x_bin , target0_y_bin , ones(n) , t/100 , maxIter , 0 , epsilon , 0 , MT0out , ME_transfer0);

    [ MT_transfer1, ~, ~ , t] = metric_trace_transfer_bn(  target1_x_bin , target1_y_bin , ones(n) , coldStep , maxIter , coldLambda , epsilon , coldAlpha , MT1out , zeros(m));
    [ MT_transfer1, ~, ~ , ~] = metric_trace_transfer(     target1_x_bin , target1_y_bin , ones(n) , t/100 , maxIter , warmLambda , epsilon , warmAlpha , MT1out , MT_transfer1 );
    [ MF_transfer1, ~, ~ , t] = metric_frob_transfer_bn(   target1_x_bin , target1_y_bin , ones(n) , coldStep , maxIter , coldLambda , epsilon , coldAlpha , MF1out , zeros(m));
    [ MF_transfer1, ~, ~ , ~] = metric_frob_transfer(      target1_x_bin , target1_y_bin , ones(n) , t/100 , maxIter , warmLambda , epsilon , warmAlpha , MF1out , MF_transfer1 );
    [ ME_transfer1, ~, ~ , t] = metric_trace_transfer_bn(  target1_x_bin , target1_y_bin , ones(n) , noRegStep , maxIter , 0 , epsilon , 0 , MT1out , zeros(m));
    [ ME_transfer1, ~, ~ , ~] = metric_trace_transfer(     target1_x_bin , target1_y_bin , ones(n) , t/100 , maxIter , 0 , epsilon , 0 , MT1out , ME_transfer1 );
    
    [ MT_transfer2, ~, ~ , t] = metric_trace_transfer_bn(  target2_x_bin , target2_y_bin , ones(n) , coldStep , maxIter , coldLambda , epsilon , coldAlpha , MT2out , zeros(m));
    [ MT_transfer2, ~, ~ , ~] = metric_trace_transfer(     target2_x_bin , target2_y_bin , ones(n) , t/100 , maxIter , warmLambda , epsilon , warmAlpha , MT2out , MT_transfer2 );
    [ MF_transfer2, ~, ~ , t] = metric_frob_transfer_bn(   target2_x_bin , target2_y_bin , ones(n) , coldStep , maxIter , coldLambda , epsilon , coldAlpha , MF2out , zeros(m));
    [ MF_transfer2, ~, ~ , ~] = metric_frob_transfer(      target2_x_bin , target2_y_bin , ones(n) , t/100 , maxIter , warmLambda , epsilon , warmAlpha , MF2out , MF_transfer2 );
    [ ME_transfer2, ~, ~ , t] = metric_trace_transfer_bn(  target2_x_bin , target2_y_bin , ones(n) , noRegStep , maxIter , 0 , epsilon , 0 , MT2out , zeros(m));
    [ ME_transfer2, ~, ~ , ~] = metric_trace_transfer(     target2_x_bin , target2_y_bin , ones(n) , t/100 , maxIter , 0 , epsilon , 0 , MT2out , ME_transfer2);

    [ MT_transfer3, ~, ~ , t] = metric_trace_transfer_bn(  target3_x_bin , target3_y_bin , ones(n) , coldStep , maxIter , coldLambda , epsilon , coldAlpha , MT3out , zeros(m));
    [ MT_transfer3, ~, ~ , ~] = metric_trace_transfer(     target3_x_bin , target3_y_bin , ones(n) , t/100 , maxIter , warmLambda , epsilon , warmAlpha , MT3out , MT_transfer3 );
    [ MF_transfer3, ~, ~ , t] = metric_frob_transfer_bn(   target3_x_bin , target3_y_bin , ones(n) , coldStep , maxIter , coldLambda , epsilon , coldAlpha , MF3out , zeros(m));
    [ MF_transfer3, ~, ~ , ~] = metric_frob_transfer(      target3_x_bin , target3_y_bin , ones(n) , t/100 , maxIter , warmLambda , epsilon , warmAlpha , MF3out , MF_transfer3 );
    [ ME_transfer3, ~, ~ , t] = metric_trace_transfer_bn(  target3_x_bin , target3_y_bin , ones(n) , noRegStep , maxIter , 0 , epsilon , 0 , MT3out , zeros(m));
    [ ME_transfer3, ~, ~ , ~] = metric_trace_transfer(     target3_x_bin , target3_y_bin , ones(n) , t/100 , maxIter , 0 , epsilon , 0 , MT3out , ME_transfer3 );

    [ MT_transfer4, ~, ~ , t] = metric_trace_transfer_bn(  target4_x_bin , target4_y_bin , ones(n) , coldStep , maxIter , coldLambda , epsilon , coldAlpha , MT4out , zeros(m));
    [ MT_transfer4, ~, ~ , ~] = metric_trace_transfer(     target4_x_bin , target4_y_bin , ones(n) , t/100 , maxIter , warmLambda , epsilon , warmAlpha , MT4out , MT_transfer4 );
    [ MF_transfer4, ~, ~ , t] = metric_frob_transfer_bn(   target4_x_bin , target4_y_bin , ones(n) , coldStep , maxIter , coldLambda , epsilon , coldAlpha , MF4out , zeros(m));
    [ MF_transfer4, ~, ~ , ~] = metric_frob_transfer(      target4_x_bin , target4_y_bin , ones(n) , t/100 , maxIter , warmLambda , epsilon , warmAlpha , MF4out , MF_transfer4 );
    [ ME_transfer4, ~, ~ , t] = metric_trace_transfer_bn(  target4_x_bin , target4_y_bin , ones(n) , noRegStep , maxIter , 0 , epsilon , 0 , MT4out , zeros(m));
    [ ME_transfer4, ~, ~ , ~] = metric_trace_transfer(     target4_x_bin , target4_y_bin , ones(n) , t/100 , maxIter , 0 , epsilon , 0 , MT4out , ME_transfer4 );

    [ MT_transfer5, ~, ~ , t] = metric_trace_transfer_bn(  target5_x_bin , target5_y_bin , ones(n) , coldStep , maxIter , coldLambda , epsilon , coldAlpha , MT5out , zeros(m));
    [ MT_transfer5, ~, ~ , ~] = metric_trace_transfer(     target5_x_bin , target5_y_bin , ones(n) , t/100 , maxIter , warmLambda , epsilon , warmAlpha , MT5out , MT_transfer5 );
    [ MF_transfer5, ~, ~ , t] = metric_frob_transfer_bn(   target5_x_bin , target5_y_bin , ones(n) , coldStep , maxIter , coldLambda , epsilon , coldAlpha , MF5out , zeros(m));
    [ MF_transfer5, ~, ~ , ~] = metric_frob_transfer(      target5_x_bin , target5_y_bin , ones(n) , t/100 , maxIter , warmLambda , epsilon , warmAlpha , MF5out , MF_transfer5 );
    [ ME_transfer5, ~, ~ , t] = metric_trace_transfer_bn(  target5_x_bin , target5_y_bin , ones(n) , noRegStep , maxIter , 0 , epsilon , 0 , MT5out , zeros(m));
    [ ME_transfer5, ~, ~ , ~] = metric_trace_transfer(     target5_x_bin , target5_y_bin , ones(n) , t/100 , maxIter , 0 , epsilon , 0 , MT5out , ME_transfer5 );

    [ MT_transfer6, ~, ~ , t] = metric_trace_transfer_bn(  target6_x_bin , target6_y_bin , ones(n) , coldStep , maxIter , coldLambda , epsilon , coldAlpha , MT6out , zeros(m));
    [ MT_transfer6, ~, ~ , ~] = metric_trace_transfer(     target6_x_bin , target6_y_bin , ones(n) , t/100 , maxIter , warmLambda , epsilon , warmAlpha , MT6out , MT_transfer6 );
    [ MF_transfer6, ~, ~ , t] = metric_frob_transfer_bn(   target6_x_bin , target6_y_bin , ones(n) , coldStep , maxIter , coldLambda , epsilon , coldAlpha , MF6out , zeros(m));
    [ MF_transfer6, ~, ~ , ~] = metric_frob_transfer(      target6_x_bin , target6_y_bin , ones(n) , t/100 , maxIter , warmLambda , epsilon , warmAlpha , MF6out , MF_transfer6 );
    [ ME_transfer6, ~, ~ , t] = metric_trace_transfer_bn(  target6_x_bin , target6_y_bin , ones(n) , noRegStep , maxIter , 0 , epsilon , 0 , MT6out , zeros(m));
    [ ME_transfer6, ~, ~ , ~] = metric_trace_transfer(     target6_x_bin , target6_y_bin , ones(n) , t/100 , maxIter , 0 , epsilon , 0 , MT6out , ME_transfer6 );

    
    testaccs=[];
    trainaccs=[];
    for i=1:5
        newTrainAccRow=zeros(1,35);
        newTestAccRow = zeros(1,35);
        
        
        % Transfering to 0 Class
        setGlobalM(MT_transfer0);
        lastTask0TT = fitcknn(target0_x,target0_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask0TT_TrainAcc=1-resubLoss(lastTask0TT);
        lastTask0TT_labels = predict( lastTask0TT , class0_x_test );
        lastTask0TT_TestAcc = sum(lastTask0TT_labels==class0_y_test)/size(class0_y_test,1);
        newTrainAccRow(1)=lastTask0TT_TrainAcc;
        newTestAccRow(1)=lastTask0TT_TestAcc;
        
        setGlobalM(MF_transfer0);
        lastTask0TF = fitcknn(target0_x,target0_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask0TF_TrainAcc=1-resubLoss(lastTask0TF);
        lastTask0TF_labels = predict( lastTask0TF , class0_x_test );
        lastTask0TF_TestAcc = sum(lastTask0TF_labels==class0_y_test)/size(class0_y_test,1);
        newTrainAccRow(2)=lastTask0TF_TrainAcc;
        newTestAccRow(2)=lastTask0TF_TestAcc;
        
        setGlobalM(MT0out);
        lastTask0T = fitcknn(target0_x,target0_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask0T_TrainAcc=1-resubLoss(lastTask0T);
        lastTask0T_labels = predict( lastTask0T , class0_x_test );
        lastTask0T_TestAcc = sum(lastTask0T_labels==class0_y_test)/size(class0_y_test,1);
        newTrainAccRow(3)=lastTask0T_TrainAcc;
        newTestAccRow(3)=lastTask0T_TestAcc;
        
        setGlobalM(MF0out);
        lastTask0F = fitcknn(target0_x,target0_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask0F_TrainAcc=1-resubLoss(lastTask0F);
        lastTask0F_labels = predict( lastTask0F , class0_x_test );
        lastTask0F_TestAcc = sum(lastTask0F_labels==class0_y_test)/size(class0_y_test,1);
        newTrainAccRow(4)=lastTask0F_TrainAcc;
        newTestAccRow(4)=lastTask0F_TestAcc;
        
        setGlobalM(ME_transfer0);
        lastTask0E = fitcknn(target0_x,target0_y,'NumNeighbors',i);
        lastTask0E_TrainAcc=1-resubLoss(lastTask0E);
        lastTask0E_labels = predict( lastTask0E , class0_x_test );
        lastTask0E_TestAcc = sum(lastTask0E_labels==class0_y_test)/size(class0_y_test,1);
        newTrainAccRow(5)=lastTask0E_TrainAcc;
        newTestAccRow(5)=lastTask0E_TestAcc;
        
        
        
        
        
        
        
        % Transfering to 1 Class
        setGlobalM(MT_transfer1);
        lastTask1 = fitcknn(target1_x,target1_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask1_TrainAcc=1-resubLoss(lastTask1);
        lastTask1_labels = predict( lastTask1 , class1_x_test );
        lastTask1_TestAcc = sum(lastTask1_labels==class1_y_test)/size(class1_y_test,1);
        newTrainAccRow(6)=lastTask1_TrainAcc;
        newTestAccRow(6)=lastTask1_TestAcc;
        
        setGlobalM(MF_transfer1);
        lastTask1TF = fitcknn(target1_x,target1_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask1TF_TrainAcc=1-resubLoss(lastTask1TF);
        lastTask1TF_labels = predict( lastTask1TF , class1_x_test );
        lastTask1TF_TestAcc = sum(lastTask1TF_labels==class1_y_test)/size(class1_y_test,1);
        newTrainAccRow(7)=lastTask1TF_TrainAcc;
        newTestAccRow(7)=lastTask1TF_TestAcc;
        
        setGlobalM(MT1out);
        lastTask1T = fitcknn(target1_x,target1_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask1T_TrainAcc=1-resubLoss(lastTask1T);
        lastTask1T_labels = predict( lastTask1T , class1_x_test );
        lastTask1T_TestAcc = sum(lastTask1T_labels==class1_y_test)/size(class1_y_test,1);
        newTrainAccRow(8)=lastTask1T_TrainAcc;
        newTestAccRow(8)=lastTask1T_TestAcc;
        
        setGlobalM(MF1out);
        lastTask1F = fitcknn(target1_x,target1_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask1F_TrainAcc=1-resubLoss(lastTask1F);
        lastTask1F_labels = predict( lastTask1F , class1_x_test );
        lastTask1F_TestAcc = sum(lastTask1F_labels==class1_y_test)/size(class1_y_test,1);
        newTrainAccRow(9)=lastTask1F_TrainAcc;
        newTestAccRow(9)=lastTask1F_TestAcc;
        
        setGlobalM(ME_transfer1);
        lastTask1E = fitcknn(target1_x,target1_y,'NumNeighbors',i);
        lastTask1E_TrainAcc=1-resubLoss(lastTask1E);
        lastTask1E_labels = predict( lastTask1E , class1_x_test );
        lastTask1E_TestAcc = sum(lastTask1E_labels==class1_y_test)/size(class1_y_test,1);
        newTrainAccRow(10)=lastTask1E_TrainAcc;
        newTestAccRow(10)=lastTask1E_TestAcc;
        
        
        % Transfering to 2 Class
        setGlobalM(MT_transfer2);
        lastTask2 = fitcknn(target2_x,target2_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask2_TrainAcc=1-resubLoss(lastTask2);
        lastTask2_labels = predict( lastTask2 , class2_x_test );
        lastTask2_TestAcc = sum(lastTask2_labels==class2_y_test)/size(class2_y_test,1);
        newTrainAccRow(11)=lastTask2_TrainAcc;
        newTestAccRow(11)=lastTask2_TestAcc;
        
        setGlobalM(MF_transfer2);
        lastTask2TF = fitcknn(target2_x,target2_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask2TF_TrainAcc=1-resubLoss(lastTask2TF);
        lastTask2TF_labels = predict( lastTask2TF , class2_x_test );
        lastTask2TF_TestAcc = sum(lastTask2TF_labels==class2_y_test)/size(class2_y_test,1);
        newTrainAccRow(12)=lastTask2TF_TrainAcc;
        newTestAccRow(12)=lastTask2TF_TestAcc;
        
        setGlobalM(MT2out);
        lastTask2T = fitcknn(target2_x,target2_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask2T_TrainAcc=1-resubLoss(lastTask2T);
        lastTask2T_labels = predict( lastTask2T , class1_x_test );
        lastTask2T_TestAcc = sum(lastTask2T_labels==class2_y_test)/size(class2_y_test,1);
        newTrainAccRow(13)=lastTask2T_TrainAcc;
        newTestAccRow(13)=lastTask2T_TestAcc;
        
        setGlobalM(MF2out);
        lastTask2F = fitcknn(target2_x,target2_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask2F_TrainAcc=1-resubLoss(lastTask2F);
        lastTask2F_labels = predict( lastTask2F , class2_x_test );
        lastTask2F_TestAcc = sum(lastTask2F_labels==class2_y_test)/size(class2_y_test,1);
        newTrainAccRow(14)=lastTask2F_TrainAcc;
        newTestAccRow(14)=lastTask2F_TestAcc;
        
        setGlobalM(ME_transfer2);
        lastTask2E = fitcknn(target2_x,target2_y,'NumNeighbors',i);
        lastTask2E_TrainAcc=1-resubLoss(lastTask2E);
        lastTask2E_labels = predict( lastTask2E , class2_x_test );
        lastTask2E_TestAcc = sum(lastTask2E_labels==class2_y_test)/size(class2_y_test,1);
        newTrainAccRow(15)=lastTask2E_TrainAcc;
        newTestAccRow(15)=lastTask2E_TestAcc;
        
        
        % Transfering to 3 Class
        setGlobalM(MT_transfer3);
        lastTask3 = fitcknn(target3_x,target3_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask3_TrainAcc=1-resubLoss(lastTask3);
        lastTask3_labels = predict( lastTask3 , class3_x_test );
        lastTask3_TestAcc = sum(lastTask3_labels==class3_y_test)/size(class3_y_test,1);
        newTrainAccRow(16)=lastTask3_TrainAcc;
        newTestAccRow(16)=lastTask3_TestAcc;
        
        setGlobalM(MF_transfer3);
        lastTask3TF = fitcknn(target3_x,target3_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask3TF_TrainAcc=1-resubLoss(lastTask3TF);
        lastTask3TF_labels = predict( lastTask3TF , class3_x_test );
        lastTask3TF_TestAcc = sum(lastTask3TF_labels==class3_y_test)/size(class3_y_test,1);
        newTrainAccRow(17)=lastTask3TF_TrainAcc;
        newTestAccRow(17)=lastTask3TF_TestAcc;
        
        setGlobalM(MT3out);
        lastTask3T = fitcknn(target3_x,target3_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask3T_TrainAcc=1-resubLoss(lastTask3T);
        lastTask3T_labels = predict( lastTask3T , class3_x_test );
        lastTask3T_TestAcc = sum(lastTask3T_labels==class3_y_test)/size(class3_y_test,1);
        newTrainAccRow(18)=lastTask3T_TrainAcc;
        newTestAccRow(18)=lastTask3T_TestAcc;
        
        setGlobalM(MF3out);
        lastTask3F = fitcknn(target3_x,target3_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask3F_TrainAcc=1-resubLoss(lastTask3F);
        lastTask3F_labels = predict( lastTask3F , class3_x_test );
        lastTask3F_TestAcc = sum(lastTask0F_labels==class3_y_test)/size(class3_y_test,1);
        newTrainAccRow(19)=lastTask3F_TrainAcc;
        newTestAccRow(19)=lastTask3F_TestAcc;
        
        setGlobalM(ME_transfer3);
        lastTask3E = fitcknn(target3_x,target3_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask3E_TrainAcc=1-resubLoss(lastTask3E);
        lastTask3E_labels = predict( lastTask3E , class3_x_test );
        lastTask3E_TestAcc = sum(lastTask3E_labels==class3_y_test)/size(class3_y_test,1);
        newTrainAccRow(20)=lastTask3E_TrainAcc;
        newTestAccRow(20)=lastTask3E_TestAcc;
        
        
        % Transfering to 4 Class
        setGlobalM(MT_transfer4);
        lastTask4 = fitcknn(target4_x,target4_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask4_TrainAcc=1-resubLoss(lastTask4);
        lastTask4_labels = predict( lastTask4 , class4_x_test );
        lastTask4_TestAcc = sum(lastTask4_labels==class4_y_test)/size(class4_y_test,1);
        newTrainAccRow(21)=lastTask4_TrainAcc;
        newTestAccRow(21)=lastTask4_TestAcc;
        
        setGlobalM(MF_transfer4);
        lastTask4TF = fitcknn(target4_x,target4_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask4TF_TrainAcc=1-resubLoss(lastTask4TF);
        lastTask4TF_labels = predict( lastTask4TF , class4_x_test );
        lastTask4TF_TestAcc = sum(lastTask4TF_labels==class4_y_test)/size(class4_y_test,1);
        newTrainAccRow(22)=lastTask4TF_TrainAcc;
        newTestAccRow(22)=lastTask4TF_TestAcc;
        
        setGlobalM(MT4out);
        lastTask4T = fitcknn(target4_x,target4_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask4T_TrainAcc=1-resubLoss(lastTask4T);
        lastTask4T_labels = predict( lastTask4T , class4_x_test );
        lastTask4T_TestAcc = sum(lastTask4T_labels==class4_y_test)/size(class4_y_test,1);
        newTrainAccRow(23)=lastTask4T_TrainAcc;
        newTestAccRow(23)=lastTask4T_TestAcc;
        
        setGlobalM(MF4out);
        lastTask4F = fitcknn(target4_x,target4_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask4F_TrainAcc=1-resubLoss(lastTask4F);
        lastTask4F_labels = predict( lastTask4F , class4_x_test );
        lastTask4F_TestAcc = sum(lastTask4F_labels==class4_y_test)/size(class4_y_test,1);
        newTrainAccRow(24)=lastTask4F_TrainAcc;
        newTestAccRow(24)=lastTask4F_TestAcc;
        
        setGlobalM(ME_transfer4);
        lastTask4E = fitcknn(target4_x,target4_y,'NumNeighbors',i);
        lastTask4E_TrainAcc=1-resubLoss(lastTask4E);
        lastTask4E_labels = predict( lastTask4E , class4_x_test );
        lastTask4E_TestAcc = sum(lastTask4E_labels==class4_y_test)/size(class4_y_test,1);
        newTrainAccRow(25)=lastTask4E_TrainAcc;
        newTestAccRow(25)=lastTask4E_TestAcc;
        
        
        
        
        
        
        % Transfering to 5 Class
        setGlobalM(MT_transfer5);
        lastTask5 = fitcknn(target5_x,target5_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask5_TrainAcc=1-resubLoss(lastTask5);
        lastTask5_labels = predict( lastTask5 , class5_x_test );
        lastTask5_TestAcc = sum(lastTask5_labels==class5_y_test)/size(class5_y_test,1);
        newTrainAccRow(26)=lastTask5_TrainAcc;
        newTestAccRow(26)=lastTask5_TestAcc;
        
        setGlobalM(MF_transfer5);
        lastTask5TF = fitcknn(target5_x,target5_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask5TF_TrainAcc=1-resubLoss(lastTask5TF);
        lastTask5TF_labels = predict( lastTask5TF , class5_x_test );
        lastTask5TF_TestAcc = sum(lastTask5TF_labels==class5_y_test)/size(class5_y_test,1);
        newTrainAccRow(27)=lastTask5TF_TrainAcc;
        newTestAccRow(27)=lastTask5TF_TestAcc;
        
        
        
        setGlobalM(MT5out);
        lastTask5T = fitcknn(target5_x,target5_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask5T_TrainAcc=1-resubLoss(lastTask5T);
        lastTask5T_labels = predict( lastTask5T , class5_x_test );
        lastTask5T_TestAcc = sum(lastTask5T_labels==class5_y_test)/size(class5_y_test,1);
        newTrainAccRow(28)=lastTask5T_TrainAcc;
        newTestAccRow(28)=lastTask5T_TestAcc;
        
        setGlobalM(MF5out);
        lastTask5F = fitcknn(target5_x,target5_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask5F_TrainAcc=1-resubLoss(lastTask5F);
        lastTask5F_labels = predict( lastTask5F , class5_x_test );
        lastTask5F_TestAcc = sum(lastTask5F_labels==class5_y_test)/size(class5_y_test,1);
        newTrainAccRow(29)=lastTask5F_TrainAcc;
        newTestAccRow(29)=lastTask5F_TestAcc;
        
        setGlobalM(ME_transfer5);
        lastTask5E = fitcknn(target5_x,target5_y,'NumNeighbors',i);
        lastTask5E_TrainAcc=1-resubLoss(lastTask5E);
        lastTask5E_labels = predict( lastTask5E , class5_x_test );
        lastTask5E_TestAcc = sum(lastTask5E_labels==class5_y_test)/size(class5_y_test,1);
        newTrainAccRow(30)=lastTask5E_TrainAcc;
        newTestAccRow(30)=lastTask5E_TestAcc;
        
        
        
        
        
        
        
        % Transfering to 6 Class
        setGlobalM(MT_transfer6);
        lastTask6 = fitcknn(target6_x,target6_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask6_TrainAcc=1-resubLoss(lastTask6);
        lastTask6_labels = predict( lastTask6 , class6_x_test );
        lastTask6_TestAcc = sum(lastTask6_labels==class6_y_test)/size(class6_y_test,1);
        newTrainAccRow(31)=lastTask6_TrainAcc;
        newTestAccRow(31)=lastTask6_TestAcc;
        
        setGlobalM(MF_transfer6);
        lastTask6TF = fitcknn(target6_x,target6_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask6TF_TrainAcc=1-resubLoss(lastTask6TF);
        lastTask6TF_labels = predict( lastTask6TF , class6_x_test );
        lastTask6TF_TestAcc = sum(lastTask6TF_labels==class6_y_test)/size(class6_y_test,1);
        newTrainAccRow(32)=lastTask6TF_TrainAcc;
        newTestAccRow(32)=lastTask6TF_TestAcc;
        
        
        setGlobalM(MT6out);
        lastTask6T = fitcknn(target6_x,target6_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask6T_TrainAcc=1-resubLoss(lastTask6T);
        lastTask6T_labels = predict( lastTask6T , class6_x_test );
        lastTask6T_TestAcc = sum(lastTask6T_labels==class6_y_test)/size(class6_y_test,1);
        newTrainAccRow(33)=lastTask6T_TrainAcc;
        newTestAccRow(33)=lastTask6T_TestAcc;
        
        setGlobalM(MF6out);
        lastTask6F = fitcknn(target6_x,target6_y,'NumNeighbors',i,'Distance',@transferLearn_GLOBAL);
        lastTask6F_TrainAcc=1-resubLoss(lastTask6F);
        lastTask6F_labels = predict( lastTask6F , class6_x_test );
        lastTask6F_TestAcc = sum(lastTask6F_labels==class6_y_test)/size(class6_y_test,1);
        newTrainAccRow(34)=lastTask6F_TrainAcc;
        newTestAccRow(34)=lastTask6F_TestAcc;
        
        setGlobalM(ME_transfer6);
        lastTask6E = fitcknn(target6_x,target6_y,'NumNeighbors',i);
        lastTask6E_TrainAcc=1-resubLoss(lastTask6E);
        lastTask6E_labels = predict( lastTask6E , class6_x_test );
        lastTask6E_TestAcc = sum(lastTask6E_labels==class6_y_test)/size(class6_y_test,1);
        newTrainAccRow(35)=lastTask6E_TrainAcc;
        newTestAccRow(35)=lastTask6E_TestAcc;
        
        testaccs=[testaccs; newTestAccRow];
        trainaccs=[trainaccs; newTrainAccRow];
        
    end
    save(['_run',int2str(ii),'data'])
    beep
end



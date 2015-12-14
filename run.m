load('image_segmentation_data2.mat')
[n,m] =  size(Xtrain_balanced);
%run with biggest possible step size
[M num_iter loss] = metric_trace(Xtrain_balanced,Ytrain_balanced,ones(n),10^-7,10000,10,0.001,zeros(m))
%run again after convergence with smaller stepsize
[M num_iter loss] = metric_trace(Xtrain_balanced,Ytrain_balanced,ones(n),10^-8,10000,10,0.001,M)
%run with biggest possible step size
[M num_iter loss] = metric_frob(Xtrain_balanced,Ytrain_balanced,ones(n),10^-7,10000,10,0.001,zeros(m))
%run again after convergence with smaller stepsize
[M num_iter loss] = metric_frob(Xtrain_balanced,Ytrain_balanced,ones(n),10^-8,10000,10,0.001,M)


%transfer learning. Example of staying close to identity (i.e. eucledian distance)
[Mtrans,k,loss] = metric_trace_transfer(pairwise_feature,pairwise_labels,ones(n),10^-13,10000,10,0.001,5,eye(m),M)

load('image_segmentation_data.mat')
[n,m] =  size(pairwise_feature);
%run with biggest possible step size
[M num_iter loss] = metric_trace(pairwise_feature,pairwise_labels,ones(n),10^-12,10000,10,0.001,zeros(m))
%run again after convergence with smaller stepsize
[M num_iter loss] = metric_trace(pairwise_feature,pairwise_labels,ones(n),10^-13,10000,10,0.001,M)
%run with biggest possible step size
[M num_iter loss] = metric_frob(pairwise_feature,pairwise_labels,ones(n),10^-12,10000,10,0.001,zeros(m))
%run again after convergence with smaller stepsize
[M num_iter loss] = metric_frob(pairwise_feature,pairwise_labels,ones(n),10^-13,10000,10,0.001,M)

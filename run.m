load('image_segmentation_data.mat')
[n,m] =  size(pairwise_feature);
[M num_iter] = metric_trace(pairwise_feature,pairwise_labels,ones(n),0.0001,10000,10,0.001)
[M num_iter] = metric_frob(pairwise_feature,pairwise_labels,ones(n),10^-14,10000,10,0.001)

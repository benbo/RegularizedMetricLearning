function [ pairwise_features , pairwise_labels ] = similarize_data( features , labels )

pairwise_features = [];
pairwise_labels = [];

for i=1:size(labels,1)-1
    disp(int2str(i))%Display progress
    for j=i+1:size(labels,1)
        
        similar = (double(labels(i)==labels(j)))*2-1;%Converting Logical to [-1,1]
        xi_minus_xj = abs(features(i,:)-features(j,:));
        
        pairwise_features = [ pairwise_features ; xi_minus_xj ];
        pairwise_labels = [ pairwise_labels ; similar ];
        
    end
 end

end


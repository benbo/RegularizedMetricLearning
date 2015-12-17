function [ hypothesis_accuracy ] = classify( X , Y , M , cutoff )

similarity = [];

for i=1:size(X,1)
    similarity = [similarity; X(i,:)*M*X(i,:)' ];
end


    similar_hypothesis = double(similarity < cutoff )*2-1;
    hypothesis_accuracy = sum(similar_hypothesis==Y)/size(Y,1);


end


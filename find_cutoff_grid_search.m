function [ best_cutoff ] = find_cutoff_grid_search( X , Y , M )

similarity = [];

for i=1:size(X,1)
    similarity = [similarity; X(i,:)*M*X(i,:)' ];
end

small = min(similarity);
large = max(similarity);

best_accuracy=0;
cutoffs = linspace(small,large,10000);
for i=1:10000
    similar_hypothesis = double(similarity < cutoffs(i))*2-1;
    hypothesis_accuracy = sum(similar_hypothesis==Y)/size(Y,1);
    if(hypothesis_accuracy > best_accuracy)
        best_cutoff = cutoffs(i);
        disp(['cutoff=',num2str(i),' acc=',num2str(hypothesis_accuracy)]);
        best_accuracy = hypothesis_accuracy;
    end
end


end


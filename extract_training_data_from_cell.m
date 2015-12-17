function [ features , labels ] = extract_training_data_from_cell( cell_dataset , label_column_number )

    cell_labels = [];
    labels = [];
    cell_features=[];
    
    % Want all but the label column
    feature_columns = [ 1:(label_column_number-1) , (label_column_number+1):(size(cell_dataset,2)) ];
    
    % If headers are present, skip them
    start = double(iscellstr( cell_dataset(1,:) ));
    
    % Separate label and features for each row of dataset
    for i=(1+start):size(cell_dataset,1)
        cell_labels = [ cell_labels ; cell_dataset(i,label_column_number) ];
        cell_features = [ cell_features ; cell_dataset(i,feature_columns) ];
    end
    
    % Assuming features are numerical, we can go directly to matrix form
    features = cell2mat( cell_features );
        
    % Labels likely will be text.  We will map them to integers 
    mapping_to_integers = unique( cell_labels );
    for i=1:size(cell_labels,1)
        index = find( strcmp(mapping_to_integers,cell_labels(i)) );
        labels = [ labels ; index-1 ];
    end
        
end


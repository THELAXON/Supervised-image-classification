% Clear the workspace and command window
clear
clc

% Load the dataset containing images of letters and their labels
load('dataset-letters.mat');

% Extract images and labels from the dataset
imgs = double(dataset.images);
letterLabels = dataset.labels;

% Reshape the images into a 3D array
images = reshape(imgs, 26000, 28, 28);

% Display 12 random images from the dataset along with their labels
figure(1);
for i = 1:12
    subplot(3, 4, i); 
    randomIndex = randi(size(images, 1));
    imshow(squeeze(images(randomIndex, :, :)));
    title(char(letterLabels(randomIndex) + 96)); % Convert label to character
end
sgtitle('Random Images from EMNIST Dataset'); 
saveas(gcf, 'sample_data.png'); % Save the figure

% Split the dataset into training and testing sets in a 50:50 ratio
randomIndices = randperm(size(images, 1));
splitIndex = floor(0.5 * length(randomIndices));
trainImages = imgs(randomIndices(1:splitIndex), :, :);
trainLabels = letterLabels(randomIndices(1:splitIndex));
testImages = imgs(randomIndices(splitIndex+1:end), :, :);
testLabels = letterLabels(randomIndices(splitIndex+1:end));

% Self-Implemented K-Nearest Neighbors (KNN) with Euclidean Distance
k = 1;
tepredict = categorical.empty(size(testImages, 1), 0);

% Loop through testing data to perform KNN with Euclidean distance
tic;% Start Timer for Euclidean distance
for i = 1:size(testImages, 1)
    % Calculate Euclidean distance of the current testing sample from all training samples
    comp1 = trainImages;
    comp2 = repmat(testImages(i, :), [size(trainImages, 1), 1]);
    l2 = sum((comp1 - comp2).^2, 2);
    
    % Get minimum k row indices
    [~, ind] = sort(l2);
    ind = ind(1:k);
    
    % Get labels for testing data
    labs = trainLabels(ind);
    tepredict(i, 1) = categorical(mode(labs));
end
% Calculate and display accuracy, and show confusion matrix
euclidean_predictions = sum(categorical(testLabels) == tepredict);
accuracy_euclidean = euclidean_predictions / size(testLabels, 1);
euclidean_time= toc;%Time elapsed for euclidean distance to be completed.
figure(2);% plot on a new a figure
confusionchart(categorical(testLabels), tepredict); % create a confusion chart for euclidean 
title(sprintf('Accuracy of Self-Implemented KNN Euclidean distance=%.2f', accuracy_euclidean));
disp(['Self-implemented KNN Euclidean Accuracy: ', num2str(accuracy_euclidean * 100), '%',' KNN Euclidean Elapsed Time: ', num2str(euclidean_time)]);

% K-Nearest Neighbors (KNN) with Cosine Similarity
tic;
for i = 1:size(testImages, 1)
    % Calculate cosine similarity of the current testing sample from all training samples
    comp1 = trainImages;
    comp2 = repmat(testImages(i, :), [size(trainImages, 1), 1]);
    
    % Calculate the cosine similarity
    dot_product = sum(comp1 .* comp2, 2);
    norm_comp1 = sqrt(sum(comp1.^2, 2));
    norm_comp2 = sqrt(sum(comp2.^2, 2));
    cosine_similarity = dot_product ./ (norm_comp1 .* norm_comp2);
    
    % Get maximum k row indices (nearest neighbors)
    [~, ind] = sort(cosine_similarity, 'descend');
    ind = ind(1:k);
    
    % Get labels for testing data
    labs = trainLabels(ind);
    tepredict(i, 1) = categorical(mode(labs));
end
% Calculate and display accuracy, and show confusion matrix
cosine_predictions = sum(categorical(testLabels) == tepredict);
accuracy_cosine = cosine_predictions / size(testLabels, 1);
cosine_time = toc;
figure(3);
confusionchart(categorical(testLabels), tepredict);
title(sprintf('Accuracy of Self-Implemented KNN cosine distance=%.2f', accuracy_cosine));
disp(['KNN Cosine Accuracy: ', num2str(accuracy_cosine * 100), '%',' KNN Cosine Elapsed Time: ', num2str(cosine_time)]);

% --- Support Vector Machine (SVM) Model ---
svm_model = fitcecoc(trainImages, trainLabels);
tic;
predictions_svm = predict(svm_model, testImages);
svm_test_time = toc; % End timer

% Calculate and display accuracy for SVM
accuracy_svm = sum(predictions_svm == testLabels) / numel(testLabels);
disp(['SVM Accuracy: ', num2str(accuracy_svm * 100), '%', ' SVM Elapsed Time: ', num2str(svm_test_time), ' seconds']);
figure(4);
confusionchart(testLabels,predictions_svm);
title(sprintf('Accuracy of svm=%.2f',accuracy_svm));

% Classification Decision Tree (fitctree)function
% Train the decision tree model
decision_tree_model = fitctree(trainImages, trainLabels);
tic;
% Predict using the trained tree model
predictions_Decision_tree = predict(decision_tree_model, testImages);
elapsed_time_decision_tree = toc; % Stop measuring time

% Calculate and display accuracy for Decision Tree models
accuracy_Decision_tree = sum(predictions_Decision_tree == testLabels) / numel(testLabels);
disp(['Decision Tree Accuracy: ', num2str(accuracy_Decision_tree * 100), '%', ' Decision Tree Elapsed Time: ', num2str(elapsed_time_decision_tree), ' seconds'])
figure(5);
confusionchart(testLabels,predictions_Decision_tree);
title(sprintf('Accuracy of Decision Tree=%.2f',accuracy_Decision_tree));

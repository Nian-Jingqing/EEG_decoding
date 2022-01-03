timepoint = 1;
channels = 63;
folds = 63;  % for cross-validation
epoch = 1;
events = 256;
  % for cross-validation
%% load data & attach label
a = ALLEEG.event;
len = 287;
label = zeros(1, 2);     
num_bin = 1;

for i = 1:len
    label_ = a(i).bini;
    
    if label_ == 1
        label(num_bin) = 1;
        num_bin = 1 + num_bin;
    end
 
    if label_ == 2
        label(num_bin) = 2;
        num_bin = 1 + num_bin;
    end

 end

%%

Y = label;
svm_model.labelBin = label'; % add to fm structure so it's saved

datas = ALLEEG.data;    % channel, timepoint, event (63, 500, 256)

accuracy = zeros(1, timepoint);

%% take channel PO8 as an example
tic
for time = 1:timepoint
    acc = 0;

    for fold = 1:folds
        X_ = datas(:, time, :);          % channel, event
        
        data = squeeze(X_);
        X = data.';
        test_size = events;

        % create training set and testing set
        X_test = X(:,fold);  % event, channel
        Y_test = Y;
        Y_train = Y;

        if fold == 1
            X_train = X(:,fold + 1:folds);     % event, channel
        elseif fold == folds
            X_train = X(:, 1:fold -1);          % event, channel
        else
            X_train = X(:, [1:fold - 1 fold + 1:folds]);
        end

        % training
        for j =1:folds-1
            svmMdl = fitcsvm(X_train(:,j), Y_train,'KernelFunction','gaussian','Standardize',true,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('Optimizer','gridsearch', 'UseParallel', true, 'ShowPlots', false ));
        end

        LabelPredicted = predict(svmMdl, X_test);

        ture_posivite = sum(Y_test(LabelPredicted == 2) == 2);
        ture_negnive = sum(Y_test(LabelPredicted == 1) == 1);

        acc = acc + (ture_posivite + ture_negnive) / test_size;
    end

    accuracy(time) = acc / folds;
    fprintf('Time Point: %d / %d', time, timepoint);

end
toc
disp(['Time: ',num2str(toc)]);

%%
times = zeros(1,200);
for j = 1:500
    times(j) = 2*j - 100;
end

figure(1)
avgfig = plot(timepoint, accuracy);
yline(0.5)
xlabel("Time")
ylabel("Accuracy")
title("Average accuacy across participants")
saveas(avgfig,'avgfig_1iter.png');



%% evaluate the model

LabelPredicted = predict(svmMdl, X_test);

ture_posivite = sum(Y_test(LabelPredicted == 2) == 2);
ture_negnive = sum(Y_test(LabelPredicted == 1) == 1);

accuracy = (ture_posivite + ture_negnive) / test_size;


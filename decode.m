timepoint = 3;
channels = 63;
folds = 63;
epoch = 1;
events = 256;
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
for time = 1:timepoint

    for fold = 1:folds
        acc = 0;
        X = datas(:, time, :);          % channel, event
        data = squeeze(X);
        test_size = events;

        % create training set and testing set
        X_test = X(fold, :);
        Y_test = Y(fold);

        if fold == 1
            X_train = X(fold + 1:folds,:);
            Y_train = Y(fold + 1:folds);
        elseif fold == folds
            X_train = X(1:fold - 1, :);
            Y_train = Y(1:fold - 1);
        else
            X_train = X([1:fold - 1 fold + 1:folds], :);
            Y_train = Y([1:fold - 1 fold + 1:folds]);
        end

        % training
        for j =1:epoch
            svmMdl = fitcsvm(X_train, Y_train,'KernelFunction','gaussian','Standardize',true,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('Optimizer','gridsearch', 'UseParallel', true, 'ShowPlots', false ));
        end

        LabelPredicted = predict(svmMdl, X_test);

        ture_posivite = sum(Y_test(LabelPredicted == 2) == 2);
        ture_negnive = sum(Y_test(LabelPredicted == 1) == 1);

        acc = acc + (ture_posivite + ture_negnive) / test_size;
    end

    accuracy(time) = acc / folds;
    fprintf('Time Point: %d / %d', time, timepoint);

end

%%
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


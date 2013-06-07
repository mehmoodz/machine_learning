clear ; close all; clc

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)


load('ex4data1.mat');
m = size(X, 1);
load('ex4weights.mat');

nn_params = [Theta1(:) ; Theta2(:)];
lambda = 1;


J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf('\nCost without regularization.\n %f',J);
pause;


lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf('\nCost with regularization.\n %f',J);
pause; 


g = sigmoidGradient([1 -0.5 0 0.5 1]);
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);


initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
checkNNGradients;
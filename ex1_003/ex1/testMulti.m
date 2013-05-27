 
%clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);
X = [ones(m, 1) X];
alpha = 0.01;
num_iters = 400;	

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);

computeCostMulti(X, y, theta);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

 

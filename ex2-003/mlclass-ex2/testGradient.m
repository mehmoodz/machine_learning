clear ; close all; clc
data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);
[m, n] = size(X);
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);
 [J,grad]=costFunction(initial_theta, X, y)
fprintf('\nProgram Executed.\n');
options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options)


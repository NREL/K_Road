load('inputs_results.mat');
Ts = 0.05;
Duration = size(inputs_results, 1)*Ts;

acceleration = inputs_results(:, 1);
steering = inputs_results(:, 2);

acceleration = acceleration';
steering = steering';
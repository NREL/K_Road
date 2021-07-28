% [reference] Autonomous Vehicle Trajectory Tracking via 
% Model-free Deep Reinforcement Learning
% Tripp, C., Aguasvivas Manzano, S.,
% Zhang, X., Lunacek, M., Graf, P. IEEE Transactions on 
% Intelligent Transportation Systems. Submitted for review
% [author] Aguasvivas Manzano, S. 
% Modified from:
% https://www.mathworks.com/help/mpc/ug/
%    lane-following-using-nonlinear-model-predictive-control.html
% The plant model was modified to be the Vehicle 3DOF Driving Model
% from the Automated Driving Toolbox found here:
% https://www.mathworks.com/help/vdynblks/ref/vehiclebody3dof.html
% [title] Helper script to initialize parameters and to declare the
% curvatures for the paths used in the study.

%% Parameters of Vehicle Dynamics and Road Curvature. 
% Specify the vehicle dynamics parameters
m = 2000;   % Mass of car
Iz = 4000;  % Moment of inertia about Z axis
lf = 1.4;   % Distance between Center of Gravity and Front axle 
lr = 1.6;   % Distance between Center of Gravity and Rear axle
Cf = 19000; % Cornering stiffness of the front tires (N/rad)
Cr = 33500; % Cornering stiffness of the rear tires (N/rad).
tau = 0.05;  % Time constant

%% Set the initial and driver-set velocities.
v0 = 17;    % Initial velocity
v_set = 17; % Driver set velocity
Vx= 17;

%% Set the controller sample time.
Ts = 0.05;
%% Obtain the lane curvature information.

[rho, t, Duration, v_seq_1, Xref, Yref, phi0, xref, yref] = FigureEightGetCurvature(v_set, v0, Ts);
% [rho, t, Duration, v_seq_1, Xref, Yref, phi0, xref, yref] = PulsedSpeedPathGetCurvature(Ts); % Signal containing curvature information
% [rho, t, v_seq_1, Duration, Xref, Yref, phi0, xref, yref] = HairpinGetCurvature(Ts);
% [rho, t, Xref, Yref, v_seq_1, Duration, phi0, xref, yref] = FalconeDoubleLaneChangeGetCurvature(Vx, Ts);
% [rho, t, Xref, Yref, v_seq_1, Duration, phi0, xref, yref] = DoubleLaneChangeGetCurvature(Vx, Ts);
% [rho, t, v_seq_1, Duration, Xref, Yref, phi0] = CurriculumAngledGetCurvature(Ts);

%% Assign some workspace parameters to be queried by Simulink
v_seq = v_seq_1.signals.values;
v_set = v_seq_1.signals.values(1);
v0 = v_seq_1.signals.values(1);
Vx = v_seq_1.signals.values(1);
r_data = [Xref', Yref'];
r_data = repmat(r_data, [1, 1, 2]);
ref_data.time = [t(1), t(end)];
ref_data.signals.values = r_data;
ref_data.signals.dimensions = [size(Xref, 2), 2];
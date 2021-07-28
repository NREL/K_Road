% [reference] Autonomous Vehicle Trajectory Tracking via 
% Model-free Deep Reinforcement Learning
% Tripp, C., Aguasvivas Manzano, S.,
% Zhang, X., Lunacek, M., Graf, P. IEEE Transactions on 
% Intelligent Transportation Systems. Submitted for review.
% [title] Benchmarking KRoad using MATLAB Vehicle 3DOF model
% [author] Aguasvivas Manzano, S. 
% Modified from:
% https://www.mathworks.com/help/mpc/ug/
%    lane-following-using-nonlinear-model-predictive-control.html
% The plant model was modified to be the Vehicle 3DOF Driving Model
% from the Automated Driving Toolbox found here:
% https://www.mathworks.com/help/vdynblks/ref/vehiclebody3dof.html
clear all;
LaneFollowingUsingNMPCData;
logs = [];
ct_times = [];
for i = 3 %[10:25, 75, 100]
    mdl = 'LaneFollowingNMPC';
    load_system(mdl)
    N_la = i;
    N_c = 2;                                                                                                                      ;
    nlobj = nlmpc(7,3,'MV',[1 2],'MD',3,'UD',4);
    nlobj.Ts = Ts;
    nlobj.PredictionHorizon = i;
    nlobj.ControlHorizon = N_c;
    nlobj.Model.StateFcn = @(x,u) LaneFollowingStateFcn(x,u);
    nlobj.Jacobian.StateFcn = @(x,u) LaneFollowingStateJacFcn(x,u);
    nlobj.Model.OutputFcn = @(x,u) [x(3);x(5);x(6)+x(7)];
    nlobj.Jacobian.OutputFcn = @(x,u) [0 0 1 0 0 0 0;0 0 0 0 1 0 0; ...
                                                    0 0 0 0 0 1 1];
    % Set the constraints for manipulated variables.
    nlobj.MV(1).Min = -26.5;           % Maximum acceleration 3 m/s^2
    nlobj.MV(1).Max = 26.5;            % Minimum acceleration -3 m/s^2
    nlobj.MV(2).Min = -deg2rad(60);    % Minimum steering angle -65 
    nlobj.MV(2).Max = deg2rad(60);     % Maximum steering angle 65

    nlobj.OV(1).ScaleFactor = 15;      % Typical value of longitudinal velocity
    nlobj.OV(2).ScaleFactor = 1.;      % Range for lateral deviation
    nlobj.OV(3).ScaleFactor = 1.;      % Range for relative yaw angle
    nlobj.MV(1).ScaleFactor = 6.;      % Range of steering angle
    nlobj.MV(2).ScaleFactor = 1.;      % Range of acceleration
    nlobj.MD(1).ScaleFactor = 1.;      % Range of Curvature
    nlobj.Weights.OutputVariables = [1 1 1];
    nlobj.Weights.ManipulatedVariablesRate = [0.1, 0.3]; %[0.1*(i/10) 0.1*(i/10)];
    x0 = [0.0 0.0 v0 0.0 0.0 0.0 0.1];
    u0 = [0. 0.];
    ref0 = [22 0 0];
    md0 = 0.1;
    validateFcns(nlobj,x0,u0,md0,{},ref0);
    % running NLMPC:
    tic
    controller_type = 1;
    sim(mdl)
    logsout1 = logsout;
    elapsed_time_nl = toc;
    % running ALMPC:
    tic
    controller_type = 2;
    sim(mdl)
    logsout2 = logsout;
    elapsed_time_linear = toc;
    mean_linear = mean(logsout2.getElement('Lateral Deviation') ...
                                    .Values.Data);
    mean_nonlinear = mean(logsout1.getElement('Lateral Deviation').Values.Data);
    logs = [logs; [logsout1, logsout1]];
    ct_times = [ct_times ; [elapsed_time_nl , ...
                                elapsed_time_linear]];
    controller_type = 1;
    bdclose(mdl)
end




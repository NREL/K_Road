function [rho,time, Xref, Yref, v_seq, Duration, phi0, xref, yref] = DoubleLaneChangeGetCurvature(Vx, Ts)
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
    % [title] Helper script declare the curvature and velocity profiles 
    % of the path defined by the title of this function

    % Copyright 2018 The MathWorks, Inc.
    Xref = 0.0:0.05:120;

    % Desired Y position
    z1 = (2.4/25)*(Xref-27.19)-1.2;
    z2 = (2.4/21.95)*(Xref-56.46)-1.2;
    Yref = 4.05/2*(1+tanh(z1)) - 5.7/2.*(1+tanh(z2));

    % Desired curvature
    DX = gradient(Xref, Ts);
    DY = gradient(Yref, Ts);
    D2Y = gradient(DY, Ts);
    D2X = gradient(DX, Ts);

    curvature = (DX.* D2Y - DY.* D2X) ...
                                ./(DX.^2+DY.^2).^(3/2);

    v_s = repelem(Vx, size(Yref, 2));
    Duration = size(Xref, 2);

    time = linspace(0., Xref(end)/Vx, Duration);
    %time = cumsum(sqrt(DX.^2 + DY.^2))./Vx; %linspace(0, length / v0, Duration);
    %time(1) = 0;

    % Stored curvature (as input for LKA)
    rho.time = time;
    rho.signals.values = curvature';
    yref.time = time;
    yref.signals.values = Yref';
    xref.time = time;
    xref.signals.values = Xref';

    v_seq.time = time;
    v_seq.signals.values = v_s';
    phi0 = 0.0;

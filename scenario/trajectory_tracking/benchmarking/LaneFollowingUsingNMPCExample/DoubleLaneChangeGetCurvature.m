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
    % lateral deviation calculated using: 
    % https://publications.lib.chalmers.se/records/fulltext/219367/219367.pdf

    % Copyright 2018 The MathWorks, Inc.
    time = 0:Ts:10;

    Xref = Vx*time;
    % Desired Y position
    z1 = (2.4/50)*(Xref-27.19)-1.2;
    z2 = (2.4/43.9)*(Xref-56.46)-1.2;
    Yref = 8.1/2*(1+tanh(z1)) - 11.4/2.*(1+tanh(z2));

    % Desired curvature
    DX = gradient(Xref);
    DY = gradient(Yref);
    D2Y = gradient(DY);
    D2X = gradient(DX);

    curvature = (DX.* D2Y - DY.* D2X) ...
                                ./(DX.^2+DY.^2).^(3/2);

    v_s = repelem(Vx, size(Yref, 2));
    Duration = size(time, 2);
    time = cumsum(sqrt(DX.^2+DY.^2))./v_s; %linspace(0., Xref(end)/Vx - Ts, Duration);
    t_1 = time(1);
    time = time - t_1;

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


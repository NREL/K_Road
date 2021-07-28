function [rho, t, v, Duration, Xref, Yref, phi0] = CurriculumAngledGetCurvature(Ts)
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
    Xref = csvread('Xref.csv')'; % load from data
    Yref = csvread('Yref.csv')'; % load from data
    v = csvread('vref.csv')'; % load from data

    x_initial = linspace(Xref(1, 1), Xref(1, 2), 10);
    y_initial = linspace(Yref(1, 1), Yref(1, 2), 10);
    v_initial = repelem(v(1, 1), 10);
    
    Xref = [x_initial, Xref(1, 2:end)];
   	Yref = [y_initial, Yref(1, 2:end)];
    v = [v_initial, v(1, 2:end)];
    
    % Desired curvature
    DX = gradient(Xref,0.01);
    DY = gradient(Yref,0.01);
    D2Y = gradient(DY,0.01);
    curvature = DX.*D2Y./(DX.^2+DY.^2).^(3/2);
    curvature(isnan(curvature)) = 0;
    
    Duration = size(Xref, 2);
    t = linspace(0, Duration, Duration); 
    
    % Stored curvature (as input for LKA)
    rho.Xref = Xref;
    rho.YRef = Yref;
    rho.time = t;
    rho.signals.values = curvature';
    phi0 = 0;
end

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
% [title] Helper script to plot the traversed path compared to the
% target path.
figure()
nonlinear_info = logsout1.getElement("Info");
linear_info = logsout2.getElement("Info");
reference_x = logsout2.getElement("xref_out");
reference_y = logsout2.getElement("yref_out");
plot(nonlinear_info.Values.InertFrm.Cg.Disp.X.Data, nonlinear_info.Values.InertFrm.Cg.Disp.Y.Data)

hold on;
plot(linear_info.Values.InertFrm.Cg.Disp.X.Data, linear_info.Values.InertFrm.Cg.Disp.Y.Data)
plot(reference_x.Values.Data, reference_y.Values.Data, 'k');

legend("X", "Y");
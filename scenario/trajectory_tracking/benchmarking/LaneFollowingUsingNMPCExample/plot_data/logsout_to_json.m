function structure = logsout_to_json(logsout, target_file_name)

structure = struct();

    for iprop = 1:logsout.numElements
        
        Name = logsout{iprop}.Name;
        Name = regexprep(Name, ' ', '_');
        if isempty(Name)
            Name = "property" + num2str(iprop);
            Name
        end
   
        try  
            structure.(Name).data = logsout{iprop}.Values.Data;
            structure.(Name).time = logsout{iprop}.Values.Time;    
        catch
          %  structure.(Name) = logsout{iprop}.Values;
            structure.(Name).InertialFrame.Cg.Displacement.X = logsout{iprop}.Values.InertFrm.Cg.Disp.X.Data;
            structure.(Name).InertialFrame.Cg.Displacement.Y = logsout{iprop}.Values.InertFrm.Cg.Disp.Y.Data;
            structure.(Name).InertialFrame.Cg.Vel.Xdot = logsout{iprop}.Values.InertFrm.Cg.Vel.Xdot.Data;
            structure.(Name).InertialFrame.Cg.Vel.Ydot = logsout{iprop}.Values.InertFrm.Cg.Vel.Ydot.Data;
            structure.(Name).InertialFrame.Cg.Ang.psi = logsout{iprop}.Values.InertFrm.Cg.Ang.psi.Data;
            
            structure.(Name).InertialFrame.FrontAnxle.Displacement.X = logsout{iprop}.Values.InertFrm.FrntAxl.Disp.X.Data;
            structure.(Name).InertialFrame.FrontAnxle.Displacement.Y = logsout{iprop}.Values.InertFrm.FrntAxl.Disp.Y.Data;
            structure.(Name).InertialFrame.FrontAnxle.Vel.Xdot = logsout{iprop}.Values.InertFrm.FrntAxl.Vel.Xdot.Data;
            structure.(Name).InertialFrame.FrontAnxle.Vel.Ydot = logsout{iprop}.Values.InertFrm.FrntAxl.Vel.Ydot.Data;
            
            structure.(Name).InertialFrame.RearAnxle.Displacement.X = logsout{iprop}.Values.InertFrm.RearAxl.Disp.X.Data;
            structure.(Name).InertialFrame.RearAnxle.Displacement.Y = logsout{iprop}.Values.InertFrm.RearAxl.Disp.Y.Data;
            structure.(Name).InertialFrame.RearAnxle.Vel.Xdot = logsout{iprop}.Values.InertFrm.RearAxl.Vel.Xdot.Data;
            structure.(Name).InertialFrame.RearAnxle.Vel.Ydot = logsout{iprop}.Values.InertFrm.RearAxl.Vel.Ydot.Data;
            
            structure.(Name).InertialFrame.Hitch.Displacement.X = logsout{iprop}.Values.InertFrm.RearAxl.Disp.X.Data;
            structure.(Name).InertialFrame.Hitch.Displacement.Y = logsout{iprop}.Values.InertFrm.RearAxl.Disp.Y.Data;
            structure.(Name).InertialFrame.Hitch.Vel.Xdot = logsout{iprop}.Values.InertFrm.RearAxl.Vel.Xdot.Data;
            structure.(Name).InertialFrame.Hitch.Vel.Ydot = logsout{iprop}.Values.InertFrm.RearAxl.Vel.Ydot.Data;
            
            structure.(Name).InertialFrame.Geom.Displacement.X = logsout{iprop}.Values.InertFrm.Geom.Disp.X.Data;
            structure.(Name).InertialFrame.Geom.Displacement.Y = logsout{iprop}.Values.InertFrm.Geom.Disp.Y.Data;
            structure.(Name).InertialFrame.Geom.Vel.Xdot = logsout{iprop}.Values.InertFrm.Geom.Vel.Xdot.Data;
            structure.(Name).InertialFrame.Geom.Vel.Ydot = logsout{iprop}.Values.InertFrm.Geom.Vel.Ydot.Data;
            
            structure.(Name).BodyFrame.Cg.Vel.xdot = logsout{iprop}.Values.BdyFrm.Cg.Vel.xdot.Data;
            structure.(Name).BodyFrame.Cg.Vel.ydot = logsout{iprop}.Values.BdyFrm.Cg.Vel.ydot.Data;
            structure.(Name).BodyFrame.Cg.Ang.beta = logsout{iprop}.Values.BdyFrm.Cg.Ang.Beta.Data;
            structure.(Name).BodyFrame.Cg.AngVel.p = logsout{iprop}.Values.BdyFrm.Cg.AngVel.p.Data;
            structure.(Name).BodyFrame.Cg.AngVel.q = logsout{iprop}.Values.BdyFrm.Cg.AngVel.q.Data;
            structure.(Name).BodyFrame.Cg.AngVel.r = logsout{iprop}.Values.BdyFrm.Cg.AngVel.r.Data;
            
            structure.(Name).BodyFrame.Cg.Acc.ax = logsout{iprop}.Values.BdyFrm.Cg.Acc.ax.Data;
            structure.(Name).BodyFrame.Cg.Acc.ay = logsout{iprop}.Values.BdyFrm.Cg.Acc.ay.Data;
            structure.(Name).BodyFrame.Cg.Acc.xddot = logsout{iprop}.Values.BdyFrm.Cg.Acc.xddot.Data;
            structure.(Name).BodyFrame.Cg.Acc.yddot = logsout{iprop}.Values.BdyFrm.Cg.Acc.yddot.Data;
            
            structure.(Name).BodyFrame.Cg.AngAcc.pdot = logsout{iprop}.Values.BdyFrm.Cg.AngAcc.pdot.Data;
            structure.(Name).BodyFrame.Cg.AngAcc.qdot = logsout{iprop}.Values.BdyFrm.Cg.AngAcc.qdot.Data;
            structure.(Name).BodyFrame.Cg.AngAcc.rdot = logsout{iprop}.Values.BdyFrm.Cg.AngAcc.rdot.Data;
            
            
            structure.(Name).BodyFrame.FrntAxl.Disp.x = logsout{iprop}.Values.BdyFrm.FrntAxl.Disp.x.Data;
            structure.(Name).BodyFrame.FrntAxl.Disp.y = logsout{iprop}.Values.BdyFrm.FrntAxl.Disp.y.Data;
            
            structure.(Name).BodyFrame.FrntAxl.Vel.xdot = logsout{iprop}.Values.BdyFrm.FrntAxl.Vel.xdot.Data;
            structure.(Name).BodyFrame.FrntAxl.Vel.ydot = logsout{iprop}.Values.BdyFrm.FrntAxl.Vel.ydot.Data;
            
            structure.(Name).BodyFrame.FrntAxl.Steer.rl = logsout{iprop}.Values.BdyFrm.FrntAxl.Steer.WhlAngFL.Data;
            structure.(Name).BodyFrame.FrntAxl.Steer.rr = logsout{iprop}.Values.BdyFrm.FrntAxl.Steer.WhlAngFR.Data;
          end
        
    end
    
    json_encode = jsonencode(structure);
    fid = fopen(target_file_name,'wt');
    fprintf(fid, json_encode);
    fclose(fid);

end
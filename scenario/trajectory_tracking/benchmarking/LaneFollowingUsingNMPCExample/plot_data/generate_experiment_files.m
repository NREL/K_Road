function generate_experiment_files(exp_name, nl_log, l_log)

    linear_name = exp_name + "_linear.json";
    nl_name = exp_name + "_nl.json";
    
    logsout_to_json(l_log, linear_name);
    logsout_to_json(nl_log, nl_name);

end
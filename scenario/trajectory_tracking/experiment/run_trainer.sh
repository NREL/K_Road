#!/bin/bash

sbatch -J "train_test" -t 120 ../../../../../../hpc/srunpy.sh path_following_trainer.py env_config.mode {name:train_test}

#w=10
#for i in {1..20}
#do
#  name="${w}_${i}_d_waypoint"
#  echo sbatch -J $name -t 60 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name $name env_config.num_waypoints $w env_config.use_distances True
#  sbatch -J $name -t 60 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name $name env_config.num_waypoints $w env_config.use_distances True
#done
#
#w=10
#for i in {1..20}
#do
#  name="${w}_${i}_waypoint"
#  echo sbatch -J $name -t 60 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name $name env_config.num_waypoints $w
#  sbatch -J $name -t 60 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name $name env_config.num_waypoints $w
#done

#name="10_0_waypoint_no_distances"
#echo sbatch -J $name -t 60 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name $name env_config.num_waypoints 10

#trajectory_tracking_test_standard.py env_config.mode train env_config.name tftest env_config.num_waypoints 10

#waypoints=(1 2 5 7 10)
##waypoints=(7 15 25 30)
#
#for w in "${waypoints[@]}"
#do
#  for i in {1..5}
#  do
#    name="${w}_${i}_waypoint_no_distances"
#    echo sbatch -J $name -t 60 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name $name env_config.num_waypoints $w
#    sbatch -J $name -t 60 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name $name env_config.num_waypoints $w
##    sbatch -J $name -t 30 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name $name env_config.num_waypoints $w
##    name="${w}_${i}_waypoint_use_distances"
##    echo sbatch -J $name -t 60 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name $name env_config.num_waypoints $w env_config.use_distances True
##    sbatch -J $name -t 60 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name $name env_config.num_waypoints $w env_config.use_distances True
#  done
#done

#w=10
#for i in {1..10}
#do
#  name="${w}_${i}_use_velocity_reference_angle"
#  echo sbatch -J $name -t 60 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name $name env_config.num_waypoints $w env_config.use_velocity_reference_angle True
#  sbatch -J $name -t 60 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name $name env_config.num_waypoints $w env_config.use_velocity_reference_angle True
#  name="${w}_${i}_use_alternate_reference_angle"
#  echo sbatch -J $name -t 60 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name $name env_config.num_waypoints $w env_config.use_alternate_reference_angle True
#  sbatch -J $name -t 60 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name $name env_config.num_waypoints $w env_config.use_alternate_reference_angle True
#  name="${w}_${i}_use_distances"
#  echo sbatch -J $name -t 60 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name $name env_config.num_waypoints $w env_config.use_distances True
#  sbatch -J $name -t 60 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name $name env_config.num_waypoints $w env_config.use_distances True
#done
#
#scaling=(0.5 1.2 1.5 2)
#for s in "${scaling[@]}"
#do
#  for i in {1..10}
#  do
#    name="${w}_${i}_scaling_${s}"
#    echo sbatch -J $name -t 60 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name $name env_config.num_waypoints $w env_config.use_velocity_reference_angle True
#    sbatch -J $name -t 60 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name $name env_config.num_waypoints $w env_config.use_velocity_reference_angle True
#  done
#done

#sbatch -J 1_waypoint -t 90 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name 1_waypoint env_config.num_waypoints 1
#sbatch -J 2_waypoint -t 90 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name 2_waypoint env_config.num_waypoints 2
#sbatch -J 3_waypoint -t 90 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name 3_waypoint env_config.num_waypoints 3
#sbatch -J 4_waypoint -t 90 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name 4_waypoint env_config.num_waypoints 4
##sbatch -J 5_waypoint -t 90 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name 5_waypoint env_config.num_waypoints 5
#sbatch -J 6_waypoint -t 90 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name 6_waypoint env_config.num_waypoints 6
#sbatch -J 7_waypoint -t 90 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name 7_waypoint env_config.num_waypoints 7
#sbatch -J 8_waypoint -t 90 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name 8_waypoint env_config.num_waypoints 8
#
#
#sbatch -J 5_waypoint_0_5_spacing -t 90 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name 5_waypoint_0_5_spacing env_config.waypoint_spacing .5
#sbatch -J 5_waypoint_0_9_spacing -t 90 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name 5_waypoint_0_9_spacing env_config.waypoint_spacing .9
#sbatch -J 5_waypoint_1_0_spacing -t 90 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name 5_waypoint_1_0_spacing env_config.waypoint_spacing 1.0
#sbatch -J 5_waypoint_1_1_spacing -t 90 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name 5_waypoint_1_1_spacing env_config.waypoint_spacing 1.1
#sbatch -J 5_waypoint_1_5_spacing -t 90 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name 5_waypoint_1_5_spacing env_config.waypoint_spacing 1.5
#sbatch -J 5_waypoint_2_spacing -t 90 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name 5_waypoint_2_spacing env_config.waypoint_spacing 2.0
#sbatch -J 5_waypoint_3_spacing -t 90 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name 5_waypoint_3_spacing env_config.waypoint_spacing 3.0
#sbatch -J 5_waypoint_4_spacing -t 90 ../../../../../../hpc/srunpy.sh trajectory_tracking_test_standard.py env_config.mode train env_config.name 5_waypoint_4_spacing env_config.waypoint_spacing 4.0
#
#
#

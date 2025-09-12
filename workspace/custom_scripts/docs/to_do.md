## task 1

- [ ] record joint_torques, joint_positions, ef_pose, actions (raw, preprocessed), ctrl_target_fingertip_midpoint_pos, fixed_pos_obs_frame, fingertip_midpoint_pos

- [ ] ik controller with gravity disabled (ef_delta_pos --> dof_delta_pos --> pid at each joint to get the dof torque)

- [ ] go with a better visualization 
flexibilty of createing a new pannel (graphs specific to different values) for different visualizations of the values, animating the video and the values


- 
- [x] make sure the frames are all consistent (actions frame--> delta pose frame --> end effector frame)
- [x] you get target joint angles from the diff ik 
- [ ] assuming that the orientation of the base frame and the env origin frame are same 
- [ ] record the action values, initial state of the robot sent via the keyboard and send it to both the robots
- [ ] kp, kd values of the 
- [ ] change the sliders and values such that if they have any mutual relationship among them 
- [ ] change the slider values with the code 
- [ ] log the values, and video also consider a better visualization and logging tools

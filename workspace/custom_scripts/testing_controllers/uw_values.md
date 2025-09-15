#### diff ik 

ef position reset
 tensor([1.6346, 0.0061, 0.0697], device='cuda:0')
ef velocity reset 
 tensor([0., 0., 0.], device='cuda:0')
ef position after no action step 
 tensor([ 1.6353, -0.0045,  0.0705], device='cuda:0')
ef velocity after no action step 
 tensor([ 0.0813, -0.7902,  0.0398], device='cuda:0')

#### task space controller env 

ef position reset
 tensor([1.6346, 0.0061, 0.0697], device='cuda:0')
ef velocity reset 
 tensor([0., 0., 0.], device='cuda:0')
ef position after no action step 
 tensor([ 1.6345, -0.0025,  0.0702], device='cuda:0')
ef velocity after no action step 
 tensor([-0.0075, -0.6341,  0.0195], device='cuda:0')

#### controller env 

pos error 
 [0. 0. 0.]
axis angle error 
 [-4.6188831e-08  1.5705653e-07  1.8106340e-07]
task prop gains 
 [100 100 100  30  30  30]
task deriv gains 
 [20.       20.       20.       10.954452 10.954452 10.954452]
lin error 
 [0. 0. 0.]
rot error 
 [-4.6188831e-08  1.5705653e-07  1.8106340e-07]
fingertip_midpoint_linvel 
 [-0.18572329 -0.36718544  0.05522489]
fingertip_midpoint_angvel 
 [-3.2309709  1.8546141  2.1741521]
dof_torque 
 [-19.829264   -4.0885296 -25.602413   18.885489   34.548378   12.523081
  35.358242    0.          0.       ]
task_wrench  
 [  3.7144659   7.343709   -1.1044978  35.393513  -20.316277  -23.816639 ]
pos error 
 [0. 0. 0.]
axis angle error 
 [-1.2199163e-07 -3.3205396e-08 -2.0880897e-08]
task prop gains 
 [100 100 100  30  30  30]
task deriv gains 
 [20.       20.       20.       10.954452 10.954452 10.954452]
lin error 
 [0. 0. 0.]
rot error 
 [-1.2199163e-07 -3.3205396e-08 -2.0880897e-08]
fingertip_midpoint_linvel 
 [ 0.23311614  0.41192135 -0.04640579]
fingertip_midpoint_angvel 
 [ 2.9600215 -1.8716605 -2.211475 ]


dof_torque 
 [ 19.57876    4.673045  25.370043 -19.066309 -31.778526 -13.345794
 -35.18519    0.         0.      ]
task_wrench  
 [ -4.6623225  -8.238427    0.9281158 -32.425415   20.503012   24.225494 ]





## points to note 
- reset gains and rot deriv scale 
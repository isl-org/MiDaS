source ~/catkin_ws/devel/setup.bash
roslaunch midas_cpp midas_cpp.launch model_name:="model_opt.pt" input_topic:="image_topic" output_topic:="midas_topic" out_orig_size:="true"
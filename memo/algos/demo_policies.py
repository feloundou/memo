
circle_policy = [-1, 0.25]
forward_policy = [-1, 0]
spinning_top_policy = [0, -1]
left_policy = [0, 1]

###
# Go forward, spin and continue
forward_times = [forward_policy] * 100
left_times = [left_policy] * 100
action_mid = forward_times + left_times
action_new = action_mid * 5
forward_spin_policy = action_new
###

## go forward, turn around, go forward, turn around
forward_times = [forward_policy] * 50
left_times = [left_policy] * 50
action_mid = forward_times + left_times
action_new = action_mid * 10
back_forth_policy = action_new


## square
forward_times = [forward_policy] * 75
left_times = [left_policy] * 25
action_mid = forward_times + left_times
action_new = action_mid * 10
square_policy = action_new

import numpy as np
import gym
from utils import *


MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

envs = ['./envs/doorkey-5x5-normal.env',
        './envs/doorkey-6x6-normal.env',
        './envs/doorkey-8x8-normal.env',

        './envs/doorkey-6x6-direct.env',
        './envs/doorkey-8x8-direct.env',

        './envs/doorkey-6x6-shortcut.env',
        './envs/doorkey-8x8-shortcut.env']

def motion_model_cost(env,info,state,action):
    next_state = state.copy()
    x = next_state[0]
    y = next_state[1]
    dir = next_state[2]
    key = next_state[3]
    door = next_state[4]
    if action == 0:
        if dir == 0:
            next_state[1] = next_state[1]-1
        elif dir == 1:
            next_state[0] = next_state[0] + 1
        elif dir == 2:
            next_state[0] = next_state[0] - 1
        elif dir == 3:
            next_state[1] = next_state[1] + 1
        cell = env.grid.get(next_state[0], next_state[1])
        if isinstance(cell, gym_minigrid.minigrid.Wall):
            next_state = None
            cost = np.inf
        elif isinstance(cell, gym_minigrid.minigrid.Door):
            if door == 0:
                next_state = None
                cost = np.inf
            else:
                cost = 1
        else:
            cost = 1

    if action == 1:
        if dir == 0:
            next_state[2] = 2
        elif dir == 1:
            next_state[2] = 0
        elif dir == 2:
            next_state[2] = 3
        elif dir == 3:
            next_state[2] = 1
        cost = 1

    if action == 2:
        if dir == 0:
            next_state[2] = 1
        elif dir == 1:
            next_state[2] = 3
        elif dir == 2:
            next_state[2] = 0
        elif dir == 3:
            next_state[2] = 2
        cost = 1

    if action == 3:
        if next_state[3] == 1:
            return None, np.inf
        front_x = next_state[0]
        front_y = next_state[1]
        if dir == 0:
            front_y = front_y - 1
        elif dir == 1:
            front_x = front_x + 1
        elif dir == 2:
            front_x = front_x - 1
        elif dir == 3:
            front_y = front_y + 1
        cell = env.grid.get(front_x, front_y)
        if isinstance(cell, gym_minigrid.minigrid.Key):
            next_state[3] = 1
            cost = 1
        else:
            next_state = None
            cost = np.inf

    if action == 4:
        if next_state[3] == 0 or door == 1:
            return None, np.inf
        front_x = next_state[0]
        front_y = next_state[1]
        if dir == 0:
            front_y = front_y - 1
        elif dir == 1:
            front_x = front_x + 1
        elif dir == 2:
            front_x = front_x - 1
        elif dir == 3:
            front_y = front_y + 1
        cell = env.grid.get(front_x, front_y)
        if isinstance(cell, gym_minigrid.minigrid.Door):
            info['door_open'] = True
            next_state[4] = 1
            cost = 1
        else:
            next_state = None
            cost = np.inf

    return next_state, cost

def partb_motion_model_cost(env,state,action):
    # state = [x, y, dir, key, d1, d2, kp, gp]
    next_state = state.copy()
    x = next_state[0]
    y = next_state[1]
    dir = next_state[2]
    key = next_state[3]
    d1 = next_state[4]
    d2 = next_state[5]
    kp = next_state[6]
    gp = next_state[7]
    cost = 0

    if action == 0:
        if dir == 0:
            next_state[1] = next_state[1]-1
        elif dir == 1:
            next_state[0] = next_state[0] + 1
        elif dir == 2:
            next_state[0] = next_state[0] - 1
        elif dir == 3:
            next_state[1] = next_state[1] + 1
        cell = env.grid.get(next_state[0], next_state[1])
        if isinstance(cell, gym_minigrid.minigrid.Wall):
            next_state = None
            cost = np.inf
        elif [next_state[0], next_state[1]] == [4,2]:
            if d1 == 0:
                next_state = None
                cost = np.inf
            else:
                cost = 1
        elif [next_state[0], next_state[1]] == [4,5]:
            if d2 == 0:
                next_state = None
                cost = np.inf
            else:
                cost = 1
        else:
            cost = 1

    if action == 1:
        if dir == 0:
            next_state[2] = 2
        elif dir == 1:
            next_state[2] = 0
        elif dir == 2:
            next_state[2] = 3
        elif dir == 3:
            next_state[2] = 1
        cost = 1

    if action == 2:
        if dir == 0:
            next_state[2] = 1
        elif dir == 1:
            next_state[2] = 3
        elif dir == 2:
            next_state[2] = 0
        elif dir == 3:
            next_state[2] = 2
        cost = 1

    if action == 3:
        if next_state[3] == 1:
            return None, np.inf
        front_x = next_state[0]
        front_y = next_state[1]
        if dir == 0:
            front_y = front_y - 1
        elif dir == 1:
            front_x = front_x + 1
        elif dir == 2:
            front_x = front_x - 1
        elif dir == 3:
            front_y = front_y + 1
        cell = env.grid.get(front_x, front_y)
        if isinstance(cell, gym_minigrid.minigrid.Key):
            next_state[3] = 1
            cost = 1
        else:
            next_state = None
            cost = np.inf

    if action == 4:
        if next_state[3] == 0:
            return None, np.inf
        elif d1 == 1 and d2 ==1:
            return None, np.inf
        front_x = next_state[0]
        front_y = next_state[1]
        if dir == 0:
            front_y = front_y - 1
        elif dir == 1:
            front_x = front_x + 1
        elif dir == 2:
            front_x = front_x - 1
        elif dir == 3:
            front_y = front_y + 1

        if [front_x, front_y]==[4,2]:
            next_state[4] = 1
            cost = 1
        elif [front_x, front_y]==[4,5]:
            next_state[5] = 1
            cost = 1
        else:
            next_state = None
            cost = np.inf

    return next_state, cost


def doorkey_problem(env,info):
    '''
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env
        
        doorkey-6x6-direct.env
        doorkey-8x8-direct.env
        
        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env
        
    Feel Free to modify this fuction
    '''
    optim_act_seq = []
    # Visualize the environment
    plot_env(env)

    # Get the agent position
    agent_pos = env.agent_pos
    x = agent_pos[0]
    y = agent_pos[1]

    # Get the agent direction
    agent_dir = info['init_agent_dir']
    if all(agent_dir == np.array([0, -1])):
        dir = 0
    elif all(agent_dir == np.array([1, 0])):
        dir = 1
    elif all(agent_dir == np.array([-1, 0])):
        dir = 2
    elif all(agent_dir == np.array([0, 1])):
        dir = 3

    is_carrying = env.carrying is not None
    if is_carrying:
        key = 1
    else:
        key = 0

    door = 0

    state = [x,y,dir,key,door]

    key_pos = info['key_pos']
    goal_pos = info['goal_pos']
    x_size = info['width']-2
    y_size = info['height']-2

    T = x_size*y_size*4*2*2-1
    par_control = np.empty((x_size,y_size,4,2,2),dtype=object)
    value_list = []
    value_mat0 = np.ones((x_size,y_size,4,2,2))*np.inf
    value_mat0[x-1,y-1,dir,key,door] = 0
    value_list.append(value_mat0)
    value_mat1 = np.array(value_mat0, copy=True)
    for u in range(5):
        next_state, cost = motion_model_cost(env,info,state,u)
        if next_state is not None:
            if cost<value_mat1[next_state[0]-1,next_state[1]-1,next_state[2],next_state[3],next_state[4]]:
                value_mat1[next_state[0]-1, next_state[1]-1, next_state[2], next_state[3],next_state[4]] = cost
                par_control[next_state[0]-1, next_state[1]-1, next_state[2], next_state[3],next_state[4]] = [state,u]

    value_list.append(value_mat1)
    outer = None
    for i in range(2,T):
        new_mat = np.array(value_list[i-1], copy=True)
        search_list = np.argwhere(new_mat<np.inf)
        for ind in search_list:
            curr_state = ind.tolist()
            curr_cost = new_mat[curr_state[0], curr_state[1], curr_state[2], curr_state[3], curr_state[4]]
            curr_state[0] += 1
            curr_state[1] += 1
            if curr_state == state:
                continue
            for u in range(5):
                next_state, cost = motion_model_cost(env, info, curr_state, u)
                # if curr_state == [3,2,3,1,1]:
                #     print(next_state, cost)
                if next_state is not None:
                    if next_state[:2] == goal_pos.tolist():
                        outer = True
                    if curr_cost+cost < new_mat[next_state[0]-1, next_state[1]-1, next_state[2], next_state[3], next_state[4]]:
                        new_mat[next_state[0]-1, next_state[1]-1, next_state[2], next_state[3],next_state[4]] = curr_cost+cost
                        # print(next_state)
                        # print(curr_state, u)
                        par_control[next_state[0] - 1, next_state[1] - 1, next_state[2], next_state[3], next_state[4]] = [curr_state, u]
        value_list.append(new_mat)
        last_mat = value_list[i-1]
        # print("next")
        if outer == True:
            break
        elif (last_mat == new_mat).all():
            break

    goalx = info['goal_pos'][0]
    goaly = info['goal_pos'][1]
    goal_state = [goalx, goaly, 0, 0, 0]
    min_VT_goal = new_mat[goal_state[0] - 1, goal_state[1] - 1, goal_state[2], goal_state[3], goal_state[4]]
    for gdi in range(4):
        for gk in range(2):
            for gd in range(2):
                cg_state = [goalx, goaly, gdi, gk, gd]
                cg_VT = new_mat[cg_state[0] - 1, cg_state[1] - 1, cg_state[2], cg_state[3], cg_state[4]]
                if cg_VT < min_VT_goal:
                    min_VT_goal = cg_VT
                    goal_state = cg_state

    goal_ac = par_control[goal_state[0]-1,goal_state[1]-1,goal_state[2],goal_state[3],goal_state[4]][1]
    optim_act_seq.append(goal_ac)
    last_state = par_control[goal_state[0]-1,goal_state[1]-1,goal_state[2],goal_state[3],goal_state[4]][0]
    while(last_state!=state):
        last_ac = par_control[last_state[0] - 1, last_state[1] - 1, last_state[2], last_state[3], last_state[4]][1]
        optim_act_seq.append(last_ac)
        last_state = par_control[last_state[0] - 1, last_state[1] - 1, last_state[2], last_state[3], last_state[4]][0]

    optim_act_seq = list(reversed(optim_act_seq))
    return optim_act_seq


def generate_policy():
    env_folder = './envs/random_envs'
    env_list = [os.path.join(env_folder, env_file) for env_file in os.listdir(env_folder)]
    policy = np.empty((6, 6, 4, 2, 2, 2, 3, 3), dtype=np.int)
    par_control = np.empty((6, 6, 4, 2, 2, 2, 3, 3), dtype=object)
    gif_n = 1
    for env_path in env_list:
        env, info = load_one_random_env(env_path)
        plot_env(env)
        optim_act_seq = []

        # Get the agent position
        agent_pos = env.agent_pos
        x = agent_pos[0]
        y = agent_pos[1]

        # Get the agent direction
        agent_dir = info['init_agent_dir']
        if all(agent_dir == np.array([0, -1])):
            dir = 0
        elif all(agent_dir == np.array([1, 0])):
            dir = 1
        elif all(agent_dir == np.array([-1, 0])):
            dir = 2
        elif all(agent_dir == np.array([0, 1])):
            dir = 3

        is_carrying = env.carrying is not None
        if is_carrying:
            key = 1
        else:
            key = 0

        door1 = env.grid.get(info['door_pos'][0][0], info['door_pos'][0][1])
        door2 = env.grid.get(info['door_pos'][1][0], info['door_pos'][1][1])
        is_d1open = door1.is_open
        is_d2open = door2.is_open
        if is_d1open:
            d1 = 1
        else:
            d1 = 0
        if is_d2open:
            d2 = 1
        else:
            d2 = 0

        key_pos = info['key_pos']
        goal_pos = info['goal_pos']
        if all(key_pos == [1,1]):
            kp = 0
        elif all(key_pos == [2,3]):
            kp = 1
        elif all(key_pos == [1,6]):
            kp = 2

        if all(goal_pos == [5,1]):
            gp = 0
        elif all(goal_pos == [6,3]):
            gp = 1
        elif all(goal_pos == [5,6]):
            gp = 2

        state = [x, y, dir, key, d1, d2, kp, gp]
        x_size = info['width'] - 2
        y_size = info['height'] - 2

        T = x_size * y_size * 4 * 2 * 2 * 2 * 3 * 3- 1

        value_list = []
        value_mat0 = np.ones((6, 6, 4, 2, 2, 2, 3, 3)) * np.inf
        value_mat0[x-1, y-1, dir, key, d1, d2, kp, gp] = 0
        value_list.append(value_mat0)
        value_mat1 = np.array(value_mat0, copy=True)

        for u in range(5):
            next_state, cost = partb_motion_model_cost(env,state,u)
            if next_state is not None:
                if cost < value_mat1[next_state[0] - 1, next_state[1] - 1, next_state[2], next_state[3], next_state[4],next_state[5],next_state[6],next_state[7]]:
                    value_mat1[next_state[0] - 1, next_state[1] - 1, next_state[2], next_state[3], next_state[4],next_state[5],next_state[6],next_state[7]] = cost
                    par_control[next_state[0] - 1, next_state[1] - 1, next_state[2], next_state[3], next_state[4],next_state[5],next_state[6],next_state[7]] = [state,u]

        value_list.append(value_mat1)

        outer = None

        for i in range(2, T):
            new_mat = np.array(value_list[i - 1], copy=True)
            search_list = np.argwhere(new_mat < np.inf)
            for ind in search_list:
                curr_state = ind.tolist()
                curr_cost = new_mat[curr_state[0], curr_state[1], curr_state[2], curr_state[3], curr_state[4],curr_state[5],curr_state[6],curr_state[7]]
                curr_state[0] += 1
                curr_state[1] += 1
                if curr_state == state:
                    continue
                for u in range(5):
                    next_state, cost = partb_motion_model_cost(env, curr_state, u)
                    if next_state is not None:
                        if next_state[:2] == goal_pos.tolist():
                            outer = True
                            goal_state = next_state
                        if curr_cost + cost < new_mat[next_state[0] - 1, next_state[1] - 1, next_state[2], next_state[3], next_state[4],next_state[5],next_state[6],next_state[7]]:
                            new_mat[next_state[0] - 1, next_state[1] - 1, next_state[2], next_state[3], next_state[4],next_state[5],next_state[6],next_state[7]] = curr_cost + cost
                            # print(next_state)
                            # print(curr_state, u)
                            par_control[next_state[0] - 1, next_state[1] - 1, next_state[2], next_state[3], next_state[4],next_state[5],next_state[6],next_state[7]] = [curr_state,u]
            value_list.append(new_mat)
            last_mat = value_list[i - 1]
            # print("next")
            if outer == True:
                break
            elif (last_mat == new_mat).all():
                break

        goal_ac = par_control[goal_state[0] - 1, goal_state[1] - 1, goal_state[2], goal_state[3], goal_state[4],goal_state[5],goal_state[6],goal_state[7]][1]
        optim_act_seq.append(goal_ac)
        last_state = par_control[goal_state[0] - 1, goal_state[1] - 1, goal_state[2], goal_state[3], goal_state[4],goal_state[5],goal_state[6],goal_state[7]][0]
        policy[last_state[0] - 1, last_state[1] - 1, last_state[2], last_state[3], last_state[4], last_state[5],
               last_state[6], last_state[7]] = goal_ac
        while (last_state != state):
            last_ac = par_control[last_state[0] - 1, last_state[1] - 1, last_state[2], last_state[3], last_state[4],last_state[5],last_state[6],last_state[7]][1]
            optim_act_seq.append(last_ac)
            last_state = par_control[last_state[0] - 1, last_state[1] - 1, last_state[2], last_state[3], last_state[4],last_state[5],last_state[6],last_state[7]][0]
            policy[last_state[0] - 1, last_state[1] - 1, last_state[2], last_state[3], last_state[4], last_state[5],
                   last_state[6], last_state[7]] = last_ac

        optim_act_seq = list(reversed(optim_act_seq))
        gifpath = './gif/random'+ str(gif_n) + '.gif'
        draw_gif_from_seq(optim_act_seq, env, gifpath)
        gif_n+=1
    np.save("policy.npy", policy)

def partb_state(env,info):
    # Get the agent position
    agent_pos = env.agent_pos
    x = agent_pos[0]
    y = agent_pos[1]

    # Get the agent direction
    agent_dir = info['init_agent_dir']
    if all(agent_dir == np.array([0, -1])):
        dir = 0
    elif all(agent_dir == np.array([1, 0])):
        dir = 1
    elif all(agent_dir == np.array([-1, 0])):
        dir = 2
    elif all(agent_dir == np.array([0, 1])):
        dir = 3

    is_carrying = env.carrying is not None
    if is_carrying:
        key = 1
    else:
        key = 0

    door1 = env.grid.get(info['door_pos'][0][0], info['door_pos'][0][1])
    door2 = env.grid.get(info['door_pos'][1][0], info['door_pos'][1][1])
    is_d1open = door1.is_open
    is_d2open = door2.is_open
    if is_d1open:
        d1 = 1
    else:
        d1 = 0
    if is_d2open:
        d2 = 1
    else:
        d2 = 0

    key_pos = info['key_pos']
    goal_pos = info['goal_pos']
    if all(key_pos == [1, 1]):
        kp = 0
    elif all(key_pos == [2, 3]):
        kp = 1
    elif all(key_pos == [1, 6]):
        kp = 2

    if all(goal_pos == [5, 1]):
        gp = 0
    elif all(goal_pos == [6, 3]):
        gp = 1
    elif all(goal_pos == [5, 6]):
        gp = 2

    state = [x, y, dir, key, d1, d2, kp, gp]
    return state

def partA():
    i = 1
    for env_path in envs:
    # env_path = './envs/example-8x8.env'
        env, info = load_env(env_path) # load an environment
        seq = doorkey_problem(env,info) # find the optimal action sequence
        path = './gif/'+str(i)+'.gif'
        draw_gif_from_seq(seq, load_env(env_path)[0],path) # draw a GIF & save
        i+=1
    
def partB():
    # env_folder = './envs/random_envs'
    # env_list = [os.path.join(env_folder, env_file) for env_file in os.listdir(env_folder)]
    # for env_path in env_list:
    #     env, info = load_one_random_env(env_path)

    env_folder = './envs/random_envs'
    env, info, env_path = load_random_env(env_folder)
    print(env_path)
    plot_env(env)
    policy = np.load("policy.npy")
    seq = []
    state = partb_state(env, info)
    cost = 0

    while(state != None):
        x = state[0]
        y = state[1]
        dir = state[2]
        key = state[3]
        d1 = state[4]
        d2 = state[5]
        kp = state[6]
        gp = state[7]
        ac = policy[x-1, y-1, dir, key, d1, d2, kp, gp]
        # print(ac)
        if ac not in range(5):
            break
        else:
            seq.append(ac)
            state,cost = partb_motion_model_cost(env,state,ac)

    path = './gif/random_doorkey.gif'
    draw_gif_from_seq(seq, env, path)


if __name__ == '__main__':
    partA()
    generate_policy()
    partB()


        
        
    

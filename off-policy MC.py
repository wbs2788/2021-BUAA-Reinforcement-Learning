# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Tutorial sample #7: The Maze Decorator

try:
    from malmo import MalmoPython
except:
    import MalmoPython

import os
import sys
import time
import json
import numpy as np
import random
# from priority_dict import priorityDictionary as PQ


# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately

def GetMissionXML(seed, gp, size=10):
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

              <About>
                <Summary>Hello world!</Summary>
              </About>

            <ServerSection>
              <ServerInitialConditions>
                <Time>
                    <StartTime>1000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
              </ServerInitialConditions>
              <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;7,44*49,73,35:1,159:4,95:13,35:13,159:11,95:10,159:14,159:6,35:6,95:6;12;"/>
                  <DrawingDecorator>
                    <DrawSphere x="-27" y="70" z="0" radius="30" type="air"/>
                  </DrawingDecorator>
                  <MazeDecorator>
                    <Seed>''' + str(seed) + '''</Seed>
                    <SizeAndPosition width="''' + str(size) + '''" length="''' + str(size) + '''" height="10" xOrigin="-32" yOrigin="69" zOrigin="-5"/>
                    <StartBlock type="emerald_block" fixedToEdge="true"/>
                    <EndBlock type="redstone_block" fixedToEdge="true"/>
                    <PathBlock type="diamond_block"/>
                    <FloorBlock type="air"/>
                    <GapBlock type="air"/>
                    <GapProbability>''' + str(gp) + '''</GapProbability>
                    <AllowDiagonalMovement>false</AllowDiagonalMovement>
                  </MazeDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="30000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>CS175AwesomeMazeBot</Name>
                <AgentStart>
                    <Placement x="0.5" y="56.0" z="0.5" yaw="0"/>
                </AgentStart>
                <AgentHandlers>
                    <DiscreteMovementCommands/>
                    <AgentQuitFromTouchingBlockType>
                        <Block type="redstone_block"/>
                    </AgentQuitFromTouchingBlockType>
                    <ObservationFromGrid>
                      <Grid name="floorAll">
                        <min x="-10" y="-1" z="-10"/>
                        <max x="10" y="-1" z="10"/>
                      </Grid>
                  </ObservationFromGrid>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''


def load_grid(world_state):
    """
    Used the agent observation API to get a 21 X 21 grid box around the agent (the agent is in the middle).

    Args
        world_state:    <object>    current agent world state

    Returns
        grid:   <list>  the world grid blocks represented as a list of blocks (see Tutorial.pdf)
    """
    while world_state.is_mission_running:
        # sys.stdout.write(".")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if len(world_state.errors) > 0:
            raise AssertionError('Could not load grid.')

        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            grid = observations.get(u'floorAll', 0)
            break
    return grid


#????????????????????????air_block???diamond_block
def find_start_end(grid):
    """
    Finds the source and destination block indexes from the list.

    Args
        grid:   <list>  the world grid blocks represented as a list of blocks (see Tutorial.pdf)

    Returns
        start: <int>   source block index in the list
        end:   <int>   destination block index in the list
    """
    #------------------------------------
    #
    #   Fill and submit this code
    #
#     return (None, None)
    #-------------------------------------
    counter = 0
    eb_index = None
    rb_index = None
    air_block=[]
    diamond_block=[]
    state=[]
    for i in grid:
        if i =='diamond_block':
            diamond_block.append(counter)
        
        if i =='air':
            air_block.append(counter)

        if i == 'emerald_block':
            eb_index = counter
           
        if i == 'redstone_block':
            rb_index = counter

        state.append(counter)    
        counter+=1
    
    return (eb_index, rb_index,air_block,diamond_block)
    # -------------------------------------
def extract_action_list_from_path(path_list):
    """
    Converts a block idx path to action list.

    Args
        path_list:  <list>  list of block idx from source block to dest block.

    Returns
        action_list: <list> list of string discrete action commands (e.g. ['movesouth 1', 'movewest 1', ...]
    """
    action_trans = {-21: 'movenorth 1', 21: 'movesouth 1', -1: 'movewest 1', 1: 'moveeast 1'}
    alist = []
    for i in range(len(path_list) - 1):
        curr_block, next_block = path_list[i:(i + 2)]
        alist.append(action_trans[next_block - curr_block])

    return alist
 # ?????????????????????????????????????????????????????????????????????????????????reward??????????????????????????????state
def Reward_state_action(s, a):
    # ????????????
    if a == 0:
    # Fill and submit this code
        new_s = s - 21
        if new_s == end:
            return ((True, 1, new_s))
        if new_s in states:
            return((False, -1, new_s))
        return((False, -1, s))
    # ????????????
    elif a == 1:
    # Fill and submit this code
        new_s = s + 21
        if new_s == end:
            return ((True, 1, new_s))
        if new_s in states:
            return((False, -1, new_s))
        return((False, -1, s))
    # ????????????
    elif a == 2:
    # Fill and submit this code
        new_s = s - 1
        if new_s == end:
            return ((True, 1, new_s))
        if new_s in states:
            return((False, -1, new_s))
        return((False, -1, s))
    # ????????????
    else:
        new_s = s + 1
        if new_s == end:
            return ((True, 1, new_s))
        if new_s in states:
            return((False, -1, new_s))
        return((False, -1, s))
    # Fill and submit this code

def epsilon_greedy(qtem, s, epsilon):
    p = np.random.random()
    if p < epsilon:
        action = np.random.choice(actions)
    else :
        action = qtem[states.index(s)]
    # -------------------------------------
    # epsilon_greedy, ?????????????????????????????????
    # -------------------------------------
    return action

# on policy ??????epsilon-greedy????????????num?????????????????????????????????????????????episode????????????episode??????????????????
def Monte_Carlo(num, epsilon, gamma):
    # -------------------------------------
    # ????????????-?????? ??????qfunc (Q[s,a])????????????
    qfunc = np.random.normal(size=(len(states),len(actions)))
    # ??????Nqfunc????????????episode??????s,a??????????????????
    Nqfunc = np.zeros((len(states),len(actions)))
    # ????????????????????????-????????????qtem?????????????????????????????????????????????
    qtem = np.argmax(qfunc, axis=1)
    for k in range(num):
        if k % 25 == 0:
            epsilon = max(epsilon * 0.99, 0.01) 
        # ??????num?????????
            # ??????epsilon-greedy???????????????K???episode????????????
        k_states = []
        k_actions = []
        k_reward = []
            # ?????????????????????????????????K???episode???????????????states,actions,reward??????
        k_start = start
        k_states.append(k_start)
            # ????????????????????????
        k_tag = False
        if k_start == end:
            k_tag = True
        while not k_tag:
                # ????????????????????????
            action = epsilon_greedy(qtem, k_start, epsilon)
            k_actions.append(action)
                # ??????epsilon-greedy?????????????????????????????????
            k_tag, reward, new_state = Reward_state_action(k_start, action)
            # ????????????????????????
                # ??????????????????????????????
                # ???????????????????????????????????????
                # ???????????????????????????
            k_states.append(new_state)
            k_reward.append(reward)
            k_start = new_state
            # ????????????????????????????????????????????????????????????->????????????????????????

            # ??????????????????????????????????????????episode?????????????????????
            # ??????????????????????????? g
        g = 0.0 # boundary g[t+1] = 0
        l = len(k_actions)
        w = 1.0
            # ?????????????????????????????????????????? g
        for i in range(l):
            t = l - i - 1
            g = gamma * g + k_reward[t]
            j_s, j_a = states.index(k_states[t]), k_actions[t]
            Nqfunc[j_s][j_a] += 2
                # ?????????????????????????????????????????????
            # ??????????????????????????????????????????????????????-?????????????????????qfunc
            qfunc[j_s][j_a] = qfunc[j_s][j_a] +\
                (g - qfunc[j_s][j_a])/Nqfunc[j_s][j_a] * w
                    # ?????????s-a?????????g?????????qfunc[s,a]??????????????????????????????????????????????????? qfunc[s,a]
                    # ??????????????????????????????????????????g???????????????????????????
            qtem[j_s] = np.argmax(qfunc[j_s])
            if j_a != qtem[j_s]:
                break
            w = w * epsilon
        # num???????????????????????????qfunc
    # -------------------------------------
    print(qfunc)
    return qfunc

#???????????????qfunc??????????????????
def get_shortest_path(qfunc):
    s_path = []
    s_path.append(start)
    cur = start
    while cur != end:
        print(cur)
        action = np.argmax(qfunc[states.index(cur)])
        tc = Reward_state_action(cur, action)
        cur = tc[2]
        s_path.append(cur)
    # -------------------------------------
    # ?????????????????????qfunc?????????????????????
    # -------------------------------------
    return s_path

# Create default Malmo objects:
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse(sys.argv)
except RuntimeError as e:
    print('ERROR:', e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

if agent_host.receivedArgument("test"):
    num_repeats = 1
else:
    num_repeats = 5

for i in range(num_repeats):
    size = int(5)
    print("Size of maze:", size)
    my_mission = MalmoPython.MissionSpec(GetMissionXML("0", 0.4 + float(i / 20.0), size), True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    my_mission.requestVideo(800, 500)
    my_mission.setViewpoint(1)
    # Attempt to start a mission:
    max_retries = 3
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))  # add Minecraft machines here as available

    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_clients, my_mission_record, 0, "%s-%d" % ('Moshe', i))
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission", (i + 1), ":", e)
                exit(1)
            else:
                time.sleep(2)

    # Loop until mission starts:
    print("Waiting for the mission", (i + 1), "to start ", )
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        # sys.stdout.write(".")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)

    print()
    print("Mission", (i + 1), "running.")

    grid = load_grid(world_state)
    air_block = []
    diamond_block = []
    start,end,air_block,diamond_block=find_start_end(grid)
    states = []   # ???????????????????????????????????????agent????????????states
    actions = np.arange(4)  # ??????actions
    num = 5000        #??????????????????
    epsilon = 0.95     #??????epsilon
    gamma = 1      #??????gamma

    s=['emerald_block','diamond_block','redstone_block']
    counter=0
    for j in grid:
        if j in s:
            states.append(counter)
        counter +=1

    q = Monte_Carlo(num, epsilon, gamma)
    path = get_shortest_path(q)
    print(path)
    action_list = extract_action_list_from_path(path)

    print("Output (start,end)", (i + 1), ":", (start, end))
    print("Output (path length)", (i + 1), ":", len(path))
    print("Output (actions)", (i + 1), ":", action_list)
    # Loop until mission ends:
    action_index = 0
    while world_state.is_mission_running:
        # sys.stdout.write(".")
        time.sleep(0.1)

        # Sending the next commend from the action list -- found using the Dijkstra algo.
        if action_index >= len(action_list):
            print("Error:", "out of actions, but mission has not ended!")
            time.sleep(2)
        else:
            agent_host.sendCommand(action_list[action_index])
        action_index += 1
        if len(action_list) == action_index:
            # Need to wait few seconds to let the world state realise I'm in end block.
            # Another option could be just to add no move actions -- I thought sleep is more elegant.
            time.sleep(2)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)

    print()
    print("Mission", (i + 1), "ended")
    # Mission has ended.

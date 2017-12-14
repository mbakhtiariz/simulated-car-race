from pytocl.driver import Driver
from pytocl.car import State, Command,MPS_PER_KMH
import numpy as np
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
import math
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
from pytocl.analysis import DataLogWriter
import os
class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    ...
    # def drive(self, carstate: State) -> Command:
    #     # Interesting stuff
    #     command = Command(...)
    #     return command

    def __init__(self, logdata=True, port = "3001"):
        super(MyDriver, self).__init__(logdata=False)
        self.steering_ctrl = CompositeController(
            ProportionalController(0.4),
            IntegrationController(0.2, integral_limit=1.5),
            DerivativeController(2)
        )
        self.acceleration_ctrl = CompositeController(
            ProportionalController(3.7),
        )
        self.data_logger = DataLogWriter() if logdata else None
        self.driver = joblib.load('neatNetworks/best_net_8.pkl')
        self.steer=[]
        self.smoothing_len = 50
        self.weights = np.exp(-(self.smoothing_len - 1 - np.arange(self.smoothing_len)) / 20.0)
        self.tic_counter = 0
        self.recovery = 0
        self.speed = []
        self.last_recovery = 0
        self.recovery_1_start = 0
        self.port = port
        self.coop_port = -1
        f = open('neatNetworks/ports.txt', 'a')
        f.write(str(self.port) + "\n")
        f.close()


    @property
    def range_finder_angles(self):
        """Iterable of 19 fixed range finder directions [deg].

        The values are used once at startup of the client to set the directions
        of range finders. During regular execution, a 19-valued vector of track
        distances in these directions is returned in ``state.State.tracks``.
        """
        return -90, -75, -60, -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, \
            30, 45, 60, 75, 90

    def on_shutdown(self):
        """
        Server requested driver shutdown.

        Optionally implement this event handler to clean up or write data
        before the application is stopped.
        """
        #if os.path.exists('neatNetworks/ports.txt'):
        #    os.remove('neatNetworks/ports.txt')
        #if os.path.exists('neatNetworks/state_' + self.port + '.txt'):
        #    os.remove('neatNetworks/state_' + self.port + '.txt')			
        f = open('neatNetworks/ports.txt','w')
        f.close()
        f = open('neatNetworks/state_' + self.port + '.txt','w')
        f.close()		
        if self.data_logger:
            self.data_logger.close()
            self.data_logger = None

    #ranges = [-0.2,-0.05,-0.01, 0.01, 0.05,0.2]

    def state_representation(self, carstate):
        #state = np.zeros((1, 22), dtype='float32')
        # state2 = np.zeros((1, 23), dtype='float32')
        state = np.zeros((1, 26), dtype='float32')
        state[0, 0] = math.sqrt(carstate.speed_x ** 2 + carstate.speed_y ** 2 + carstate.speed_z ** 2)
        state[0, 1] = carstate.distance_from_center
        state[0, 2] = carstate.angle
        state[0, 3] = carstate.opponents[4]
        state[0, 4] = carstate.opponents[13]
        state[0, 5] = carstate.opponents[22]
        state[0, 6] = carstate.opponents[31]
        state[0, 7:] = carstate.distances_from_edge
        #state[0, 3:] = carstate.distances_from_edge
        return state

    def normal_decision(self, state, carstate, command):
        response = self.driver.activate(tuple(state[0, :]))
        command.steering = response[0] / 2.0
        if response[1] > 0:
            command.accelerator = max(0.7, min(1, response[1] * 2))
            command.brake = 0
        else:
            command.accelerator = 0
            command.brake = abs(response[1]) / 6.0

        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1

        if carstate.rpm < 2500:
            command.gear = min(carstate.gear - 1, 1)

        if not command.gear:
            command.gear = carstate.gear or 1

    def find_port(self):
        with open('neatNetworks/ports.txt', 'r') as file:
            for line in file:
                if(line.strip() != self.port):
                    self.coop_port = line.strip()
                    print ("friend port is:", self.coop_port)

    def read_coop_state(self):
        if self.coop_port == -1:
            self.find_port()
        values = []
        file_name = 'neatNetworks/state_'+ str(self.coop_port) +'.txt'
        if os.path.isfile(file_name):
            f = open(file_name, 'r')
            for l in f:
                values = [float(x) for x in l.strip().split(",")]#val=[coop_race_pos,coop_dist_from_start]
        return values


    def aggressive_run(self, carstate, command):
        dist_to_left_op = carstate.opponents[18]
        dist_to_right_op = carstate.opponents[0]
        if dist_to_left_op<=dist_to_right_op and dist_to_left_op<5:
            command.steering=0.5
        elif dist_to_right_op<=dist_to_left_op and dist_to_right_op<5:
            command.steering = -0.5


    def make_decision(self, state, carstate, command):
        if self.recovery == 0:
            self.normal_decision(state, carstate, command)
            if self.tic_counter % 25 ==0:
                coop_state = self.read_coop_state()
                # if we don't know yet about our cooperator
                if len(coop_state) < 2:
                    return
                #else:
                    #print(self.port,"++++distance from friend:", coop_state[1] - carstate.distance_from_start)

                dist = abs(coop_state[1] - carstate.distance_from_start)
                # if we are too close to our co_operator

                if coop_state[0] < carstate.race_position and dist < 5:
                    #print(self.port, "---slow down a bit....           ")
                    command.brake = 0.5
                    command.accelerator = 0
                # if we are close to friend and an opponent on either sides
                # then be a killer!
                elif coop_state[0] < carstate.race_position and dist < 15:
                    #print(self.port, "------------entering aggressive mode :/")
                    self.aggressive_run(carstate, command)

        elif self.recovery==1:
            command.gear=-1
            command.accelerator = 0.3
            if carstate.distances_from_edge[10]>carstate.distances_from_edge[8] and carstate.distance_from_center > 0:
                command.brake=0
                command.steering = 2.0*carstate.angle/abs(carstate.angle)
                #print("..................1-1..................")
            elif carstate.distances_from_edge[10]<= carstate.distances_from_edge[8] and carstate.distance_from_center > 0:
                command.brake=0
                command.steering = 3*carstate.angle/abs(carstate.angle)
                #print("..................1-2..................:")
            elif carstate.distances_from_edge[10]>carstate.distances_from_edge[8] and carstate.distance_from_center < 0:
                command.brake=0
                command.steering = -2*carstate.angle/abs(carstate.angle)
                #print("..................1-3..................")
            elif carstate.distances_from_edge[10]<=carstate.distances_from_edge[8] and carstate.distance_from_center < 0:
                command.brake=0
                command.steering = -3*carstate.angle/abs(carstate.angle)
                #print("..................1-4..................")
            #if carstate.distance_from_center>0:
            if abs(carstate.distances_from_edge[9])>2:
                command.steering = -6*carstate.angle
                #print(".............front>2...............")
            #print("___________",'\n',carstate.current_lap_time, self.recovery_1_start, carstate.speed_x,'\n',"____________________" )
            if abs(carstate.distances_from_edge[9])>6 or (carstate.current_lap_time - self.recovery_1_start > 3 and carstate.speed_x <= 0):
                self.recovery = 2
                command.gear=1
                command.brake=1
                print("_______Recovery1 done successfully!_______", carstate.angle)
            #print(self.recovery,command.accelerator,command.brake,command.gear,command.steering,carstate.angle,carstate.distance_from_center)
        elif self.recovery==2:
            self.recovery_1_start = carstate.current_lap_time
            command.steering = 1*carstate.angle/abs(carstate.angle)
            #print("doing recovery 2222222")
            command.accelerator=0.5
            command.gear = 1
            command.brake = 0
            if abs(carstate.angle)<2:
                print("_______Recovery2 done successfully!_______")
                self.recovery=3
                command.brake=0
                command.gear=1
                command.accelerator = 1
                command.steering = 10
                self.last_recovery = self.tic_counter
            if abs(carstate.distances_from_edge[9])<2 and carstate.angle * carstate.distance_from_center < 0:
                self.recovery = 1
            #print(self.recovery,command.accelerator,command.brake,command.gear,command.steering,carstate.angle,carstate.distance_from_center)
        else:
            #print ("int mode 3...")
            command.brake = 0
            command.gear = 1
            command.accelerator = 1
            if carstate.speed_x>6:
                self.recovery = 0
                command.brake = 0
                command.gear = 1	
                command.accelerator = 1
                self.last_recovery = self.tic_counter
                print ("all recov done!")
            #print(self.recovery,command.accelerator,command.brake,command.gear,command.steering,carstate.angle,carstate.distance_from_center)

        return command

    def write_mystate(self, carstate):
        f = open('neatNetworks/state_' + self.port + '.txt', 'w')
        f.write(str(carstate.race_position)+","+str(carstate.distance_from_start))
        f.close()

    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """
        command = Command()
        self.tic_counter += 1
        self.speed.append(carstate.speed_x)
        
        if self.recovery == 0 and self.tic_counter>500 and  abs(np.mean(np.array(self.speed[-40:])))<6 and (self.last_recovery +500 < self.tic_counter or carstate.distances_from_edge[9]<2):
            self.recovery = 1
            self.recovery_1_start = carstate.current_lap_time
            print (">>>>Recovery mode activated!<<<<")


        state = self.state_representation(carstate)

        self.make_decision(state, carstate, command)
        #state2[0,1:] = state[0]
        if self.tic_counter%25==0:
            self.write_mystate(carstate)
            print("rank:,",carstate.race_position," port:", self.port, " distfromstart:",carstate.distance_from_start," gear:",carstate.gear," speed:",carstate.speed_x)
        return command




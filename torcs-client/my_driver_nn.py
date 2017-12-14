from pytocl.driver import Driver
from pytocl.car import State, Command
import numpy as np
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
import math
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
from pytocl.analysis import DataLogWriter

class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    ...
    # def drive(self, carstate: State) -> Command:
    #     # Interesting stuff
    #     command = Command(...)
    #     return command

    def __init__(self, logdata=True):
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
        self.acc_clf = joblib.load('nn_train/ann_acc.pkl')
        self.brk_clf = joblib.load('nn_train/ann_brk.pkl')
        self.str_clf = joblib.load('nn_train/ann_str_2.pkl')
        self.thresholds = {'acc':[0.01, np.inf], 'brk':[0.3, 0.7, np.inf], 'str':[-0.5,-0.2,-0.05,-0.01,0, 0.01, 0.05,0.2,0.5, np.inf]}
        self.tic_counter = 0
        self.recovery = 0
        self.speed = []
        self.last_recovery = 0
        self.recovery_1_start = 0


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
        if self.data_logger:
            self.data_logger.close()
            self.data_logger = None

    def find_value_brk(self, c):
        if c==0:
            return 0
        if c==1:
            return 0.5
        if c==2:
            return 1


    def find_value_brk_p(self, p):
        if p[1]>p[2] and p[1]>0.2:
            return 0.5
        if p[2]>=p[1] and p[2]>0.2:
            return 1
        return 0

    #ranges = [-0.2,-0.05,-0.01, 0.01, 0.05,0.2]
    def find_value_str_2(self, c):
        if c==0:
            return -0.3
        if c==1:
            return -0.1
        if c==2:
            return -0.015
        if c==3:
            return 0
        if c==4:
            return 0.015
        if c==5:
            return 0.1
        if c==6:
            return 0.3




    def find_value_str(self, c):
        if c==0:
            return -0.7
        if c==1:
            return -0.3
        if c==2:
            return -0.1
        if c==3:
            return -0.02
        if c==4:
            return -0.004
        if c==5:
            return 0.004
        if c==6:
            return 0.02
        if c==7:
            return 0.1
        if c==8:
            return 0.3
        if c==9:
            return 0.7

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def state_representation(self, carstate):
        state = np.zeros((1, 22), dtype='float32')
        # state2 = np.zeros((1, 23), dtype='float32')
        #state = np.zeros((1, 26), dtype='float32')
        state[0, 0] = math.sqrt(carstate.speed_x ** 2 + carstate.speed_y ** 2 + carstate.speed_z ** 2)
        state[0, 1] = carstate.distance_from_center
        state[0, 2] = carstate.angle
        state[0, 3:] = carstate.distances_from_edge
        #state[0, 3:] = carstate.distances_from_edge
        return state


#_________________________________________________________________________________________________

    def make_decision(self, state, carstate, command):
        # if we are at a turn
        if self.recovery==0:
            #command.accelerator = self.acc_clf.predict(state)[0]
            command.steering = self.find_value_str_2(self.str_clf.predict(state)[0])
            print(self.acc_clf.predict(state)[0])
            if self.acc_clf.predict(state)[0] > 0:
                command.accelerator = max(0.7, min(1, self.acc_clf.predict(state)[0] * 1.5))
                command.brake = 0
            else:
                command.accelerator = 0
                command.brake = abs(self.acc_clf.predict(state)[0]) / 2.0
            #self.steer.append(command.steering)
            # smooth steering
            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1

            if carstate.rpm < 2500	:
                command.gear = min(carstate.gear - 1, 1)

            if not command.gear:
                command.gear = carstate.gear or 1
            print(self.recovery,command.accelerator,command.brake,command.gear,command.steering,carstate.angle,carstate.distance_from_center)
        elif self.recovery==1:
            command.gear=-1
            command.accelerator = 0.3
            if carstate.distances_from_edge[10]>carstate.distances_from_edge[8] and carstate.distance_from_center > 0:
                command.brake=0
                command.steering = 2.0*carstate.angle/abs(carstate.angle)
                print(">>>>>>>>>>>>>>>>>1-1<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",carstate.speed_x,carstate.distances_from_edge[9])
            elif carstate.distances_from_edge[10]<= carstate.distances_from_edge[8] and carstate.distance_from_center > 0:
                command.brake=0
                command.steering = 3*carstate.angle/abs(carstate.angle)
                print(">>>>>>>>>>>>>>>>>1-2<<<<<<<<<<<<<<<<<<<<<<<<<<<<<:",carstate.speed_x,carstate.distances_from_edge[9])
            elif carstate.distances_from_edge[10]>carstate.distances_from_edge[8] and carstate.distance_from_center < 0:
                command.brake=0
                command.steering = -2*carstate.angle/abs(carstate.angle)
                print(">>>>>>>>>>>>>>>>>1-3<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",carstate.speed_x,carstate.distances_from_edge[9])
            elif carstate.distances_from_edge[10]<=carstate.distances_from_edge[8] and carstate.distance_from_center < 0:
                command.brake=0
                command.steering = -3*carstate.angle/abs(carstate.angle)
                print(">>>>>>>>>>>>>>>>>1-4<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",carstate.speed_x,carstate.distances_from_edge[9])
            #if carstate.distance_from_center>0:
            if abs(carstate.distances_from_edge[9])>2:
                command.steering = -6*carstate.angle
                #print(".............front>2...............")
            print("___________",'\n',carstate.current_lap_time, self.recovery_1_start, carstate.speed_x,'\n',"____________________" )
            if abs(carstate.distances_from_edge[9])>6 or (carstate.current_lap_time - self.recovery_1_start > 3 and carstate.speed_x <= 0):
                self.recovery = 2
                command.gear=1
                command.brake=1
                print(">>>>Recovery1 done successfully!<<<<", carstate.angle)
            print(self.recovery,command.accelerator,command.brake,command.gear,command.steering,carstate.angle,carstate.distance_from_center)
        elif self.recovery==2:
            self.recovery_1_start = carstate.current_lap_time
            command.steering = 1*carstate.angle/abs(carstate.angle)
            print("doing recovery", carstate.angle)
            command.accelerator=0.5
            command.gear = 1
            command.brake = 0
            if abs(carstate.angle)<2:
                print(">>>>Recovery2 done successfully!<<<<", carstate.angle)
                self.recovery=3
                command.brake=0
                command.gear=1
                command.accelerator = 1
                command.steering = 10
                self.last_recovery = self.tic_counter
            if abs(carstate.distances_from_edge[9])<2 and carstate.angle * carstate.distance_from_center < 0:
                self.recovery = 1
            print(self.recovery,command.accelerator,command.brake,command.gear,command.steering,carstate.angle,carstate.distance_from_center)
        else:
            print ("int mode 3...")
            command.brake = 0
            command.gear = 1
            command.accelerator = 1
            if carstate.speed_x>6:
                self.recovery = 0
                command.brake = 0
                command.gear = 1	
                command.accelerator = 1
                self.last_recovery = self.tic_counter
                print ("all done!")
            print(self.recovery,command.accelerator,command.brake,command.gear,command.steering,carstate.angle,carstate.distance_from_center)

        return command



#-----------------------------------------------------------------------------------------------------------------
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

        return command

































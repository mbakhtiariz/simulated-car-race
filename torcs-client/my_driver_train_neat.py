from pytocl.driver import Driver
from pytocl.car import State, Command, MPS_PER_KMH
import numpy as np
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
import math
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
from pytocl.analysis import DataLogWriter
import time
import os

class MyDriver(Driver):

    def __init__(self, logdata=False):
        self.initial_time = time.time()
        self.steering_ctrl = CompositeController(
            ProportionalController(0.4),
            IntegrationController(0.2, integral_limit=1.5),
            DerivativeController(2)
        )
        self.acceleration_ctrl = CompositeController(
            ProportionalController(3.7),
        )
        self.data_logger = DataLogWriter() if logdata else None
        self.fitness = 0
        self.out_time = 0
        self.avg_speed = 0
        self.tic_counter= 0
        self.speed = []
        self.speed_ctrl_steps = 250 # num of tics for avg speed for termination check (5s)
        self.passed_start = False
        self.steer = []
        self.steer_deviation = 0
        self.smoothing_len = 50
        self.weights = np.exp(-(self.smoothing_len-1-np.arange(self.smoothing_len))/20.0)
        self.coefficients = {'out':-2, 'speed':20, 'dist':5, 'dev': -0.5, 'damage':-0.05,
                             'rank':-100}
        self.path = 'neatNetworks/'
        self.exp_num = 8
        self.model = joblib.load(self.path + 'neat_model_' + str(self.exp_num) + '.pkl')
        self.best_fitness_path = self.path+"best_fitness_%d.txt"%self.exp_num
        if os.path.exists(self.best_fitness_path):
            f = open(self.best_fitness_path)
            for l in f:
                print(l)
            self.best_fitness = float(l.strip())
            f.close()
        else:
            self.best_fitness = 0

    def on_restart(self):
        self.fitness = 0
        self.out_time = 0
        self.avg_speed = 0
        self.tic_counter = 0
        self.speed = []
        self.passed_start = False
        self.steer = []
        self.steer_deviation = 0

    def state_representation(self, carstate):
        state = np.zeros((1, 26), dtype='float32')
        state[0, 0] = math.sqrt(carstate.speed_x ** 2 + carstate.speed_y ** 2 + carstate.speed_z ** 2)
        state[0, 1] = carstate.distance_from_center
        state[0, 2] = carstate.angle
        state[0, 3] = carstate.opponents[4]
        state[0, 4] = carstate.opponents[13]
        state[0, 5] = carstate.opponents[22]
        state[0, 6] = carstate.opponents[31]
        state[0, 7:] = carstate.distances_from_edge
        return state

    def act(self, state, carstate, command):

        activation = self.model.activate(tuple(state[0, :]))
        command.steering = activation[0]
        if activation[1]>0:
            command.accelerator = activation[1]
            command.brake = 0
        else:
            command.accelerator = 0
            command.brake = abs(activation[1])/4.0
        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1

        if carstate.rpm < 2500:
            command.gear = min(carstate.gear - 1, 1)

        if not command.gear:
            command.gear = carstate.gear or 1

        #v_x = 150
        #self.accelerate(carstate, v_x, command)
        # for steer smoothing
        '''
        length = int(min(self.tic_counter, self.smoothing_len))
        #smooth steering
        command.steering = np.sum(np.array(self.steer[-length:])*self.weights[-length:])/np.sum(self.weights[-length:])
        '''

    def update_variables(self, state, carstate, command):
        self.speed.append(state[0, 0])
        self.avg_speed = (self.avg_speed * self.tic_counter + state[0, 0]) / (self.tic_counter + 1)
        self.tic_counter += 1.0
        if not self.passed_start and carstate.distance_from_start < 20:
            self.passed_start = True
        self.steer.append(command.steering)
        if (abs(carstate.distance_from_center) > 1):
            self.out_time += 1
        self.steer_deviation += np.std(np.array(self.steer))

    def fitness_eval(self, carstate):
        self.fitness = max(1, 1200 + self.out_time * self.coefficients['out'] +
                           self.avg_speed * self.coefficients['speed'] +
                           carstate.distance_raced * self.coefficients['dist'] +
                           self.steer_deviation * self.coefficients['dev'] +
                           carstate.damage * self.coefficients['damage'] +
                           carstate.race_position*self.coefficients['rank'])

    def termination_needed(self, carstate):
        # termination conditions:
        recent_speed_average = np.mean(np.array(self.speed[-self.speed_ctrl_steps:]))
        if self.fitness < 500:
            return True

        if len(self.speed) > self.speed_ctrl_steps and recent_speed_average < 2:  # if the car is stuck somewhere
            print(">>>>>speed average in last 250 tics", recent_speed_average)
            self.fitness -= 1000
            return True

        if carstate.damage > 10000:
            print(">>>>>carstate.damage > 7000", carstate.damage)
            return True

        # if it is stuck just at the beginning
        if carstate.current_lap_time > 5 and carstate.distance_raced < 10:
            print("after 5 secs, only raced %f meters" % carstate.distance_raced)
            self.fitness -= 1000
            return True

        if carstate.current_lap_time > 60:
            print(">>>>>carstate.current_lap_time > 30", carstate.current_lap_time)
            self.fitness += 1000
            return True


        return False

    def drive(self, carstate: State) -> Command:
        command = Command()
        state = self.state_representation(carstate)
        self.act(state, carstate, command)
        self.update_variables(state, carstate, command)
        self.fitness_eval(carstate)
        if self.termination_needed(carstate):
            command.meta = 1
            command.finish = True
            if self.fitness>self.best_fitness:
                joblib.dump(self.model, self.path + 'best_net_%d.pkl' % self.exp_num)
                f = open(self.best_fitness_path, "w")
                f.write(str(self.fitness))
                f.close()
        self.write_fitness_to_file(max(self.fitness, 1))
        print('fitness [%d,= out:%d, avg_sp:%d, len:%d, dev:%d, dmg:%d, rnk:%d]\r' %
              (self.fitness, self.out_time * self.coefficients['out'],
               self.avg_speed * self.coefficients['speed'],
               carstate.distance_raced * self.coefficients['dist'],
               self.steer_deviation * self.coefficients['dev'],
               carstate.damage * self.coefficients['damage'],
               carstate.race_position * self.coefficients['rank']), end="")
        return command


    def accelerate(self, carstate, target_speed, command):
        # compensate engine deceleration, but invisible to controller to
        # prevent braking:
        speed_error = 1.0025 * target_speed * MPS_PER_KMH - carstate.speed_x
        acceleration = self.acceleration_ctrl.control(speed_error,
                                                      carstate.current_lap_time)

        # stabilize use of gas and brake:
        acceleration = math.pow(acceleration, 3)

        if acceleration > 0:
            if abs(carstate.distance_from_center) >= 1:
                # off track, reduced grip:
                acceleration = min(0.4, acceleration)

            command.accelerator = min(acceleration, 1)

            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1

        if carstate.rpm < 2500:
            command.gear = min(carstate.gear - 1, 1)

        if not command.gear:
            command.gear = carstate.gear or 1



    def write_fitness_to_file(self, fit_val):
        with open(self.path+'fitness_'+str(self.exp_num)+'.txt', 'w') as file:
            file.write(str(fit_val)+'\n')
        file.close()


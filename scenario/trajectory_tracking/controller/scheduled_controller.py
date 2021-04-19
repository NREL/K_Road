import csv
import math

import numpy as np
from matplotlib import pyplot

from k_road.k_road_process import KRoadProcess
from factored_gym import ActionTransform
from factored_gym.controller.controller import *


class ScheduledController(ActionTransform, Controller):
    
    def __init__(self, inputs: [np.ndarray] = None, input_csv: str = ''):
        if input_csv != '':
            inputs = []
            with open(input_csv, newline='') as csvfile:
                for row in csv.reader(csvfile, delimiter=',', quotechar='"'):
                    inputs.append(np.array([float(e) for e in row]))
        else:
            if inputs is None:
                inputs = []
                steps_per_second = int(1 / .05)
                for i in range(10 * steps_per_second):
                    inputs.append(np.array([2.0, 0.0]))
                for i in range(10 * steps_per_second):
                    inputs.append(np.array([0.0, 0.0]))
                for i in range(10 * steps_per_second):
                    inputs.append(np.array([0.0, 10 * (math.pi / 180)]))
                for i in range(10 * steps_per_second):
                    inputs.append(np.array([0.0, 0.0]))
                for i in range(10 * steps_per_second):
                    inputs.append(np.array([0.0, -20 * (math.pi / 180)]))
                for i in range(20 * steps_per_second):
                    inputs.append(np.array([-1.0, 0.0]))
                for i in range(10 * steps_per_second):
                    inputs.append(np.array([0.0, 0.0]))
        self.inputs: [] = inputs
        
        print('input len {}'.format(len(self.inputs)))
        pyplot.plot(self.inputs)
        pyplot.show()
        with open('/home/ctripp/pulsed_controls.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for input in self.inputs:
                writer.writerow(input.tolist())
    
    def get_action(self, process: KRoadProcess):
        return self.transform_action(process, np.array([0.0, 0.0]))
    
    def transform_action(self, process: KRoadProcess, action):
        result = self.inputs[process.time_step_number % len(self.inputs)]
        return result

import random
from copy import deepcopy
from math import *
from typing import *

from pymunk import Vec2d

from k_road.util import clamp


def fixed_trajectory_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
    # paths with some smooth curving nature
    length = 1000
    target_speed = random.uniform(5, 40)
    # target_speed = 10.0

    waypoints = [(Vec2d(0, 0), target_speed)]
    turning_rate = 0.0
    turn_length = 0.0
    angle = 0.0
    distance = 0.0
    while distance < length:

        if turn_length <= 0.0:
            if turning_rate != 0:
                turning_rate = 0.0
            else:
                # https://www.fhwa.dot.gov/publications/research/safety/17098/004.cfm
                min_radius = 1.0 * (target_speed ** 2) / (15 * (.01 * 0.0 + .21))
                # radians / meter
                max_rate = 1.0 / min_radius
                turning_rate = random.uniform(-1.0, 1.0) * max_rate  # [rad / m]
                # turning_rate = random.uniform(-1.0, 1.0) * (90.0 / target_speed) * (pi / 180.0)  # [rad / m]

            max_turn_length = min(length,
                                  length if turning_rate == 0.0 else \
                                      (120 * pi / 180.0) / abs(turning_rate))
            turn_length = random.uniform(1, max_turn_length)
            # print('seg: ', turning_rate, turn_length, max_turn_length)

        segment_length = 1.0

        segment_angle = (turning_rate + random.gauss(0.0, 2.0 * (pi / 180))) * segment_length
        angle += segment_angle
        delta = Vec2d(segment_length, 0.0)
        delta.rotate(angle)

        last = waypoints[-1]
        next = last[0] + delta
        waypoints.append((next, target_speed))
        distance += segment_length

        turn_length -= segment_length

    return waypoints, False


def curriculum_trajectory_factory(
        min_speed=4,
        max_speed=50,
        min_turn_rate_multiplier=0,
        max_turn_rate_multiplier=2,
        total_length=10000,
        random_seed=3,
        ) -> Callable[['TrajectoryTrackingProcess'], Tuple[List[Tuple[Vec2d, float]], bool]]:
    r = random.Random(random_seed)

    # speed = (max_speed + min_speed) / 2

    # start with a straight line

    # waypoints = [(Vec2d(0, 0), speed), (Vec2d(.001, 0), speed)]
    waypoints = [(Vec2d(0, 0), min_speed)]

    length = 0.0
    angle = 0.0

    def add_segment(segment_length, segment_angle):
        nonlocal angle, length, speed
        angle += segment_angle
        length += segment_length

        delta = Vec2d(segment_length, 0.0)
        delta.rotate(angle)
        waypoints.append((waypoints[-1][0] + delta, speed))

    # segment_length = .5
    # turn_rate = 0.001
    while length < total_length:
        d = length / total_length

        # max_delta_speed = d * 2 * segment_length

        # speed = min(max_speed, max(min_speed, speed + r.gauss(0, d * 5)))
        speed = r.uniform(min_speed, min_speed + (d * (max_speed - min_speed)))
        suggested_turning_radius = ((speed * 3.6) ** 2) / (127 * (0.0 + .12))
        turn_rate = (d * 5) / suggested_turning_radius

        # turn_arc = r.uniform(.1, .75 * 2 * pi)
        turn_length = r.uniform(20, 100)
        turn_direction = 1 if bool(r.getrandbits(1)) else -1
        # turn_rate *= turn_direction

        # remaining_arc = turn_arc
        # while remaining_arc > 1e-3:
        while turn_length > 0:
            # line_angle = min(remaining_arc, .5 * pi / 180.0)
            # remaining_arc -= line_angle
            # max_line_angle = .5 * pi / 180
            line_angle = min(2 * pi / 180, turn_rate * turn_length)

            # line_length = min(turn_length, line_angle / (turn_rate + 1e-6))
            line_length = turn_length if turn_rate < 1e-3 else line_angle / turn_rate
            line_angle *= turn_direction
            add_segment(line_length, line_angle)
            turn_length -= line_length

    def curriculum_angled_trajectory_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
        return deepcopy(waypoints), False

    return curriculum_angled_trajectory_generator


def curriculum_speed_factory(
        min_speed=3,
        max_speed=45,
        min_turn_rate_multiplier=0,
        max_turn_rate_multiplier=2,
        total_length=10000,
        random_seed=2,
        ) -> Callable[['TrajectoryTrackingProcess'], Tuple[List[Tuple[Vec2d, float]], bool]]:
    r = random.Random(random_seed)

    speed = (max_speed + min_speed) / 2

    # start with a straight line

    waypoints = [(Vec2d(0, 0), speed), (Vec2d(.001, 0), speed)]

    length = 0.0
    angle = 0.0

    def add_segment(segment_length, segment_angle):
        nonlocal angle, length, speed
        angle += segment_angle
        length += segment_length

        delta = Vec2d(segment_length, 0.0)
        delta.rotate(angle)
        waypoints.append((waypoints[-1][0] + delta, speed))

    while length < total_length:
        d = length / total_length

        segment_length = 100

        max_delta_speed = d * .5 * (max_speed - min_speed)
        speed = min(max_speed, max(min_speed, speed + r.uniform(-max_delta_speed, max_delta_speed)))
        add_segment(segment_length, 0)

    def curriculum_angled_path_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
        return deepcopy(waypoints), False

    return curriculum_angled_path_generator


def curriculum_trajectory_factory_v3(
        min_speed=3,
        max_speed=45,
        min_turn_rate_multiplier=0,
        max_turn_rate_multiplier=10,
        # min_turn_arc=10.0 * pi / 180,
        # max_turn_arc=180.0 * pi / 180,
        min_turn_length=5,
        max_turn_length=200,
        stage_length=1000,
        num_turning_rate_stages=10,
        transition_length=0.0,
        random_seed=2,
        max_speed_delta=10,
        ) -> Callable[['TrajectoryTrackingProcess'], Tuple[List[Tuple[Vec2d, float]], bool]]:
    def curriculum_trajectory_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
        r = random.Random(random_seed)
        # speed_step = (max_speed - min_speed) / num_speed_stages

        angle = 0.0

        waypoints = [(Vec2d(0, 0), min_speed), (Vec2d(.01, 0), min_speed)]

        '''
        increase curvature
            increasing speeds?
            random speeds

        random speed, curvature, length
        '''
        # length_remaining = stage_length
        # while length_remaining > 0:
        #     # https://www.fhwa.dot.gov/publications/research/safety/17098/004.cfm
        #     # http://dotapp7.dot.state.mn.us/edms/download?docId=1062356
        #
        #     target_speed = random.uniform(min_speed, max_speed)
        #     suggested_turning_radius = ((target_speed * 3.6) ** 2) / (127 * (0.0 + .12))
        #     turn_rate_multiplier = min_turn_rate_multiplier + (t + random.uniform(0, 1)) * turning_rate_step
        #     turn_rate = turn_rate_multiplier / suggested_turning_radius
        #
        #
        turning_rate_step = (max_turn_rate_multiplier - min_turn_rate_multiplier) / num_turning_rate_stages
        target_speed = min_speed
        for t in range(num_turning_rate_stages):
            # https://www.fhwa.dot.gov/publications/research/safety/17098/004.cfm
            # http://dotapp7.dot.state.mn.us/edms/download?docId=1062356

            # num_turning_steps = 1 + int(ceil((min_turning_rate - max_turning_rate) / turning_rate_step))
            # print('num_turning_steps', num_turning_steps, min_turning_rate, mid_turning_rate, max_turning_rate)

            # place straightaway
            if transition_length > 0:
                target_speed = r.uniform(min_speed, max_speed)
                waypoints.append((waypoints[-1][0] + Vec2d(transition_length, 0.0).rotated(angle), target_speed))

            print('stage ', t)
            # place curves
            stage_length_remaining = stage_length
            while stage_length_remaining > 0:
                # target_speed = r.uniform(min_speed, max_speed)
                target_speed = max(min_speed,
                                   min(max_speed, target_speed + r.uniform(-max_speed_delta, max_speed_delta)))
                suggested_turning_radius = ((target_speed * 3.6) ** 2) / (127 * (0.0 + .12))
                turn_rate_multiplier = min_turn_rate_multiplier + (t + r.uniform(0, 1)) * turning_rate_step
                turn_rate = turn_rate_multiplier / suggested_turning_radius

                # turn_arc = random.uniform(min_turn_arc, max_turn_arc)
                turn_direction = 1 if bool(r.getrandbits(1)) else -1

                turn_length = min(stage_length_remaining, r.uniform(min_turn_length, max_turn_length))
                stage_length_remaining -= turn_length
                print('curve ', target_speed, turn_rate_multiplier, turn_rate, turn_length)

                # remaining_arc = turn_arc
                # while remaining_arc > 1e-3:
                while turn_length > 1e-3:
                    # arc = min(remaining_arc, .5 * pi / 180.0)
                    # remaining_arc -= arc
                    arc = .5 * pi / 180
                    line_length = min(turn_length, arc / (turn_rate + 1e-6))
                    line_angle = turn_direction * arc
                    angle += line_angle

                    delta = Vec2d(line_length, 0.0)
                    # print(line_length)
                    delta.rotate(angle)
                    last = waypoints[-1]
                    next = last[0] + delta
                    waypoints.append((next, target_speed))
                    turn_length -= line_length

        # pprint(waypoints)
        # print('length: ', length)
        return waypoints, False

    return curriculum_trajectory_generator


def curriculum_trajectory_factory_v1(
        min_speed=5,
        max_speed=45,
        speed_step=5.0,
        num_turning_rate_steps=5,
        min_turn_rate_multiplier=.5,
        max_turn_rate_multiplier=2.0,
        num_turns_per_step=2,
        min_turn_arc=45 * pi / 180,
        max_turn_arc=180 * pi / 180,
        stage_length_scale=20,
        ) -> Callable[['TrajectoryTrackingProcess'], Tuple[List[Tuple[Vec2d, float]], bool]]:
    def curriculum_trajectory_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:

        length = 0.0
        angle = 0.0
        num_speed_steps = 1 + int(ceil((max_speed - min_speed) / speed_step))
        waypoints = [(Vec2d(0, 0), min_speed), (Vec2d(10.0, 0), min_speed)]
        # print('num_speed_steps', num_speed_steps)
        for s in range(num_speed_steps):
            min_step_speed = s * speed_step + min_speed

            # https://www.fhwa.dot.gov/publications/research/safety/17098/004.cfm
            # http://dotapp7.dot.state.mn.us/edms/download?docId=1062356
            mid_turning_radius = 1.0 * ((min_step_speed * 3.6) ** 2) / (127 * (0.0 + .12))

            mid_turning_rate = 1.0 / mid_turning_radius
            min_turning_rate = min_turn_rate_multiplier * mid_turning_rate
            max_turning_rate = max_turn_rate_multiplier * mid_turning_rate

            # num_turning_steps = 1 + int(ceil((min_turning_rate - max_turning_rate) / turning_rate_step))
            # print('num_turning_steps', num_turning_steps, min_turning_rate, mid_turning_rate, max_turning_rate)
            turning_rate_step = (max_turning_rate - min_turning_rate) / num_turning_rate_steps

            # place straightaway
            target_speed = min_step_speed + random.uniform(0, speed_step)
            straightaway_length = stage_length_scale * target_speed
            waypoints.append(waypoints[-1] + Vec2d(straightaway_length, 0.0).rotated(angle))

            for t in range(num_turning_rate_steps):
                base_turn_rate = min_turning_rate + t * turning_rate_step

                for i in range(num_turns_per_step):
                    target_speed = min_step_speed + random.uniform(0, speed_step)

                    turn_rate = min(max_turning_rate, base_turn_rate + random.uniform(0, turning_rate_step))
                    turn_arc = random.uniform(min_turn_arc, max_turn_arc)

                    # turn_direction = 1 if (i % 2 == 0) else -1
                    turn_direction = 1 if bool(random.getrandbits(1)) else -1

                    # turn_length = turn_rate / turn_arc
                    remaining_arc = turn_arc
                    while remaining_arc > 1e-3:
                        arc = min(remaining_arc, 5.0 * pi / 180.0)
                        remaining_arc -= arc

                        line_length = arc / turn_rate
                        line_angle = turn_direction * arc
                        angle += line_angle

                        delta = Vec2d(line_length, 0.0)
                        # print(line_length)
                        delta.rotate(angle)
                        last = waypoints[-1]
                        next = last[0] + delta
                        waypoints.append((next, target_speed))
                        length += line_length

        # pprint(waypoints)
        # print('length: ', length)
        return waypoints, False

    return curriculum_trajectory_generator


def curriculum_trajectory_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
    # paths with some smooth curving nature
    length = 5000.0
    # target_speed = random.uniform(10, 40)
    target_speed = 10.0

    waypoints = [(Vec2d(0, 0), target_speed)]
    turning_rate = 0.0
    turn_length = 0.0
    angle = 0.0
    distance = 0.0
    while distance < length:
        difficulty = distance / length
        if turn_length <= 0.0:
            if turning_rate != 0:
                turning_rate = 0.0
                turn_length = random.uniform(1, 50.0)
            else:

                min_radius = (4.0 * (1.0 - difficulty) + 1.0 * difficulty) * \
                             (target_speed ** 2) / (15 * (.01 * 0.0 + .21))
                # radians / meter
                max_rate = 1.0 / min_radius
                turning_rate = random.uniform(-1.0, 1.0) * max_rate  # [rad / m]
                target_speed = min(40.0, max(5.0, target_speed + difficulty * random.uniform(-5, 5)))
                # turning_rate = random.uniform(-1.0, 1.0) * (90.0 / target_speed) * (pi / 180.0)  # [rad / m]

                max_turn_length = min(length,
                                      min(200.0,
                                          length if turning_rate == 0.0 else \
                                              (120 * pi / 180.0) / fabs(turning_rate)))
                turn_length = random.uniform(1, max_turn_length)
            # print('seg: ', turning_rate, turn_length, max_turn_length)

        segment_length = turn_length \
            if turning_rate == 0 else \
            min(turn_length, (2 * pi / 180.0) / fabs(turning_rate))

        # segment_length = 1.0

        # segment_angle = (turning_rate + random.gauss(0.0, difficulty * 5.0 * (pi / 180))) * segment_length
        segment_angle = turning_rate * segment_length

        # print(segment_angle, segment_length, turning_rate, turn_length)
        angle += segment_angle
        delta = Vec2d(segment_length, 0.0)
        delta.rotate(angle)

        last = waypoints[-1]
        next = last[0] + delta
        waypoints.append((next, target_speed))
        distance += segment_length

        turn_length -= segment_length

    return waypoints, False


def curriculum_curved_trajectory_factory(
        min_speed=5,
        max_speed=45,
        speed_step=5.0,
        num_turning_rate_steps=5,
        min_turn_rate_multiplier=.5,
        max_turn_rate_multiplier=1.5,
        num_turns_per_step=2,
        min_turn_arc=60 * pi / 180,
        max_turn_arc=120 * pi / 180,
        ) -> Callable[['TrajectoryTrackingProcess'], Tuple[List[Tuple[Vec2d, float]], bool]]:
    def curriculum_curved_trajectory_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:

        length = 0.0
        angle = 0.0
        num_speed_steps = 1 + int(ceil((max_speed - min_speed) / speed_step))
        waypoints = [(Vec2d(0, 0), min_speed), (Vec2d(10.0, 0), min_speed)]
        # print('num_speed_steps', num_speed_steps)
        for s in range(num_speed_steps):
            min_step_speed = s * speed_step + min_speed

            # http://dotapp7.dot.state.mn.us/edms/download?docId=1062356
            mid_turning_radius = 1.0 * ((min_step_speed * 3.6) ** 2) / (127 * (0.0 + .12))

            mid_turning_rate = 1.0 / mid_turning_radius
            min_turning_rate = min_turn_rate_multiplier * mid_turning_rate
            max_turning_rate = max_turn_rate_multiplier * mid_turning_rate

            # num_turning_steps = 1 + int(ceil((min_turning_rate - max_turning_rate) / turning_rate_step))
            # print('num_turning_steps', num_turning_steps, min_turning_rate, mid_turning_rate, max_turning_rate)
            turning_rate_step = (max_turning_rate - min_turning_rate) / num_turning_rate_steps
            for t in range(num_turning_rate_steps):
                base_turn_rate = min_turning_rate + t * turning_rate_step

                for i in range(num_turns_per_step):
                    target_speed = min_step_speed + random.uniform(0, speed_step)
                    turn_rate = min(max_turning_rate, base_turn_rate + random.uniform(0, turning_rate_step))
                    turn_arc = random.uniform(min_turn_arc, max_turn_arc)

                    # turn_direction = 1 if (i % 2 == 0) else -1
                    turn_direction = 1 if bool(random.getrandbits(1)) else -1

                    # turn_length = turn_rate / turn_arc
                    remaining_arc = turn_arc
                    while remaining_arc > 1e-3:
                        arc = min(remaining_arc, 5.0 * pi / 180.0)
                        remaining_arc -= arc

                        line_length = arc / turn_rate
                        line_angle = turn_direction * arc
                        angle += line_angle

                        delta = Vec2d(line_length, 0.0)
                        # print(line_length)
                        delta.rotate(angle)
                        last = waypoints[-1]
                        next = last[0] + delta
                        waypoints.append((next, target_speed))
                        length += line_length

        # pprint(waypoints)
        # print('length: ', length)
        return waypoints, False

    return curriculum_curved_trajectory_generator

def curriculum_angled_trajectory_factory(
        min_speed=5,
        max_speed=45,
        speed_step=4.0,
        num_turning_rate_steps=4,
        min_turn_rate_multiplier=0,
        max_turn_rate_multiplier=1.5,
        # num_turns_per_step=2,
        step_length=300,
        transition_length=10,
        min_turn_arc=60 * pi / 180,
        max_turn_arc=120 * pi / 180,
        random_seed=2,
        ) -> Callable[['TrajectoryTrackingProcess'], Tuple[List[Tuple[Vec2d, float]], bool]]:
    r = random.Random(random_seed)

    # length = 0.0
    angle = 0.0
    num_speed_steps = 1 + int(ceil((max_speed - min_speed) / speed_step))

    previous_speed = min_speed

    # start with a straight line
    # waypoints = [(Vec2d(0, 0), min_speed), (Vec2d(transition_length, 0), min_speed)]
    waypoints = [(Vec2d(0, 0), previous_speed)]

    # print('num_speed_steps', num_speed_steps)
    for s in range(num_speed_steps):
        min_step_speed = s * speed_step + min_speed

        # http://dotapp7.dot.state.mn.us/edms/download?docId=1062356
        mid_turning_radius = 1.0 * ((min_step_speed * 3.6) ** 2) / (127 * (0.0 + .12))

        mid_turning_rate = 1.0 / mid_turning_radius
        min_turning_rate = min_turn_rate_multiplier * mid_turning_rate
        max_turning_rate = max_turn_rate_multiplier * mid_turning_rate

        # num_turning_steps = 1 + int(ceil((min_turning_rate - max_turning_rate) / turning_rate_step))
        # print('num_turning_steps', num_turning_rate_steps)
        turning_rate_step = (max_turning_rate - min_turning_rate) / num_turning_rate_steps
        for t in range(num_turning_rate_steps):
            base_turn_rate = min_turning_rate + t * turning_rate_step

            # for i in range(num_turns_per_step):
            remaining_length = step_length
            while remaining_length > 0:
                target_speed = min_step_speed + r.uniform(0, speed_step)

                # straightaway at beginning of each turn
                line_length = min(transition_length, remaining_length)
                delta = Vec2d(line_length, 0.0)
                delta.rotate(angle)
                next = waypoints[-1][0] + delta
                waypoints.append((next, (target_speed + previous_speed) / 2))
                remaining_length -= line_length

                turn_rate = min(max_turning_rate, base_turn_rate + r.uniform(0, turning_rate_step))
                turn_arc = r.uniform(min_turn_arc, max_turn_arc)
                turn_direction = 1 if bool(r.getrandbits(1)) else -1

                # turn_length = turn_rate / turn_arc
                remaining_arc = turn_arc
                while remaining_arc > 1e-3 and remaining_length > 0:
                    # arc = 0.0
                    # max_arc = 5.0 * pi / 180.0
                    # min_arc = 1.0 * pi / 180.0
                    # if remaining_arc >= max_arc:
                    #     arc = r.uniform(min_arc, max_arc)
                    # else:
                    #     arc = remaining_arc
                    arc = min(remaining_arc, 2.0 * pi / 180.0)
                    remaining_arc -= arc

                    line_length = min(remaining_length, arc / turn_rate)
                    line_angle = turn_direction * arc

                    # if arc >= remaining_arc:
                    line_angle += r.gauss(0.0, 2.0 * pi / 180.0)
                    target_speed = clamp(target_speed + r.gauss(0.0, .5), min_speed, max_speed)
                    angle += line_angle

                    delta = Vec2d(line_length, 0.0)
                    delta.rotate(angle)
                    next = waypoints[-1][0] + delta
                    waypoints.append((next, target_speed))
                    remaining_length -= line_length
                previous_speed = target_speed

    # Plotting curriculum path for jul14_2020_experiments:
    # import matplotlib.pyplot as plt
    # import numpy as np
    # point_locations = np.array([[point[0].x, point[0].y] for point in waypoints])
    # speeds = np.array([elem[1] for elem in waypoints])
    #
    # plt.figure()
    # plt.plot(point_locations[:, 0], point_locations[:, 1], color = 'gray', label = 'path')
    # plt.fill_between(x = point_locations[:, 0], y1 = point_locations[:, 1] - speeds,
    #                         y2 = point_locations[:, 1] + speeds, color = 'gray', label = '2 * target speed',
    #                                 alpha = 0.7)
    # plt.scatter(point_locations[0, 0], point_locations[0, 1], color='b', marker='o', label='start')
    # plt.scatter(point_locations[-1, 0], point_locations[-1, 1], color='r', marker='o', label='end')
    # plt.legend()
    # plt.xlabel('X [m]', fontsize = 14)
    # plt.ylabel('Y [m]', fontsize = 14)
    # plt.title("RL Training Path", fontsize = 24)
    # plt.savefig('curriculum_path.svg', format = 'svg')
    ####
    def curriculum_angled_trajectory_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
        return waypoints, False

    return curriculum_angled_trajectory_generator


def lee_2018_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
    """
        Path found in Lee 2018.
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0194110
    """
    target_speed: float = 20.
    distance_1: float = 200.
    radius_1: float = 200.
    radius_2: float = 100.
    waypoints = []
    waypoints += [(Vec2d(0, 0), target_speed)]
    waypoints += [(Vec2d(distance_1, 0), target_speed)]

    for phase in range(pi / 2., 0, -pi / 1000):
        waypoints += [(Vec2d(radius_1 * cos(phase) - distance_1, \
                             radius_1 * sin(phase) - radius_1), target_speed)]

    for phase in range(pi, 3 * pi / 2., pi / 1000):
        waypoints += [(Vec2d(radius_2 * cos(phase) + \
                             distance_1 + radius_1 + radius_2, \
                             radius_2 * sin(phase) - radius_1))]
    return waypoints


def sine_trajectory_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
    target_speed = 30.
    return [(Vec2d(1000 * (i / 100.0), 20.0 * sin(2 * (i / 100.0) * 2 * pi)), target_speed)
            for i in range(100)], False


def sine_trajectory_factory(
        length: float = 3000.0,
        amplitude: float = 20,
        wavelength: float = 10.0,
        target_speed: float = 10.0,
        ) -> Callable[['TrajectoryTrackingProcess'], Tuple[List[Tuple[Vec2d, float]], bool]]:
    def trajectory_generator(process: 'TrajectoryTrackingProcess'):
        segment_length = wavelength / 2000
        num_segments = ceil(length / segment_length)
        return [(Vec2d(
            length * (float(i / num_segments)),
            amplitude * sin(2 * (float(i / num_segments)) * 2 * pi)),
                 target_speed)
                   for i in range(num_segments)], False

    return trajectory_generator


def carla_json_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
    """
        This will query the file from carla-collect.
        if online, the value for filename will be queried from online
        if online, the value for filename will be from the local files
    """
    import json

    filename: str = 'town01/long_scenario_000.json'
    online: bool = True
    if online:
        import requests

        req = requests.get('https://github.nrel.gov/raw/HPC-CAVs/carla-collect/' \
                           'master/docs/scenarios/' + filename)
        content = req.json()
    else:
        content = json.load(filename)

    start_point = [content['meta']['start_point_x'], content['meta']['start_point_y']]
    path_list = []
    x = []
    y = []
    pairs = []
    target_speed: float = 30.

    for k, seg in enumerate(content['segments']):
        segment_list = seg['segment']
        if len(segment_list) > 2:
            for elem in segment_list:
                if [elem[0], elem[1]] not in pairs:
                    x += [float(elem[0])]
                    y += [float(elem[1])]
                    pairs += [[elem[0], elem[1]]]
                    path_list += [(Vec2d(float(elem[0]),
                                         float(elem[1])), target_speed)]
    # path_list = path_list[:-2]
    # N = 5
    # x_new = []
    # y_new = []

    # enum = 0

    # while enum < len(x) - N:
    #    xx = x[enum:enum+N]
    #    yy = y[enum:enum+N]

    #    tck, u = splprep([np.array(xx), np.array(yy)])

    #    xxx = []
    #    for i in range(len(u) - 1):
    #        xxx += np.linspace(u[i], u[i+1], 4).tolist()

    #    new_points = splev(xxx, tck)

    #    x_new += [new_points[0].tolist()]
    #    y_new += [new_points[1].tolist()]

    #    enum += N

    # x_new = np.array(x_new).reshape(-1, 1)
    # y_new = np.array(y_new).reshape(-1, 1)

    # new_path_list = []
    # for i in range(x_new.shape[0]):
    #    new_path_list += [(Vec2d(x_new[i, 0], y_new[i, 0]), target_speed)]

    return path_list, False


def figure_eight_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
    # TODO: Two circles that touch at intersection
    # find formula on curriculum_angled_path_generator
    length: float = 3000.
    target_speed: float = 15.0
    A: float = 500.
    B: float = 500.
    wavelength: float = target_speed
    segment_length = wavelength / 10.
    num_segments = ceil(length / segment_length)
    return [(Vec2d(
        A * cos(float(i / num_segments) * 2. * pi),
        B * 0.5 * sin(2. * float(i / num_segments) * 2. * pi)),
             target_speed)
               for i in range(num_segments)], False

def hairpin_turn_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
    # TODO: Clean me
    import numpy as np
    length_arm_1: float = 50.
    length_arm_2: float = 50.
    radius_turn: float = 20.  # with code from July 21st lower than this will cause confusion in path finder
    target_speed_1: float = 10.
    target_speed_turn: float = 5.
    target_speed_2: float = 10.
    segment_length = .5
    path = [(Vec2d(0.0, 0.0), target_speed_1), (Vec2d(0.0, length_arm_1), target_speed_turn)]
    center_circle = [radius_turn, length_arm_1]
    starting_angle = np.pi
    end_angle = 0.0
    num_circle_segments = int(ceil(np.pi / 2 * radius_turn) / segment_length)
    angles = np.linspace(starting_angle, end_angle, num_circle_segments)
    for i in range(int(num_circle_segments) - 1):
        angle = angles[i]
        path += [(Vec2d(radius_turn * np.cos(angle) + center_circle[0], radius_turn * np.sin(angle) + center_circle[1]), \
                  target_speed_turn)]
    angle = angles[-1]
    path += [(Vec2d(radius_turn * np.cos(angle) + center_circle[0], radius_turn * np.sin(angle) + center_circle[1]), \
              target_speed_2)]
    path += [(Vec2d(radius_turn * np.cos(angle) + center_circle[0], radius_turn * np.sin(angle) + \
                    center_circle[1] - length_arm_2), target_speed_2)]

    return path, False

def hairpin_turn_flat_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
    # TODO: Clean me
    import numpy as np

    length_arm_1 : float = 50.
    length_arm_2 : float = 50.
    radius_turn : float = 20. # with code from July 21st lower than this will cause confusion in path finder

    target_speed_1 : float = 5.
    target_speed_turn : float = 5.
    target_speed_2 : float = 5.

    segment_length = .5
    path = [(Vec2d(0.0, 0.0), target_speed_1), (Vec2d(0.0, length_arm_1), target_speed_turn)]
    center_circle = [radius_turn, length_arm_1]
    starting_angle = np.pi
    end_angle = 0.0
    num_circle_segments =int(ceil(np.pi/2*radius_turn)/segment_length)
    angles = np.linspace(starting_angle, end_angle, num_circle_segments)
    for i in range(int(num_circle_segments)):
        angle = angles[i]
        path += [(Vec2d(radius_turn*np.cos(angle) + center_circle[0], radius_turn*np.sin(angle) + center_circle[1]),\
                                    target_speed_turn)]
    path += [(Vec2d(radius_turn*np.cos(angle) + center_circle[0], radius_turn *np.sin(angle) + \
                                        center_circle[1] - length_arm_2), target_speed_2)]
    return path, False

def right_turn_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
    # TODO: Clean me
    import numpy as np
    length_arm_1 : float = 50.
    length_arm_2 : float = 50.
    radius_turn : float = 20.

    target_speed_1 : float = 10.
    target_speed_turn : float = 5.
    target_speed_2 : float = 20.

    segment_length = .5
    path = [(Vec2d(0.0, 0.0), target_speed_1), (Vec2d(0.0, length_arm_1), target_speed_turn)]
    center_circle = [radius_turn, length_arm_1]
    starting_angle = np.pi
    end_angle = np.pi / 2.
    num_circle_segments =int(ceil(np.pi/2*radius_turn)/segment_length)
    angles = np.linspace(starting_angle, end_angle, num_circle_segments)
    for i in range(int(num_circle_segments)):
        angle = angles[i]
        if i < int(num_circle_segments) - 1:
            target_sp = target_speed_turn
        else:
            target_sp = target_speed_2
        path += [(Vec2d(radius_turn*np.cos(angle) + center_circle[0], radius_turn*np.sin(angle) + center_circle[1]),\
                                    target_sp)]

    path += [(Vec2d(radius_turn*np.cos(angle) + center_circle[0] + length_arm_2, radius_turn *np.sin(angle) + \
                                        center_circle[1]), target_speed_2)]
    return path, False

def right_turn_flat_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
    # TODO: Clean me
    import numpy as np
    length_arm_1 : float = 50.
    length_arm_2 : float = 50.
    radius_turn : float = 20.

    target_speed_1 : float = 10.
    target_speed_turn : float = 10.
    target_speed_2 : float = 10.

    segment_length = .5
    path = [(Vec2d(0.0, 0.0), target_speed_1), (Vec2d(0.0, length_arm_1), target_speed_turn)]
    center_circle = [radius_turn, length_arm_1]
    starting_angle = np.pi
    end_angle = np.pi / 2.
    num_circle_segments =int(ceil(np.pi/2*radius_turn)/segment_length)
    angles = np.linspace(starting_angle, end_angle, num_circle_segments)
    for i in range(int(num_circle_segments)):
        angle = angles[i]
        if i < int(num_circle_segments) - 1:
            target_sp = target_speed_turn
        else:
            target_sp = target_speed_2
        path += [(Vec2d(radius_turn*np.cos(angle) + center_circle[0], radius_turn*np.sin(angle) + center_circle[1]),\
                                    target_sp)]

    path += [(Vec2d(radius_turn*np.cos(angle) + center_circle[0] + length_arm_2, radius_turn *np.sin(angle) + \
                                        center_circle[1]), target_speed_2)]
    return path, False

def left_lane_change_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
    import numpy as np

    length: float = 100.
    target_speed: float = 10.0
    A: float = 3.5
    wavelength: float = 10.
    segment_length = 0.5
    num_segments = ceil(length / segment_length)
    return [(Vec2d(
        i - num_segments // 2,
        A * np.arctan(i - num_segments // 2)), target_speed)
               for i in range(num_segments)], False


def right_lane_change_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
    import numpy as np

    length: float = 100.
    target_speed: float = 10.0
    A: float = 3.5
    wavelength: float = 10.
    segment_length = 0.5
    num_segments = ceil(length / segment_length)
    return [(Vec2d(
        i - num_segments // 2,
        -A * np.arctan(i - num_segments // 2)), target_speed)
               for i in range(num_segments)], False


def snider_2009_track_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
    from numpy import genfromtxt

    filename = "../stanley_race_track.csv"
    track = genfromtxt(filename, delimiter=',')
    target_speed: float = 20.
    path = [(Vec2d(
        track[i, 0],
        track[i, 1]), target_speed)
        for i in range(track.shape[0])]

    return path, True


def falcone_2007_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
    """
        Falcone 2007: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4162483
        TODO:
            - Stretch according with the target speed
    """
    import numpy as np

    x = np.arange(0, 120., 0.5)
    z1 = (2.4 / 25.) * (x - 27.19) - 1.2
    z2 = (2.4 / 21.95) * (x - 56.46) - 1.2
    dx1 = 25.
    dx2 = 21.95
    dy1 = 4.05
    dy2 = 5.7
    target_speed: float = 17.

    # yaw_ref = dy1/2.*(1.+np.tanh(z1)) - dy2/2.*(1.+np.tanh(z2))
    # y_ref = np.arctan2((1.2*dy1*dx2*np.cosh(z2)**2 - 1.2*dy2*dx1*np.cosh(z1)**2),
    #                        dx1*np.cosh(z1)**2 * dx2*np.cosh(z2)**2)
    # y_ref = 10*np.arctan(dy1*np.square(1./np.cosh(z1)) * 1.2/dx1 -
    #                        dy2*np.square(1./np.cosh(z2))*(1.2/dx2)) + np.pi/2.
    y_ref = dy1 / 2. * (1. + np.tanh(z1)) - dy2 / 2. * (1. + np.tanh(z2))

    path = [(Vec2d(x[i], y_ref[i]), target_speed) for i in range(len(x))]

    return path, False


def square_trajectory_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
    """
        This is a twice differentiable square wave
    """
    delta: float = 0.01
    target_speed = random.uniform(5, 40)
    length: float = 10000.
    amplitude: float = 20
    num_segments = ceil(length / 100.)
    wavelength: float = 100.0
    target_speed: float = 10.0
    frequency = 2.
    return [(Vec2d(
        i,
        (2 * amplitude / pi) * atan(sin(2. * pi * frequency * float(i / num_segments) \
                                        ) / delta)), target_speed) for i in range(num_segments)], False


def circle_trajectory_factory(
        length: float = 3000.0,
        amplitude: float = 500,
        wavelength: float = 100.0,
        target_speed: float = 10.0,
        ) -> Callable[['TrajectoryTrackingProcess'], Tuple[List[Tuple[Vec2d, float]], bool]]:
    def path_generator(process: 'TrajectoryTrackingProcess'):
        import numpy as np

        segment_length = wavelength / 20
        num_segments = ceil(length / segment_length)
        return [(Vec2d(
            amplitude * np.cos(float(i / num_segments) * 2 * pi),
            amplitude * np.sin(float(i / num_segments) * 2 * pi)), target_speed)
                   for i in range(num_segments)], True

    return path_generator


def straight_trajectory_factory(
        length: float = 1000.0,
        target_speed: Optional[float] = None,
        ) -> Callable[['TrajectoryTrackingProcess'], Tuple[List[Tuple[Vec2d, float]], bool]]:
    if target_speed is None:
        target_speed = random.uniform(5, 40)

    def straight_path_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
        return [(Vec2d(0, 0), target_speed), (Vec2d(length, 0), target_speed)], False

    return straight_path_generator


def right_turn_change_speed():
    # TODO: right turn whhere the speed after the turn is lower
    # speed1, speed#2, speed#3
    pass


def straight_variable_speed_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
    length: float = 1000.
    starting_speed: float = 50.
    final_speed: float = 10.
    segment_length = 10.
    num_segments = ceil(length / segment_length)
    return [(Vec2d(i * 10., 0),
             starting_speed + (final_speed - starting_speed) / num_segments * i) for i in range(num_segments)], \
           False


def straight_variable_speed_pulse_generator(process: 'TrajectoryTrackingProcess') -> Tuple[List[Tuple[Vec2d, float]], bool]:
    """
        Speed changes like this:
                 _____________
        _________             _____________
    """
    length: float = 1000.
    starting_speed: float = 10.
    pulse_speed: float = 40.
    segment_length = 10.
    num_segments = ceil(length / segment_length)
    path = []
    speed = []
    for i in range(num_segments):
        if i * segment_length < length / 3.:
            target_speed = starting_speed
        elif i * segment_length > length / 3. and i * segment_length < 2 * length / 3.:
            target_speed = pulse_speed
        elif i * segment_length > 2 * length / 3.:
            target_speed = starting_speed
        path += [(Vec2d(i * 10., 0),target_speed)]
        speed+= [target_speed]
    return path, False

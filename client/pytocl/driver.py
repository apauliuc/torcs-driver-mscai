import logging

import math

from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController

import csv
import numpy as np
from random import choice
from string import ascii_uppercase

_logger = logging.getLogger(__name__)


class Driver:
    """
    Driving logic.

    Implement the driving intelligence in this class by processing the current
    car state as inputs creating car control commands as a response. The
    ``drive`` function is called periodically every 20ms and must return a
    command within 10ms wall time.
    """

    def __init__(self, logdata=True):
        file_path = 'data/csv/default_driver/race_{}.csv'.format(''.join(choice(ascii_uppercase) for i in range(4)))
        self.f = open(file_path, 'a+', encoding='UTF-8')
        self.writer = csv.writer(self.f)
        headers = 'ACCELERATION,BRAKE,STEERING,SPEED,TRACK_POSITION,ANGLE_TO_TRACK_AXIS,' \
                  'TRACK_EDGE_0,TRACK_EDGE_1,TRACK_EDGE_2,TRACK_EDGE_3,TRACK_EDGE_4,TRACK_EDGE_5,' \
                  'TRACK_EDGE_6,TRACK_EDGE_7,TRACK_EDGE_8,TRACK_EDGE_9,TRACK_EDGE_10,TRACK_EDGE_11,' \
                  'TRACK_EDGE_12,TRACK_EDGE_13,TRACK_EDGE_14,TRACK_EDGE_15,TRACK_EDGE_16,' \
                  'TRACK_EDGE_17,TRACK_EDGE_18'.split(',')
        self.writer.writerow(headers)

        self.steering_ctrl = CompositeController(
            ProportionalController(0.4),
            IntegrationController(0.2, integral_limit=1.5),
            DerivativeController(2)
        )
        self.acceleration_ctrl = CompositeController(
            ProportionalController(3.7),
        )
        self.data_logger = DataLogWriter() if logdata else None

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
        self.f.close()

    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """
        command = Command()
        self.steer(carstate, 0.0, command)

        # ACC_LATERAL_MAX = 6400 * 5
        # v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
        v_x = 80

        self.accelerate(carstate, v_x, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        # State: [speedX, speedY, angle, currentGear, RPM, *wheelSpin, *sensorValues]
        # Response: [gear, steering, accelerate, brake]
        # only add data if passed the warm-up stage
        data = np.array([
            command.accelerator,
            command.brake,
            command.steering,
            carstate.speed_x,
            carstate.distance_from_center,
            carstate.angle,
            *carstate.distances_from_edge,
        ])

        # write to file
        self.writer.writerow(data)
        self.f.flush()

        return command

    def accelerate(self, carstate, target_speed, command):
        # compensate engine deceleration, but invisible to controller to
        # prevent braking:
        speed_error = 1.0025 * target_speed * MPS_PER_KMH - carstate.speed_x
        acceleration = self.acceleration_ctrl.control(
            speed_error,
            carstate.current_lap_time
        )

        # stabilize use of gas and brake:
        acceleration = math.pow(acceleration, 3)

        if acceleration > 0:
            if abs(carstate.distance_from_center) >= 1:
                # off track, reduced grip:
                acceleration = min(0.4, acceleration)

            command.accelerator = min(acceleration, 1)

            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1

        # else:
        #     command.brake = min(-acceleration, 1)

        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

    def steer(self, carstate, target_track_pos, command):
        steering_error = target_track_pos - carstate.distance_from_center
        command.steering = self.steering_ctrl.control(
            steering_error,
            carstate.current_lap_time
        )

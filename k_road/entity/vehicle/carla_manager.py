from k_road.entity.vehicle.carla_vehicle import CarlaVehicle


class CarlaManager:

    def __init__(self):
        self.id_counter: int = 0
        self.vehicles: {int: CarlaVehicle} = {}
        self.carla_connection = None

    def make_vehicle(self):
        id = self.id_counter
        self.id_counter += 1
        vehicle = CarlaVehicle(self, id)
        self.vehicles[id] = vehicle
        return vehicle


manager = CarlaManager()


def make_carla_vehicle(self, position, velocity, yaw, angular_velocity):
    return manager.make_vehicle()

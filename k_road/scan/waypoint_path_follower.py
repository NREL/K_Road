from enum import IntEnum

from k_road.scan.pathfinder import *


class Pathfinder(object):
    pass


class WaypointPathFollower:
    class State(IntEnum):
        searching_for_path_start = 0,
        searching_for_nearest_path = 1,
        heading_to_path = 2,
        on_path = 3,
        extending_path = 4,

    def __init__(self,
                 space: pymunk.Space,
                 start: Vec2d,
                 search_radius: float,
                 start_mid_path: bool = False,
                 sequence=None,
                 path: Optional[Path] = None,
                 verbose: bool = False,
                 ) -> None:
        """
        :param space: the pymunk space to search in
        :param start: starting point of the search (starting point of first path)
        :param search_radius: how far from a point to search for a path to follow
        :param start_mid_path: true if it is ok to start in the middle of a path,
                false if only path starts should be considered
        :param sequence: if specified, only paths with the given sequence will be followed
        """
        self.verbose: bool = verbose
        self.space: pymunk.Space = space
        self.state: WaypointPathFollower.State = \
            WaypointPathFollower.State.searching_for_nearest_path \
                if start_mid_path else \
                WaypointPathFollower.State.searching_for_path_start
        self.position: Vec2d = start
        self.search_radius: float = search_radius
        self.sequence = sequence
        self.path: Optional[Path] = path
        self.path_target: Optional[Vec2d] = None
        self.distance_along_path: float = 0.0

    # def get_closest_waypoint(self, target: Vec2d):
    #     """
    #     Tries to find the closest point along the path to the target.
    #     Moves forward from the current position, trying to find the waypoint closest to the target that is
    #     the current waypoint or is in forward along the path from the current waypoint.
    #     The function of distance from the target along the path may not be convex, but
    #     :param target:
    #     :return:
    #     """
    #     pass

    def get_next_waypoint(self, distance_to_next_waypoint: float) -> Tuple['WaypointPathFollower.State', Vec2d]:
        """
        Tries to find the next waypoint along paths, the specified distance along the paths.
        First, it will search for the nearest start of a path and move to it.
        Then, it will traverse along the path until it reaches the end.
        This process is repeated as desired.
        The waypoint along this traversal will be placed to have a total path distance equal to
        distance_to_next_waypoint, so waypoints will not necessarily appear at path ends, and paths shorter than the
        requested distance will be skipped (although they will be followed to the next path).
        :param distance_to_next_waypoint: total distance along paths to this waypoint
        :return: state, waypoint position
        """

        while True:
            self.log('get_next_waypoint() ', self.state, self.position, self.path_target, self.distance_along_path,
                     distance_to_next_waypoint)

            if self.state == WaypointPathFollower.State.searching_for_path_start:

                # find a path that starts near the current position and move to it
                path = (None, None)
                if self.path is not None:
                    path = find_next_path(self.space, self.position, self.search_radius, self.sequence, self.path,
                                          offset=1)
                if path[1] is None:
                    paths = find_nearby_paths(self.space, self.position, self.search_radius, self.sequence,
                                              self.path)
                    if paths:
                        if self.path is None:
                            path = min(paths, key=lambda e: (e[1].start - self.position).length)[1]
                        else:
                            path = min(paths,
                                       default=None,
                                       key=make_path_order_function(self.path))

                if path is None and self.path is not None:
                    # then just continue generating waypoints along the current path
                    self.state = WaypointPathFollower.State.extending_path
                    continue

                self.path = path[1]

                self.path_target = self.path.start
                self.distance_along_path = 0.0
                self.state = WaypointPathFollower.State.heading_to_path

            elif self.state == WaypointPathFollower.State.searching_for_nearest_path:
                # find the nearest path and move to it
                pqi, self.path = find_best_path_hint(self.space, self.position, self.search_radius, self.sequence,
                                                     self.path)

                if pqi is None:
                    # self.waypoint = None
                    break

                # compute distance along path of the destination
                self.path_target = pqi.point
                self.distance_along_path = (self.path_target - self.path.start).length
                self.state = WaypointPathFollower.State.heading_to_path

            elif self.state == WaypointPathFollower.State.heading_to_path:
                # move to the start of the path
                delta = self.path_target - self.position
                distance = delta.length
                if distance_to_next_waypoint <= distance:
                    direction = delta.normalized()
                    self.position = self.position + distance_to_next_waypoint * direction
                    break

                self.log('heading_to_path', distance, self.path.id)
                distance_to_next_waypoint -= distance
                self.position = self.path.start
                self.state = WaypointPathFollower.State.on_path

            elif self.state == WaypointPathFollower.State.on_path:
                distance = self.path.length - self.distance_along_path
                self.log('on_path', self.path.length, self.distance_along_path, distance, self.path.id)

                if distance_to_next_waypoint <= distance:
                    self.distance_along_path += distance_to_next_waypoint
                    self.distance_along_path = float(self.distance_along_path)  # somehow this var is not always a float
                    self.position = self.path.start + self.path.direction * self.distance_along_path
                    break

                distance_to_next_waypoint -= distance
                self.position = self.path.end
                self.state = WaypointPathFollower.State.searching_for_path_start

            elif self.state == WaypointPathFollower.State.extending_path:
                self.distance_along_path += distance_to_next_waypoint
                self.position = self.path.start + self.path.direction * self.distance_along_path
                break

            else:
                break

        return self.state, self.position

    @staticmethod
    def get_waypoints_along_path(path: Path, offset, delta, length=None):
        """
        Makes a sequence of waypoints along a given path
        :param path: path to generate waypoints from
        :param offset: distance along the path the start from
        :param delta: distance between waypoints
        :param length: distance along path to generate, or None to use the path's length
        :return: list of waypoints, remaining uncovered distance (negative if offset > length)
        """
        return WaypointPathFollower.get_waypoints_along_line(
            path.start + offset * path.direction,
            path.direction,
            delta,
            path.length if length is None else length)

    @staticmethod
    def get_waypoints_along_line(start: Vec2d, direction: Vec2d, delta, length):
        """
        Makes a sequence of waypoints along a line
        :param start: starting point
        :param direction: direction to go from the starting point
        :param delta: distance between waypoints
        :param length: distance along path to generate
        :return: list of waypoints, remaining uncovered distance (negative if offset > length)
        """
        distance = 0
        waypoints = []
        while distance <= length:
            waypoints.append(start + direction * distance)
            distance = distance + delta

        return waypoints, length - (distance - delta)

    def log(self, *args) -> None:
        if self.verbose:
            print(*args)

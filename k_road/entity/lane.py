from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# TODO: Finish Lane implementation
# class Lane(Entity):
#     """
#     Represents a lane.
#     Every lane has left and right borders (lane markings) and a path.
#     """
#
#     def __init__(self,
#                  parent: 'KRoadProcess',
#                  left_border: 'LaneMarking',
#                  right_border: 'LaneMarking',
#                  path: 'Path'
#                  ):
#         super().__init__(EntityType.lane, parent, body, )
#         pass
#         #
#         #
#         # left_shape = left_border.body.shapes[0]
#         # shape = pymunk.shapes.Poly()
#         #
#         # super().__init__(EntityType.path, CollisionCategory.area, parent, body, [shape])
#         # self.direction: Vec2d = (end - start).normalized()
#
#     # def render(self, view: 'KRoadView') -> None:
#     #     view.draw_entity(self, (0, 0, 128), True)

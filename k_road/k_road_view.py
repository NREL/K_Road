import math
from typing import (
    TYPE_CHECKING,
    Tuple,
    Union,
)

import pygame
import pygame.gfxdraw
import pymunk
from pymunk import (
    ShapeFilter,
    Vec2d,
)

from factored_gym import Process
from factored_gym import ProcessTimed
from factored_gym import View

if TYPE_CHECKING:
    from k_road.k_road_process import KRoadProcess
from k_road.entity.entity import Entity


class KRoadView(View):

    def __init__(
            self,
            view_size: Union[Vec2d, Tuple[int, int]],
            view_center: Vec2d,
            view_scale: float,
            time_dilation: float):
        print('pygame display init')
        pygame.display.quit()
        pygame.init()  # only works when we are only using pygame once here... this is a static global function

        self._view_size: (int, int) = (int(math.ceil(view_size[0])), int(math.ceil(view_size[1])))

        self._view_scale: float = view_scale
        self._time_dilation: float = time_dilation

        self._view_world_size = Vec2d(self._view_size[0], self._view_size[1]) / self._view_scale
        self.set_view_center(view_center)

        self._surface: pygame.Surface = pygame.display.set_mode(self._view_size, pygame.RESIZABLE)
        self._clock: pygame.time.Clock = pygame.time.Clock()

    def reset(self, process: ProcessTimed) -> None:
        self._surface.fill((0, 0, 0))

    def begin_render(self, process: ProcessTimed, view: View) -> None:
        self._surface.fill((240, 240, 240))  # set background
        # get all objects in frame
        view_position = self._view_position
        view_world_size = self._view_world_size
        # BB(left, bottom, right, top)
        shapes = process.space.bb_query(pymunk.BB(
            view_position[0], view_position[1],
            view_position[0] + view_world_size[0], view_position[1] + view_world_size[1]),
            ShapeFilter())

        entities = [shape.body.entity for shape in shapes if shape.body is not None and shape.body.entity is not None]

        # render objects in entity type order
        # TODO: make plane ordering more sensible
        render_planes = {}
        for entity in entities:
            plane = entity.type_
            if plane in render_planes:
                render_planes[plane].append(entity)
            else:
                render_planes[plane] = [entity]

        for plane in sorted(render_planes.keys(), reverse=True):
            for entity in render_planes[plane]:
                entity.render(self)

    def render(self, process: 'KRoadProcess', view: View) -> None:
        pygame.event.pump()
        pygame.display.flip()  # update display
        if self._time_dilation != 0:  # limit framerate based on step size and time dilation
            self._clock.tick(self._time_dilation / process.time_step_length)

    def close(self, process: Process) -> None:
        """Try to clean up pygame resources."""
        pygame.display.quit()

    # def draw_segment(self, entity: Entity, color) -> None:
    #     pygame.draw.aalines(self.surface, color, False,
    #                         [self.convert_shape_vector(entity, entity.shape.a),
    #                          self.convert_shape_vector(entity, entity.shape.b)],
    #                         )
    #     # int(max(1, self.scale(entity.shape.radius * 2)
    #
    # def draw_polygon(self, entity: Entity, shape, outline_color) -> None:
    #     if shape is None:
    #         shape = entity.shape
    #
    #     # fill_color = outline_color if fill_color is None else fill_color
    #     vertices = [self.convert_shape_vector(entity, vertex) for vertex in shape.get_vertices()]
    #     # radius = shape.radius
    #     # pygame.draw.aalines(self.surface, fill_color, vertices)
    #     pygame.draw.aalines(self.surface, outline_color, True, vertices)

    def scale(self, vec: Union[Vec2d, float, int]) -> Union[Vec2d, float, int]:
        """Scales a relative simulator vector into a relative 148 vector."""
        return vec * self._view_scale

    def convert(self, vec: Vec2d) -> Vec2d:
        """Converts the given simulation vector into a surface vector"""
        vec = (vec - self._view_position) * self._view_scale
        return Vec2d(int(vec[0]),
                     int(self._view_size[1] - vec[1]))  # flip y-coordinate (view is y-down, model is y-up)

    def convert_shape_vector(self, body: pymunk.Body, vector: Vec2d) -> Vec2d:
        """Converts a shape vector belonging to the given body into a surface vector."""
        # return self.convert(entity.body.position + vector.rotated(entity.body.angle))
        return self.convert(body.local_to_world(vector))

    def set_view_center(self, view_center: Vec2d) -> None:
        """
        Sets the viewport position
        """
        self._view_position: Vec2d = view_center - .5 * self._view_world_size

    def set_viewport(self, view_position: Vec2d, view_scale: float, time_dilation: float) -> None:
        """
        Sets the viewport position, scale, and time dilation.
        """
        self._view_position = view_position
        self._view_scale = view_scale
        self._time_dilation = time_dilation

    def draw_entity(self,
                    entity: Entity,
                    color=(128, 128, 128),
                    filled: bool = False):
        for body in entity.bodies:
            for shape in body.shapes:
                self.draw_shape_on_body(body, shape, color, filled)

    def draw_shape_on_entity(self,
                             entity: Entity,
                             shape: pymunk.Shape,
                             color=(128, 128, 128),
                             filled: bool = False) -> None:
        self.draw_shape_on_body(entity.body, shape, color, filled)

    def draw_shape_on_body(self,
                           body: pymunk.Body,
                           shape: pymunk.Shape,
                           color=(128, 128, 128),
                           filled: bool = False) -> None:
        if isinstance(shape, pymunk.Circle):
            self.draw_shape_circle(body, shape, color, filled)
        elif isinstance(shape, pymunk.Poly):
            self.draw_shape_polygon(body, shape, color, filled)
        elif isinstance(shape, pymunk.Segment):
            self.draw_shape_segment(body, shape, color, filled)
        else:
            assert False

    def draw_shape_circle(self,
                          body: pymunk.Body,
                          circle: pymunk.Circle,
                          color=(128, 128, 128),
                          filled: bool = False) -> None:
        self.draw_circle_on_body(body, color, filled, circle.offset, circle.radius)

    def draw_circle_on_entity(self,
                              entity: Entity,
                              color,
                              filled: bool,
                              offset: Vec2d,
                              radius: float) -> None:
        self.draw_circle_on_body(entity.body, color, filled, offset, radius)

    def draw_circle_on_body(self,
                            body: pymunk.Body,
                            color,
                            filled: bool,
                            offset: Vec2d,
                            radius: float) -> None:
        position = self.convert_shape_vector(body, offset)
        radius = math.ceil(self.scale(radius))
        if filled:
            pygame.draw.circle(self._surface, color, position, radius)
        else:
            pygame.gfxdraw.aacircle(self._surface, position.x, position.y, radius, color)

    def draw_circle(self, color, position: Vec2d, radius: float) -> None:
        position = self.convert(position)
        pygame.gfxdraw.aacircle(self._surface, position.x, position.y, math.ceil(self.scale(radius)), color)

    def draw_shape_polygon(self,
                           body: pymunk.Body,
                           poly: pymunk.Poly,
                           color,
                           filled: bool = False) -> None:
        self.draw_polygon_on_body(body, color, filled, poly.get_vertices())

    def draw_polygon_on_entity(self,
                               entity: Entity,
                               color,
                               filled: bool,
                               vertices: [Vec2d]):
        self.draw_polygon_on_body(entity.body, color, filled, vertices)

    def draw_polygon_on_body(self,
                             body: pymunk.Body,
                             color,
                             filled: bool,
                             vertices: [Vec2d]):
        vertices = [self.convert_shape_vector(body, vertex) for vertex in vertices]
        if filled:
            pygame.draw.polygon(self._surface, color, vertices)
        else:
            pygame.draw.aalines(self._surface, color, True, vertices)

    def draw_shape_segment(self,
                           body: pymunk.Body,
                           segment: pymunk.Segment,
                           color,
                           filled: bool = False) -> None:
        self.draw_segment_on_body(body, color, filled, segment.a, segment.b, segment.radius)

    def draw_segment_on_entity(self,
                               entity: Entity,
                               color,
                               filled: bool,
                               a: Vec2d,
                               b: Vec2d,
                               radius: float
                               ):
        self.draw_segment_on_body(entity.body, color, filled, a, b, radius)

    def draw_segment_on_body(self,
                             body: pymunk.Body,
                             color,
                             filled: bool,
                             a: Vec2d,
                             b: Vec2d,
                             radius: float
                             ):
        a = body.local_to_world(a)
        b = body.local_to_world(b)
        self.draw_segment(color, filled, a, b, radius)

    def draw_line(self,
                  color,
                  a: Vec2d,
                  b: Vec2d,
                  ) -> None:
        pygame.draw.aaline(self._surface, color, self.convert(a), self.convert(b))

    def draw_segment(self,
                     color,
                     filled: bool,
                     a: Vec2d,
                     b: Vec2d,
                     radius: float) -> None:
        a = self.convert(a)
        b = self.convert(b)
        radius = self.scale(radius)
        if radius <= 1:
            pygame.draw.aaline(self._surface, color, a, b)
        else:
            radius = math.ceil(radius)
            if filled:
                pygame.draw.circle(self._surface, color, a, radius)
                pygame.draw.circle(self._surface, color, b, radius)
                pygame.draw.aaline(self._surface, color, a, b)
            else:
                normal = (b - a).normalized().rotated(math.pi / 2)
                offset = normal * radius
                pygame.draw.circle(self._surface, color, a, radius, 1)
                pygame.draw.circle(self._surface, color, b, radius, 1)
                pygame.draw.aaline(self._surface, color, a + offset, b + offset)
                pygame.draw.aaline(self._surface, color, a - offset, b - offset)

    def draw_shape_segment_dashed(self,
                                  body: pymunk.Body,
                                  segment: pymunk.Segment,
                                  color,
                                  filled: bool,
                                  dash_length: float,
                                  dash_proportion: float = .5,
                                  dash_offset: float = 0.0) -> None:
        self.draw_segment_on_body_dashed(body, color, filled, segment.a, segment.b, segment.radius, dash_length,
                                         dash_proportion, dash_offset)

    def draw_segment_on_entity_dashed(self,
                                      entity: Entity,
                                      color,
                                      filled: bool,
                                      a: Vec2d,
                                      b: Vec2d,
                                      radius: float,
                                      dash_length: float,
                                      dash_proportion: float = .5,
                                      dash_offset: float = 0.0) -> None:
        self.draw_segment_on_body_dashed(entity.body, color, filled, a, b, radius, dash_length, dash_proportion,
                                         dash_offset)

    def draw_segment_on_body_dashed(self,
                                    body: pymunk.Body,
                                    color,
                                    filled: bool,
                                    a: Vec2d,
                                    b: Vec2d,
                                    radius: float,
                                    dash_length: float,
                                    dash_proportion: float = .5,
                                    dash_offset: float = 0.0) -> None:
        a = body.local_to_world(a)
        b = body.local_to_world(b)
        self.draw_segment_dashed(color, filled, a, b, radius, dash_length, dash_proportion, dash_offset)

    def draw_segment_dashed(self,
                            color,
                            filled: bool,
                            a: Vec2d,
                            b: Vec2d,
                            radius: float,
                            dash_length: float,
                            dash_proportion: float = .5,
                            dash_offset: float = 0.0) -> None:
        delta = (b - a)
        length = delta.get_length()
        direction = delta.normalized()
        drawn_dash_length = dash_length * dash_proportion

        offset = dash_offset
        while offset < length:
            drawn_dash_extent = min(offset + drawn_dash_length, length)
            dash_start = a + offset * direction
            dash_end = a + drawn_dash_extent * direction
            self.draw_segment(color, filled, dash_start, dash_end, radius)
            offset = offset + dash_length

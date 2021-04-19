import math
from functools import reduce
from typing import Iterable

import pymunk
from pymunk import Vec2d

from factored_gym import ProcessTimed
from k_road.entity.entity import Entity
from k_road.entity.entity_category import (
    CollisionMasks,
    EntityCategory,
)
from k_road.k_road_view import KRoadView


class KRoadProcess(ProcessTimed):

    def __init__(self,
                 time_step_length: float = .05,
                 time_dilation: float = 2.0,
                 max_viewport_resolution: (int, int) = (1500, 1000),
                 viewport_padding: float = .025
                 ):
        self.__time_step_length: float = time_step_length
        self.__time_dilation: float = time_dilation
        self.__max_viewport_resolution: (int, int) = max_viewport_resolution
        self.__viewport_padding: float = viewport_padding

        self.__space: pymunk.Space = pymunk.Space()
        self.__space.gravity = Vec2d(0, 0)
        self.__space.iterations = 1

        self.__entity_count: int = 0
        self.__entities: {int: Entity} = {}
        self.__dynamic_entities: {int, Entity} = {}

        self.__body_count: int = 0

        self.__shape_count: int = 0

        self.__step_number: int = 0

        default_collision_handler = self.__space.add_default_collision_handler()
        default_collision_handler.begin = lambda arbiter, space, data: self.__handle_collision(arbiter, space, data)

    def make_view(self, mode) -> KRoadView:
        """
        Factory method from Process class that returns a KRoadView.
        This implementation puts the view over all shapes in the KRoadProcess, plus a bit of padding.
        Override this to get a different viewport or to use a different view type.
        """
        left, bottom, right, top = self.bounding_box

        # pad bounds by a little bit
        padding = self.__viewport_padding
        horizontal_padding = padding * (right - left)
        vertical_padding = padding * (top - bottom)
        left = left - horizontal_padding
        right = right + horizontal_padding
        bottom = bottom - vertical_padding
        top = top + vertical_padding

        center = Vec2d((left + right) / 2, (top + bottom) / 2)

        horizontal_size = right - left
        vertical_size = top - bottom

        scale = min(
            self.__max_viewport_resolution[0] / horizontal_size,
            self.__max_viewport_resolution[1] / vertical_size)

        view_size = (math.ceil(scale * horizontal_size), math.ceil(scale * vertical_size))

        return KRoadView(
            view_size,
            center,
            scale,
            self.__time_dilation)

    @property
    def bounding_box(self) -> (float, float, float, float):
        """
        Scans all shapes and returns a bounding box that covers all of them.
        :return left, bottom, right, and top bounds
        """
        bbs = [(bb.left, bb.bottom, bb.right, bb.top) for bb in (shape.bb for shape in self.shapes)]
        if len(bbs) <= 0:
            return -100.0, -100.0, 100.0, 100.0
        return reduce(
            lambda a, b: (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])),
            bbs)

    def step_physics(self) -> None:
        """
        Steps the entity and physics simulation one timestep without stepping anything else.
        """
        self.__step_number = self.__step_number + 1
        for entity in self.__dynamic_entities.values():
            entity.step()
        self.__space.step(self.__time_step_length)

    @property
    def entities(self) -> Iterable[Entity]:
        return self.__entities.values()

    @property
    def dynamic_entities(self) -> Iterable[Entity]:
        return self.__dynamic_entities.values()

    @property
    def bodies(self) -> Iterable[pymunk.Body]:
        return [body for entity in self.entities for body in entity.bodies]

    @property
    def shapes(self) -> Iterable[pymunk.Shape]:
        return [shape for body in self.bodies for shape in body.shapes]

    @property
    def time_dilation(self) -> float:
        return self.__time_dilation

    def add_entity(self, entity: Entity) -> None:
        """
        Registers an entity and its shapes with this process. Do not call this directly; it is called in the Entity
        constructor.
        """

        entity.id = self.__entity_count
        self.__entity_count = self.__entity_count + 1
        self.__entities[entity.id] = entity
        # print('add entity ', entity.id, entity.type_, entity.category)
        for body in entity.bodies:
            body.entity = entity  # allows us to get the entity from the body (as returned by queries and collisions)
            self.add_body(body)

    def add_body(self, body: pymunk.Body):
        body.id = self.__body_count
        self.__body_count = self.__body_count + 1

        self.__space.add(body)
        for shape in body.shapes:
            self.add_shape(shape)
        # print('add_body ', body.id, body.entity.id)

    def add_shape(self, shape: pymunk.Shape) -> None:
        shape.id = self.__shape_count
        self.__shape_count = self.__shape_count + 1

        entity_category = shape.body.entity.category
        shape.filter = pymunk.ShapeFilter(categories=entity_category, mask=CollisionMasks[entity_category])

        self.__space.add(shape)
        # shape.cache_bb()
        # print('add_shape ', shape.id, shape.body.id, shape.body.entity.id)

    def discard_entity(self, entity: Entity) -> None:
        """
        Discards an entity from this process
        """
        for body in entity.bodies:
            self.discard_body(body)

        if entity.id in self.__dynamic_entities:
            del self.__dynamic_entities[entity.id]

        del self.__entities[entity.id]

    def discard_body(self, body: pymunk.Body) -> None:
        for shape in body.shapes:
            self.discard_shape(shape)
        self.__space.remove(body)

    def discard_shape(self, shape: pymunk.Shape) -> None:
        self.__space.remove(shape)

    def set_dynamic(self, entity: Entity) -> None:
        self.__dynamic_entities[entity.id] = entity

    def can_place_entity(self, entity, expansion: float = 0.0) -> bool:
        """
        Checks if an entity could be placed without colliding with other entities.
        Currently designed to only work with polygonal entities belonging to CollisionCategory.dynamic.
        """
        assert entity.category == EntityCategory.dynamic, 'only dynamic entities are supported'

        space = self.space

        for body in entity.bodies:
            for shape in entity.shapes:
                assert isinstance(shape, pymunk.Poly), 'Only polygon based entities are supported.'

                # vertices = [body.local_to_world(vert) for vert in shape.get_vertices()]
                # print('check ', entity.position, [body.local_to_world(vert) for vert in shape.get_vertices()])
                # poly = pymunk.Poly(None, vertices, radius=shape.radius * 10)
                poly = pymunk.Poly(None, shape.get_vertices(), radius=shape.radius + expansion)

                c = math.cos(body.angle)
                s = math.sin(body.angle)
                poly.update(pymunk.Transform(c, s, -s, c, body.position.x, body.position.y))

                # print('check ', poly.radius, shape.get_vertices(), poly.get_vertices())
                # poly = pymunk.Poly(None, shape.get_vertices(), radius=0.0)
                # shape.cache_bb()
                # sqis2 = space.shape_query(shape)
                sqis = space.shape_query(poly)

                # if len(sqis) < len(sqis2):
                #     print('what ', sqis, sqis2)

                for sqi in sqis:
                    other_entity = sqi.shape.body.entity
                    if other_entity == entity:
                        continue
                    entity_category = other_entity.category
                    if entity_category == EntityCategory.dynamic:  # or entity_category == EntityCategory.off_road:
                        # print('can\'t place: ', other_entity.type_, other_entity.category, other_entity.position,
                        # entity.position)
                        return False
                # print('pass')
        return True

    def reset(self, process: 'KRoadProcess') -> None:
        """
        Resets this space.
        The user is responsible for discarding or resetting entities and shapes.
        """
        self.__step_number = 0

    def step(self, action) -> None:
        self.step_physics()

    @property
    def time_step_number(self) -> int:
        return self.__step_number

    @property
    def time(self) -> float:
        return self.__step_number * self.__time_step_length

    @property
    def time_step_length(self) -> float:
        return self.__time_step_length

    def render(self, process: 'KRoadProcess', view: KRoadView) -> None:
        super().render(process, view)

    @property
    def space(self) -> pymunk.Space:
        return self.__space

    def __discard_shape(self, shape) -> None:
        self.__space.remove(shape)

    def __handle_collision(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data) -> bool:
        entity_a = arbiter.shapes[0].body.entity
        entity_b = arbiter.shapes[1].body.entity

        # print('__handle_collision ',
        #       arbiter.shapes[0].id, arbiter.shapes[1].id,
        #       arbiter.shapes[0].body.id, arbiter.shapes[1].body.id,
        #       arbiter.shapes[0].body.entity.id, arbiter.shapes[1].body.entity.id
        #       )
        a = entity_a.handle_collision(arbiter, entity_b)
        b = entity_a.live and entity_b.live and entity_b.handle_collision(arbiter, entity_a)
        return a and b

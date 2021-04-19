from typing import Dict

import pymunk
from pymunk import Vec2d

from k_road.entity.entity_category import EntityCategory
from scenario import RayScanResult


class RayScanner:

    def __init__(self, distance: float, total_scanned_angle: float, beam_radius: float, resolution: int):
        self.distance: float = distance
        self.total_scanned_angle: float = total_scanned_angle
        self.beam_radius: float = beam_radius
        self.resolution: int = resolution
        self.scan_step_angle = total_scanned_angle / (resolution - 1) if total_scanned_angle > 0 else 0
        self.shape_filter: pymunk.shape_filter.ShapeFilter = \
            pymunk.ShapeFilter(categories=EntityCategory.sensor)

    def scan(self, position: Vec2d, start_angle: float, space: pymunk.Space, result_filter=lambda rsr: True):
        contacts: [(Vec2d, RayScanResult)] = []

        # alpha_correction_factor = self.beam_radius / self.distance

        for i in range(self.resolution):
            scan_angle = start_angle + self.scan_step_angle * i
            scan_endpoint = position + Vec2d(self.distance, 0).rotated(scan_angle)
            # self.scan_endpoints.append(scan_endpoint)
            segment_query_infos: [pymunk.SegmentQueryInfo] = \
                space.segment_query(position, scan_endpoint, self.beam_radius, self.shape_filter)

            segment_contacts: [RayScanResult] = []
            for segment_query_info in segment_query_infos:
                shape = segment_query_info.shape
                if shape is None or shape.sensor:
                    continue

                # corrected_alpha = min(1.0, segment_query_info.alpha + alpha_correction_factor)

                entity = shape.body.entity
                rsr = RayScanResult(
                    segment_query_info.alpha,
                    entity.type_,
                    entity.id,
                    segment_query_info.point,
                    segment_query_info.normal,
                    shape,
                    entity
                )

                if result_filter(rsr):
                    segment_contacts.append(rsr)
            contacts.append((scan_endpoint, segment_contacts))
        return contacts

    @staticmethod
    def get_closest_of_each_type(segment_contacts: [RayScanResult]) -> Dict[int, RayScanResult]:
        closest = {}
        for contact in segment_contacts:
            type_ = contact.type_
            if type_ not in closest or closest[type_].alpha > contact.alpha:
                closest[type_] = contact
        return closest

    def scan_closest_of_each_type(self, position: Vec2d, start_angle: float, space: pymunk.Space,
                                  result_filter=lambda rsr: True) -> \
            [(Vec2d, Dict[int, RayScanResult])]:
        scan_results = self.scan(position, start_angle, space, result_filter)
        return [(scan_result[0], self.get_closest_of_each_type(scan_result[1])) for scan_result in scan_results]

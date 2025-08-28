from collections import Counter
from pathlib import Path
from typing import Dict, List, Union, Tuple

import lanelet2
import numpy as np
from lanelet2.io import Origin
from lanelet2.matching import getDeterministicMatches, getProbabilisticMatches, removeNonRuleCompliantMatches
from lanelet2.projection import UtmProjector
from shapely.geometry import LineString
from lanelet2.core import Lanelet
from lanelet2.routing import RoutingGraph, LaneletRelation

from .anchor_generation import Anchor, create_anchors_for_lanelet
from .anchor_generation.discover_anchors import discover_lanelets
from .anchor_tools.interpolate_lanelet import interpolate_lanelet
from .anchor_tools.lanelet_matching import (
    LaneletAnchorMatches,
    LaneletMatchingConfig,
    LaneletMatchingProbConfig,
    LaneletMatchProb,
    VehiclePose,
    _convert_distances_to_probabilities,
)

# TODO remove matplotlib dependency when importing this file -> not needed for package installation (only examples)


class AnchorGenerator:
    """Represents a Lanelet2 map provided by an osm file and provides anchor generation methods."""

    def __init__(
        self,
        osm_file: Union[Path, str],
        origin_latitude: float,
        origin_longitude: float,
    ) -> None:
        origin = Origin(origin_latitude, origin_longitude)
        projector = UtmProjector(origin)
        self.traffic_rules = lanelet2.traffic_rules.create(
            lanelet2.traffic_rules.Locations.Germany,
            lanelet2.traffic_rules.Participants.Vehicle,
        )

        self.lanelet_map = lanelet2.io.load(str(osm_file), projector)
        self.routing_graph = lanelet2.routing.RoutingGraph(
            self.lanelet_map, self.traffic_rules
        )
        self.matching_config: Union[
            LaneletMatchingConfig, LaneletMatchingProbConfig
        ] = LaneletMatchingProbConfig()

    @property
    def lanelet_ids(self) -> List[int]:
        """Listing of the lanelet ids in given map.

        Returns:
            List[int]: List of existing lanelet ids.
        """
        return [lanelet.id for lanelet in self.lanelet_map.laneletLayer]

    def create_anchors_for_lanelet(
        self,
        lanelet_id: int,
        anchor_length: float = 100,
        distance_method: str = "iou",
    ) -> List[Anchor]:
        """Creates anchors for a given lane. All paths from the start lane are considered until the max_length is reached (start lanelet not included). The second path exploration limit is given by the lane chnage type: once a vehicle changed left, it cannot change right anymore and vice versa.

        Args:
            lanelet_id (int): Identifier of the startpoint lanelet. The starting lanelet will always be included to the anchor.
            anchor_length (float, optional): Desired length of the anchor in meters. Start lanelet is NOT included. Note: Anchors can have a length shorter than length, if there is a dead end. Defaults to 100.
            distance_method (str, optional):  Method to calculate the distance between two centerlines. Available methods are iou, dtw, and hausdorff. Defaults to "iou".

        Returns:
            List[Anchor]: Sorted list of all anchors. The most important anchor is at index 0.
        """
        return create_anchors_for_lanelet(
            lanelet_map=self.lanelet_map,
            routing_graph=self.routing_graph,
            lanelet_id=lanelet_id,
            max_length=anchor_length,
            distance_method=distance_method,
        )

    def get_reachable_lanelets_from(
        self,
        lanelet_id: int,
        max_length: float = 100,
    ) -> List[Lanelet]:
        start_lanelet = self.lanelet_map.laneletLayer[lanelet_id]
        reachable_lanelets = discover_lanelets(
            routing_graph=self.routing_graph,
            start_lanelet=start_lanelet,
            max_length=max_length,
        )

        return reachable_lanelets

    def interpolate_lanelet(
        self,
        lanelet_id: int,
        ratio: float,
        num_points: int = 100,
    ) -> LineString:
        """Interpolates lanelet at given ratio along its width.

        Args:
            lanelet_id (int): Identifier of the startpoint lanelet. The starting lanelet will always be included to the anchor.
            ratio (float): Interpolation ratio betweeen 0 and 1. 0 corresponds to right border, 1 to left border.
            num_points (int, optional): Number of points used for interpolation. Defaults to 100.

        Returns:
            LineString: Returns the lanelet interpolation at given ratio.
        """
        return interpolate_lanelet(
            lanelet_map=self.lanelet_map,
            lanelet_id=lanelet_id,
            ratio=ratio,
            num_points=num_points,
        )

    def match_vehicle_onto_lanelets_deterministically(
        self,
        vehicle_pose: VehiclePose,
        max_dist_to_lanelet: float = 0.5,
    ) -> Dict[str, LaneletMatchProb]:
        """Match vehicle onto lanelet deterministically using Lanelet2 Matching.

        Args:
            vehicle_pose (VehiclePose): Pose of vehicle
            max_dist_to_lanelet (float, optional): Euclidean distance to which we find lanelets

        Returns:
            Dict[str, LaneletMatchProb]: Mapping between lanelet ID and the lanelet match
        """
        mapping = {}
        lanelet_matches = getDeterministicMatches(
            self.lanelet_map,
            vehicle_pose.as_object2d(),
            np.double(max_dist_to_lanelet),
        )

        # todo: fix getting unique matches only
        lanelet_matches = list({m.lanelet.id: m for m in lanelet_matches}.values())

        if len(lanelet_matches) != 0:
            # Compute Matching Probabilities
            probs = _convert_distances_to_probabilities(
                distances=np.array([m.distance for m in lanelet_matches])
            )
            mapping = {
                match.lanelet.id: LaneletMatchProb(match, prob)
                for match, prob in zip(lanelet_matches, probs)
                if prob > 0.001
            }
        return mapping

    def match_vehicle_onto_lanelets_probabilistically(
        self,
        vehicle_pose: VehiclePose,
        max_dist_to_lanelet: float = 0.5,
        remove_non_rule_compliant_matches: bool = False,
        debug: bool = False,
    ) -> Dict[str, LaneletMatchProb]:
        """Match vehicle onto lanelet probabilistically using Lanelet2 Matching.

        Args:
            vehicle_pose (VehiclePose): Pose of vehicle

        Returns:
            Dict[str, LaneletMatchProb]: Mapping between lanelet ID and the lanelet match
        """
        mapping = {}

        lanelet_matches = getProbabilisticMatches(
            self.lanelet_map,
            vehicle_pose.as_object2d_with_covariance(self.matching_config),
            np.double(max_dist_to_lanelet),
        )
        if remove_non_rule_compliant_matches:
            # This is not reliable, sometimes it fixes the issue, but most of the time, it's just adding noise to the lanelet matching
            lanelet_matches = removeNonRuleCompliantMatches(lanelet_matches, self.traffic_rules)

        if len(lanelet_matches) != 0:
            # Compute Matching Probabilities
            probs = _convert_distances_to_probabilities(
                distances=np.array([m.mahalanobisDistSq for m in lanelet_matches])
            )
            mapping = {
                match.lanelet.id: LaneletMatchProb(match, prob)
                for match, prob in zip(lanelet_matches, probs)
                if prob > 0.001
            }
        if debug:
            print(f'match_vehicle_onto_lanelets_probabilistically, mapping:')
            for ll_match in mapping.values():
                print(f'\t{ll_match.lanelet}')
        return mapping

    def create_anchors_for_vehicle(
        self,
        vehicle_pose: VehiclePose,
        anchor_length: float = 100,
        num_anchors: int = 5,
        probabilitisc_matching: bool = True,
        max_dist_to_lanelet: float = 0.5,
    ) -> List[LaneletAnchorMatches]:
        """Compute diverse map based anchor paths by first matching the vehicle onto the Lanelet map and subsequently generating and filtering anchor paths.

        Args:
            vehicle_pose (VehiclePose): Position and orientation of the considered vehicle
            anchor_length (float, optional): Desired length of the anchor in meters. Start lanelet is NOT included. Note: Anchors can have a length shorter than length, if there is a dead end. Defaults to 100.
            num_anchors (int, optional): Number of anchor paths that should be computed. Defaults to 5.
            probabilitisc_matching (bool, optional): Whether the matching of the vehicle onto the Lanelet is probabilistic or deterministic. Defaults to True.
            max_dist_to_lanelet (float, optional): Euclidean distance to which we find lanelets

        Returns:
            Dict[str, LaneletAnchorMatches]: Mapping between the start Lanelet ID and the computed anchor paths
        """
        if probabilitisc_matching:
            lanelet_matches = self.match_vehicle_onto_lanelets_probabilistically(
                vehicle_pose,
                max_dist_to_lanelet
            )
        else:
            lanelet_matches = self.match_vehicle_onto_lanelets_deterministically(
                vehicle_pose,
                max_dist_to_lanelet
            )
        lanelet_probs = np.asarray([m.probability for m in lanelet_matches.values()])
        if len(lanelet_probs) == 0:
            return []
        samples = Counter(
            np.random.choice(
                list(lanelet_matches.keys()),
                size=num_anchors,
                p=list(lanelet_probs / np.sum(lanelet_probs)),
            )
        )
        anchor_paths = []
        for ll_id, num_anchors in dict(samples).items():
            ll_anchors = self.create_anchors_for_lanelet(
                lanelet_id=int(ll_id),
                anchor_length=anchor_length,
            )

            anchor_paths.append(
                LaneletAnchorMatches(
                    lanelet_match=lanelet_matches[ll_id],
                    anchors=ll_anchors,
                    selection=[True] * min(num_anchors, len(ll_anchors))
                    + [False] * max(0, len(ll_anchors) - num_anchors),
                )
            )
        return anchor_paths

    @staticmethod
    def prediction_to_vehicle_poses(
            ego_info,
            prediction, # [H, 2]
    ) -> List[VehiclePose]:
        def rotate_bbox(bbox_points, center_old, psi_old, center_new, psi_new):
            # Translate bbox to origin
            shifted = bbox_points - center_old

            # Undo original rotation
            c0, s0 = np.cos(-psi_old), np.sin(-psi_old)
            R0_inv = np.array([[c0, -s0],
                               [s0, c0]])
            unrotated = shifted @ R0_inv.T

            # Apply new rotation
            c1, s1 = np.cos(psi_new), np.sin(psi_new)
            R1 = np.array([[c1, -s1],
                           [s1, c1]])
            rotated = unrotated @ R1.T

            # Translate to new position
            new_bbox = rotated + center_new
            return new_bbox

        width, length, _ = ego_info["size"]
        initial_vehicle_pose = VehiclePose.from_nusc(
            ego_info['translation'][0], ego_info['translation'][1], ego_info['rotation'], width, length
        )
        vehicle_poses = [initial_vehicle_pose]
        prev_pose = initial_vehicle_pose
        for i, pred in enumerate(prediction):
            x, y = pred[0], pred[1]
            psi = np.arctan2(y - prev_pose.y, x - prev_pose.x)
            bbox = rotate_bbox(prev_pose.bbox, np.array([prev_pose.x, prev_pose.y]), prev_pose.psi,
                               np.array([x, y]), psi)
            cur_pose = VehiclePose(x=x, y=y, psi=psi, bbox=bbox, length=length, width=width)
            vehicle_poses.append(cur_pose)
            prev_pose = cur_pose
        return vehicle_poses

    def get_lanelet_and_relation_from_vehicle_poses(
            self,
            vehicle_poses,
            max_dist_to_lanelet: float = 0.5,
            remove_non_rule_compliant_matches: bool = False,
            debug: bool = False,
    ) -> Tuple[List[List[str]], List[List[Lanelet]], Dict[str, Dict[str, str]]]:
        """
        Returns:
            List of matching lanelet IDs corresponding to the vehicle poses
            and list of matching lanelets
            and their relationship as an adjacency map (dictionary of dictionaries).
        """
        lanelet_ids, all_lanelets, relations = [], {}, {}
        set_all_lanelets = set()
        for vehicle_pose in vehicle_poses:
            ll_mappings = self.match_vehicle_onto_lanelets_probabilistically(
                vehicle_pose,
                max_dist_to_lanelet=max_dist_to_lanelet,
                remove_non_rule_compliant_matches=remove_non_rule_compliant_matches,
                debug=False,
            )
            cur_pose_ll_ids = []
            for ll_id, ll_match in ll_mappings.items():
                cur_pose_ll_ids.append(ll_id)
                all_lanelets[ll_id] = ll_match.lanelet
                set_all_lanelets.add(ll_match.lanelet)
            lanelet_ids.append(cur_pose_ll_ids)
        if debug:
            print(f'set_all_lanelets:')
            for ll in set_all_lanelets:
                print(f'\t{ll}')

        # assert len(lanelet_ids) == len(vehicle_poses) > 0
        # if len(lanelet_ids[0]) > 0 and str(lanelet_ids[0][0]) == '18790':
        #     for ll_id, lanelet in all_lanelets.items():
        #         print(f'{ll_id}: {lanelet}')
        for u_id, u in all_lanelets.items():
            for v_id, v in all_lanelets.items():
                if u_id == v_id:
                    rel = 'Self'
                else:
                    rel = self.routing_graph.routingRelation(u, v, includeConflicting=True)
                    rel = str(rel).split('.')[-1]
                relations.setdefault(u_id, {})[v_id] = rel

        # only return lanelets that is matched probabilistically (ids stored in lanelet_ids)
        lanelets = [[all_lanelets[ll_id] for ll_id in per_step_lls] for per_step_lls in lanelet_ids]

        return lanelet_ids, lanelets, relations

    def get_matching_lanelets_from_vehicle_poses(
            self,
            vehicle_poses,
            max_dist_to_lanelet: float = 0.5,
    ) -> List[List[LaneletMatchProb]]:
        assert len(vehicle_poses) == 13
        matching_lanelets = []
        for vehicle_pose in vehicle_poses:
            cur_pose_matched_lanelet = self.match_vehicle_onto_lanelets_probabilistically(
                vehicle_pose,
                max_dist_to_lanelet=max_dist_to_lanelet,
            )
            matching_lanelets.append(list(cur_pose_matched_lanelet.values()))
        return matching_lanelets

    def get_lanelet_relations_from_lanelets(
            self,
            lanelets: List[List[LaneletMatchProb]],
    ):
        lanelet_relations = []
        for i in range(1, len(lanelets)):
            cur_lanelet_relations = []
            for prev_lanelet in lanelets[i - 1]:
                for cur_lanelet in lanelets[i]:
                    if cur_lanelet.lanelet.id == prev_lanelet.lanelet.id:
                        lanelet_relation = 'self'
                    else:
                        lanelet_relation = self.routing_graph.routingRelation(
                            prev_lanelet.lanelet, cur_lanelet.lanelet, includeConflicting=True
                        )
                    cur_lanelet_relations.append(lanelet_relation)
            lanelet_relations.append(cur_lanelet_relations)
        return lanelet_relations

    def check_rules_for_prediction(
            self,
            initial_position,
            prediction,
            rules: List[str]
    ):
        """Check for each rule in the list

        Args:

        Returns:
            Dict[str, List[float]]: mapping from each rule to a list of values corresponding
            to the rule violation level of each vehicle pose (waypoint).
        """
        vehicle_poses = self.prediction_to_vehicle_poses(prediction) # TODO: weighted vehicle poses based on prob
        prediction_lanelets = []
        for vehicle_pose in vehicle_poses:
            matching_lanelets = self.match_vehicle_onto_lanelets_probabilistically(
                vehicle_pose,
                max_dist_to_lanelet = 0.5,
            )
            prediction_lanelets.append(list(matching_lanelets.values()))

        assert len(prediction_lanelets) == 12 # predicts next 6s (12 frames)
        rule_violations = {
            'lane_direction': [],
            'off_road': [],
            'lane_change': [],
        }
        lanelet_relations = []
        for i in range(len(prediction_lanelets)):
            rule_violation = {
                'lane_direction': 0,
                'off_road': 0,
                'lane_change': 0
            }
            prev_lanelets = prediction_lanelets[i - 1] if i > 0 else initial_position
            cur_lanelets = prediction_lanelets[i]
            num_permutations = len(prev_lanelets) * len(cur_lanelets)
            if num_permutations == 0:
                rule_violation['off_road'] = 1.0
            cur_lanelet_relations = []
            for prev_lanelet in prev_lanelets:
                for cur_lanelet in cur_lanelets:
                    lanelet_relation = self.routing_graph.routingRelation(
                        prev_lanelet.lanelet, cur_lanelet.lanelet, includeConflicting=True
                    )
                    cur_lanelet_relations.append(lanelet_relation)
            lanelet_relations.append(cur_lanelet_relations)
        print(f'lanelet relations: {lanelet_relations}')
        pass

        # TODO:
        # Find all matching lanelets for each waypoint (with probability)
        # Check rules for [prev_lanelet, cur_lanelet] for each permutation, if violate, value = 1.0
        #   => (if 2+ lanelets) take weighted sum of all permutation (using probability of the matching lanelet)
        #   => rule violation level
        #
        #   3 cases:
        #       no cur_lanelet => rule_1 = 0.0, rule_2 = 1.0, rule_3 = 0.0
        #       [prev_lanelet, cur_lanelet] not neighbor <= TODO: check neighborhood relationship

        # TODO #2: check rule for all 5 predictions

    def get_reachable_lanelets_for_vehicle(
            self,
            vehicle_pose: VehiclePose,
            max_length: float = 100,
            probabilitisc_matching: bool = True,
            max_dist_to_lanelet: float = 0.5,
    ) -> List[List[Lanelet]]:
        """Compute diverse map based anchor paths by first matching the vehicle onto the Lanelet map and subsequently generating and filtering anchor paths.

        Args:
            vehicle_pose (VehiclePose): Position and orientation of the considered vehicle
            max_length (float, optional): Desired length of the anchor in meters. Start lanelet is NOT included. Note: Anchors can have a length shorter than length, if there is a dead end. Defaults to 100.
            probabilitisc_matching (bool, optional): Whether the matching of the vehicle onto the Lanelet is probabilistic or deterministic. Defaults to True.
            max_dist_to_lanelet (float, optional): Euclidean distance to which we find lanelets

        Returns:
            List[List[Lanelet]]: List of reachable lanelets for each matching starting Lanelet (according to vehicle_pose)
        """
        if probabilitisc_matching:
            lanelet_matches = self.match_vehicle_onto_lanelets_probabilistically(
                vehicle_pose,
                max_dist_to_lanelet
            )
        else:
            lanelet_matches = self.match_vehicle_onto_lanelets_deterministically(
                vehicle_pose,
                max_dist_to_lanelet
            )

        # Find all reachable lanelets from each of the match (regardless of probability)
        reachable_lanelets = []
        for ll_id in lanelet_matches.keys():
            reachable_lanelets.append(
                self.get_reachable_lanelets_from(lanelet_id=int(ll_id), max_length=max_length)
            )

        return reachable_lanelets

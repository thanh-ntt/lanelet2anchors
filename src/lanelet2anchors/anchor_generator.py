from collections import Counter
from pathlib import Path
from typing import Dict, List, Union

import lanelet2
import numpy as np
from lanelet2.io import Origin
from lanelet2.matching import getDeterministicMatches, getProbabilisticMatches
from lanelet2.projection import UtmProjector
from shapely.geometry import LineString
from lanelet2.core import Lanelet

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
        print(f'init AnchorGenerator')
        origin = Origin(origin_latitude, origin_longitude)
        projector = UtmProjector(origin)
        traffic_rules = lanelet2.traffic_rules.create(
            lanelet2.traffic_rules.Locations.Germany,
            lanelet2.traffic_rules.Participants.Vehicle,
        )

        self.lanelet_map = lanelet2.io.load(str(osm_file), projector)
        self.routing_graph = lanelet2.routing.RoutingGraph(
            self.lanelet_map, traffic_rules
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
        print(f'max_dist_to_lanelet: {max_dist_to_lanelet}')
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

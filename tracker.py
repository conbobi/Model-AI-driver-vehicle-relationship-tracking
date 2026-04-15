from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from boxmot import ByteTrack  # type: ignore
except ImportError:
    ByteTrack = None


@dataclass
class _TrackState:
    track_id: int
    bbox: list[float]
    conf: float
    cls_id: int
    age: int = 0
    hits: int = 1


class SimpleTracker:
    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        iou_threshold: float = 0.3,
        max_age: int = 15,
        use_boxmot: bool = True,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_track_id = 1
        self.active_tracks: list[_TrackState] = []

        self.tracker = None
        if use_boxmot and ByteTrack is not None:
            self.tracker = ByteTrack(
                track_high_thresh=track_high_thresh,
                track_low_thresh=track_low_thresh,
                new_track_thresh=new_track_thresh,
            )

    def update(self, detections, img):
        """
        detections: list[[x1, y1, x2, y2, conf, class_id]]
        returns: list[[x1, y1, x2, y2, track_id, conf, class_id]]
        """
        if self.tracker is not None:
            return self._update_boxmot(detections, img)
        return self._update_iou(detections)

    def _update_boxmot(self, detections, img):
        if len(detections) == 0:
            tracks = self.tracker.update(np.empty((0, 5)), img.shape[:2], img.shape[:2])
        else:
            dets = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections], dtype=np.float32)
            tracks = self.tracker.update(dets, img.shape[:2], img.shape[:2])

        results = []
        for track in tracks:
            x1, y1, x2, y2, tid, conf = track[:6]
            cls_id = -1
            best_iou = 0.0
            for det in detections:
                iou = self._iou([x1, y1, x2, y2], det[:4])
                if iou > best_iou:
                    best_iou = iou
                    cls_id = int(det[5])
            results.append([float(x1), float(y1), float(x2), float(y2), int(tid), float(conf), int(cls_id)])
        return results

    def _update_iou(self, detections):
        for track in self.active_tracks:
            track.age += 1

        unmatched_detections = set(range(len(detections)))
        matches: list[tuple[int, int]] = []

        if self.active_tracks and detections:
            iou_candidates = []
            for track_index, track in enumerate(self.active_tracks):
                for det_index, det in enumerate(detections):
                    if track.cls_id != int(det[5]):
                        continue
                    iou = self._iou(track.bbox, det[:4])
                    if iou >= self.iou_threshold:
                        iou_candidates.append((iou, track_index, det_index))

            for _, track_index, det_index in sorted(iou_candidates, reverse=True):
                if det_index not in unmatched_detections:
                    continue
                if self.active_tracks[track_index].age == 0:
                    continue
                unmatched_detections.remove(det_index)
                matches.append((track_index, det_index))
                self.active_tracks[track_index].age = 0

        matched_track_indices = {track_index for track_index, _ in matches}
        for track_index, det_index in matches:
            det = detections[det_index]
            track = self.active_tracks[track_index]
            track.bbox = list(map(float, det[:4]))
            track.conf = float(det[4])
            track.cls_id = int(det[5])
            track.hits += 1

        self.active_tracks = [
            track
            for index, track in enumerate(self.active_tracks)
            if index in matched_track_indices or track.age <= self.max_age
        ]

        for det_index in sorted(unmatched_detections):
            det = detections[det_index]
            self.active_tracks.append(
                _TrackState(
                    track_id=self.next_track_id,
                    bbox=list(map(float, det[:4])),
                    conf=float(det[4]),
                    cls_id=int(det[5]),
                )
            )
            self.next_track_id += 1

        results = []
        for track in self.active_tracks:
            if track.age > self.max_age:
                continue
            x1, y1, x2, y2 = track.bbox
            results.append([x1, y1, x2, y2, track.track_id, track.conf, track.cls_id])
        return results

    @staticmethod
    def _iou(box1, box2) -> float:
        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        x2 = min(float(box1[2]), float(box2[2]))
        y2 = min(float(box1[3]), float(box2[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
        area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

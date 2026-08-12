"""Exact-visual MuJoCo objects for the UniVTAC HDMI insertion task."""

from pathlib import Path

from robosuite.models.objects import MujocoXMLObject


ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "objects"


class HDMIObject(MujocoXMLObject):
    def __init__(self, name="prism"):
        super().__init__(
            str(ASSET_ROOT / "hdmi.xml"),
            name=name,
            joints="default",
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class HDMISlotObject(MujocoXMLObject):
    def __init__(self, name="slot"):
        super().__init__(
            str(ASSET_ROOT / "hdmi_slot.xml"),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=False,
        )

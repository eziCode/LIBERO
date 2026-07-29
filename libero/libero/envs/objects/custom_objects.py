import numpy as np

from robosuite.models.objects import HammerObject
from robosuite.utils.mjcf_utils import add_to_dict
from libero.libero.envs.base_object import register_object
from libero.libero.envs.objects.hope_objects import HopeBaseObject
from robosuite.models.objects.primitive.box import BoxObject

# We will wrap HammerObject so it's registered
@register_object
class Hammer(HammerObject):
    def __init__(self, name):
        super().__init__(
            name=name,
            handle_radius=(0.0175, 0.0175),
            handle_length=(0.2, 0.2)
        )

        # Adding mock properties to be compatible with LIBERO
        self.object_properties = {"vis_site_names": {}}
        self.rotation = [0, 0]
        self.rotation_axis = "z"
        self.category_name = "hammer"

    @property
    def init_quat(self):
        """
        Forces the hammer to always face left by returning a fixed orientation.
        """
        return np.array([0.5, -0.5, 0.5, -0.5])

    def _get_geom_attrs(self):
        obj_args = super()._get_geom_attrs()

        # Add a green visualization line that extends outward from the hammer face.
        line_length = self.head_halfsize * 4.0
        line_thickness = self.head_halfsize * 0.05
        face_center_x = self.head_halfsize * 2.8
        face_center_z = self.handle_length / 2.0 + self.head_halfsize

        # Only extend the per-geom lists, not the whole obj_args dict.
        per_geom = {
            "geom_types": obj_args["geom_types"],
            "geom_locations": obj_args["geom_locations"],
            "geom_quats": obj_args["geom_quats"],
            "geom_sizes": obj_args["geom_sizes"],
            "geom_names": obj_args["geom_names"],
            "geom_rgbas": obj_args["geom_rgbas"],
            "geom_materials": obj_args["geom_materials"],
            "geom_frictions": obj_args["geom_frictions"],
            "obj_types": obj_args["obj_types"],
            "density": obj_args["density"],
        }
        if isinstance(per_geom["obj_types"], str):
            per_geom["obj_types"] = [per_geom["obj_types"]] * len(per_geom["geom_types"])

        add_to_dict(
            dic=per_geom,
            geom_types="box",
            geom_locations=(face_center_x + line_length / 2.0, 0, face_center_z),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=np.array([line_length / 2.0, line_thickness, line_thickness]),
            geom_names="hammer_head_line",
            geom_rgbas=[0.0, 1.0, 0.0, 1.0],
            geom_materials=None,
            geom_frictions=None,
            obj_types="visual",
            density=1.0,
        )

        obj_args.update(per_geom)
        return obj_args

@register_object
class Board(BoxObject):
    def __init__(self, name):
        super().__init__(
            name=name,
            size=[0.1, 0.1, 0.02],
            rgba=[0.6, 0.4, 0.2, 1.0],
            density=1000.0,
            joints=None,
        )
        self.object_properties = {"vis_site_names": {}}
        self.rotation = [0, 0]
        self.rotation_axis = "z"
        self.category_name = "board"

@register_object
class Nail(BoxObject):
    def __init__(self, name):
        super().__init__(
            name=name,
            size=[0.01, 0.01, 0.05],
            rgba=[0.8, 0.8, 0.8, 1.0],
            density=2000.0,
            joints=[{"type": "slide", "axis": "0 0 1", "damping": "1.0", "frictionloss": "1.0"}],
        )
        self.object_properties = {"vis_site_names": {}}
        self.rotation = [0, 0]
        self.rotation_axis = "z"
        self.category_name = "nail"

from libero.libero.envs.objects.turbosquid_objects import WhiteYellowMug
@register_object
class FlippedMug(WhiteYellowMug):
    def __init__(self, name):
        super().__init__(name=name)
        self.object_properties = {"vis_site_names": {}}
        self.init_quat = [1.0, 1.0, 0.0, 1.0]
        self.rotation = [0.0, 0.0]
        self.rotation_axis = "z"
        self.category_name = "white_yellow_mug"


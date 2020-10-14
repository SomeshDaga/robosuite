"""
Gripper for Barrett's WAM robot arm (has three fingers).
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper_model import GripperModel


class WAMBarrettGripperBase(GripperModel):
    """
    Gripper for WAM robot arm (has three fingers).

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/wam_barrett_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def dof(self):
        # The Barrett Hand technically has 8 dof, but we only use 6
        return 6

    @property
    def init_qpos(self):
        return np.array([0, 0.5, 0, 0, 0.5, 0, 0.5, 0])

    @property
    def _joints(self):
        return [
            "wam/bhand/index/prox_joint", "wam/bhand/index/med_joint", "wam/bhand/index/dist_joint",
            "wam/bhand/middle/prox_joint", "wam/bhand/middle/med_joint", "wam/bhand/middle/dist_joint",
            "wam/bhand/thumb/med_joint", "wam/bhand/thumb/dist_joint"
        ]

    @property
    def _actuators(self):
        return [
            "index_med",
            "index_dist",
            "middle_med",
            "middle_dist",
            "thumb_med",
            "thumb_dist"
        ]

    @property
    def _contact_geoms(self):
        return [
            "bhand_palm_collision",
            "bhand_index_prox_collision", "bhand_index_med_collision", "bhand_index_dist_collision",
            "bhand_middle_prox_collision", "bhand_middle_med_collision", "bhand_middle_dist_collision",
            "bhand_thumb_med_collision", "bhand_thumb_dist_collision"
        ]

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["bhand_index_prox_collision", "bhand_index_med_collision", "bhand_index_dist_collision",
                            "bhand_middle_prox_collision", "bhand_middle_med_collision", "bhand_middle_dist_collision"],
            "right_finger": ["bhand_thumb_med_collision", "bhand_thumb_dist_collision"]
        }


class WAMBarrettGripper(WAMBarrettGripperBase):
    """
    Modifies WAMBarrettGripperBase to only take one action.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(self.current_action - self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        return 0.005

    @property
    def dof(self):
        return 1

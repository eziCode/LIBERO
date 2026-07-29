from typing import List


class Expression:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class UnaryAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, arg1):
        raise NotImplementedError


class BinaryAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, arg1, arg2):
        raise NotImplementedError


class MultiarayAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, *args):
        raise NotImplementedError


class TruePredicateFn(MultiarayAtomic):
    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        return True


class FalsePredicateFn(MultiarayAtomic):
    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        return False


class InContactPredicateFn(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg1.check_contact(arg2)


class In(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg2.check_contact(arg1) and arg2.check_contain(arg1)


class On(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg2.check_ontop(arg1)

        # if arg2.object_state_type == "site":
        #     return arg2.check_ontop(arg1)
        # else:
        #     obj_1_pos = arg1.get_geom_state()["pos"]
        #     obj_2_pos = arg2.get_geom_state()["pos"]
        #     # arg1.on_top_of(arg2) ?
        #     # TODO (Yfeng): Add checking of center of mass are in the same regions
        #     if obj_1_pos[2] >= obj_2_pos[2] and arg2.check_contact(arg1):
        #         return True
        #     else:
        #         return False


class Up(BinaryAtomic):
    def __call__(self, arg1):
        return arg1.get_geom_state()["pos"][2] >= 1.0


class Stack(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return (
            arg1.check_contact(arg2)
            and arg2.check_contain(arg1)
            and arg1.get_geom_state()["pos"][2] > arg2.get_geom_state()["pos"][2]
        )


class PrintJointState(UnaryAtomic):
    """This is a debug predicate to allow you print the joint values of the object you care"""

    def __call__(self, arg):
        print(arg.get_joint_state())
        return True


class Open(UnaryAtomic):
    def __call__(self, arg):
        return arg.is_open()


class Close(UnaryAtomic):
    def __call__(self, arg):
        return arg.is_close()


class TurnOn(UnaryAtomic):
    def __call__(self, arg):
        return arg.turn_on()


class TurnOff(UnaryAtomic):
    def __call__(self, arg):
        return arg.turn_off()


class IsShaken(UnaryAtomic):
    def __call__(self, arg1):
        if hasattr(arg1, "has_been_shaken"):
            return arg1.has_been_shaken
        return False

class IsHammered(UnaryAtomic):
    def __call__(self, arg1):
        if hasattr(arg1, "has_been_hammered"):
            return arg1.has_been_hammered
        return False

class IsUpright(UnaryAtomic):
    def __call__(self, arg1):
        import robosuite.utils.transform_utils as T
        quat = arg1.get_geom_state()["quat"]
        # Convert MuJoCo (w,x,y,z) to (x,y,z,w) for robosuite utilities
        mat = T.quat2mat(T.convert_quat(quat, to="xyzw"))
        # Check if the local Z axis is aligned with the world Z axis (upright)
        is_upright = mat[2, 2] > 0.95
        
        # Check if the cup is actually on the table (not in air or grasped)
        # Table surface is at 0.90. Mug origin at base when upright is 0.90.
        obj_pos = arg1.env.sim.data.body_xpos[arg1.env.obj_body_id[arg1.object_name]]
        is_on_table = 0.89 < obj_pos[2] < 0.93
            
        # Diagnostic print to help tune the success condition
        # print(f"[DEBUG] Success check: Upright={mat[2,2]:.3f} (goal > 0.95), Z={obj_pos[2]:.3f} (goal 0.89-0.93), OnTable={is_on_table}")
        
        return is_upright and is_on_table


class IsGrasped(UnaryAtomic):
    def __init__(self):
        super().__init__()
        self.initial_z = None

    def __call__(self, arg1):
        pos = arg1.get_geom_state()["pos"]
        
        # Capture the resting position on the first frame
        if self.initial_z is None:
            self.initial_z = pos[2]
            
        # The object is grasped when it has been lifted at least 5cm off its starting resting point
        return pos[2] > (self.initial_z + 0.05)

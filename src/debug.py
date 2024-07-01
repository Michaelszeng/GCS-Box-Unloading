from pydrake.all import (
    LeafSystem,
    AbstractValue,
    RigidTransform,
)

from scenario import NUM_BOXES

import numpy as np

class Debugger(LeafSystem):
    """
    Simple Leafsystem that prints information from input port for debugging
    purposes.
    """

    def __init__(self, print_frequency=0.025):
        LeafSystem.__init__(self)

        kuka_state = self.DeclareVectorInputPort(name="kuka_state", size=12)  # 6 pos, 6 vel

        body_poses = AbstractValue.Make([RigidTransform()])
        self.DeclareAbstractInputPort("body_poses", body_poses)

        kuka_actuation = self.DeclareVectorInputPort(name="kuka_actuation", size=6)

        self.DeclarePeriodicUnrestrictedUpdateEvent(print_frequency, 0.0, self.debug_print)

        self.on = False


    def debug_print(self, context, state):
        kuka_state = self.get_input_port(0).Eval(context)
        q_current = kuka_state[:6]
        q_dot_current = kuka_state[6:]

        kuka_actuation = self.get_input_port(2).Eval(context)
        
        if self.on:
            print(f"q_current: {q_current}")
            print(f"q_dot_current: {q_dot_current}")
            print(f"kuka_actuation: {kuka_actuation}")
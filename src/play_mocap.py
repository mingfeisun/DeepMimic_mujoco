from os import getcwd
from mujoco_py import load_model_from_xml, MjSim, MjViewer

from config import Config

class PlayMocap(object):
    def __init__(self):
        xmlpath = Config.xml_path
        with open(xmlpath) as fin:
            MODEL_XML = fin.read()

        model = load_model_from_xml(MODEL_XML)
        self.sim = MjSim(model)

    def show_frame(self, this_frame):
        viewer = MjViewer(self.sim)
        sim_state = self.sim.get_state()
        sim_state.qpos[:] = this_frame[:]
        self.sim.set_state(sim_state)
        self.sim.forward()
        viewer.render()

if __name__ == "__main__":
    test = PlayMocap()
import time

import A1.a1module
import A2.a2module
import B1.b1module
import B2.b2module

class Timer:
    timer = 0

    def __init__(self):
        self.reset()

    def reset(self):
        self.timer = time.time()

    def print(self):
        return str(time.time() - self.timer)


def main():
    # Run solutions to task A1
    print("A1: RUN1", flush = True)
    timer = Timer()
    A1.a1module.run_task(gen_convergence_plot = True, plot_conv_path = "./A1/convergence.png", plot_conf_path = "./A1/conf_mat.png")
    print("TASK FINISHED IN " + timer.print() + "s\n", flush = True)
    

    # Run solutions to task A2
    print("A2: RUN1", flush = True)
    timer.reset()
    A2.a2module.run_task(gen_convergence_plot = True, plot_conv_path = "./A2/convergence.png", plot_conf_path = "./A2/conf_mat.png")
    print("TASK FINISHED IN " + timer.print() + "s\n", flush = True)


    # Run solutions to task B1
    print("B1: RUN1: USE 50% RESCALING AND EDGE DETECTION", flush = True)
    timer.reset()
    B1.b1module.run_task(enable_edge_detection = True, enable_resize = True, resize_scaling = 0.5, gen_convergence_plot = True, plot_conv_path = "./B1/convergenceEdge.png", plot_conf_path = "./B1/conf_mat_Edge.png")
    print("TASK FINISHED IN " + timer.print() + "s\n", flush = True)

    print("B1: RUN2: USE 50% RESCALING W/O EDGE DETECTION", flush = True)
    timer.reset()
    B1.b1module.run_task(enable_edge_detection = False, enable_resize = True, resize_scaling = 0.5, gen_convergence_plot = True, plot_conv_path = "./B1/convergenceGray.png", plot_conf_path = "./B1/conf_mat_Gray.png")
    print("TASK FINISHED IN " + timer.print() + "s\n", flush = True)


    # Run solutions to task B2
    print("B2: RUN1: KEEP SUNGLASSES DATA POINTS UNALTERED", flush = True)
    timer.reset()
    B2.b2module.run_task(add_sunglasses_lab = False, rm_train_sun_dp = False, rm_test_sun_dp = False, gen_convergence_plot = True, plot_conv_path = "./B2/convergenceUnaltered.png", plot_conf_path = "./B2/conf_mat_Unaltered.png")
    print("TASK FINISHED IN " + timer.print() + "s\n", flush = True)

    print("B2: RUN2: REMOVE SUNGLASSES DATA POINTS FROM THE TRAINING DATA", flush = True)
    timer.reset()
    B2.b2module.run_task(add_sunglasses_lab = False, rm_train_sun_dp = True, rm_test_sun_dp = False, gen_convergence_plot = True, plot_conv_path = "./B2/convergence.png", plot_conf_path = "./B2/conf_mat.png")
    print("TASK FINISHED IN " + timer.print() + "s\n", flush = True)

    print("B2: RUN3: REMOVE SUNGLASSES DATA POINTS FROM BOTH TRAINING AND TEST DATA", flush = True)
    timer.reset()
    B2.b2module.run_task(add_sunglasses_lab = False, rm_train_sun_dp = True, rm_test_sun_dp = True)
    print("TASK FINISHED IN " + timer.print() + "s\n", flush = True)

    print("B2: RUN4: ADD SUNGLASSES LABELS", flush = True)
    timer.reset()
    B2.b2module.run_task(add_sunglasses_lab = True, rm_train_sun_dp = False, rm_test_sun_dp = False, gen_convergence_plot = True, plot_conv_path = "./B2/convergenceExtraLab.png", plot_conf_path = "./B2/conf_mat_ExtraLab.png")
    print("TASK FINISHED IN " + timer.print() + "s\n", flush = True)

 
if __name__ == "__main__":
    main()

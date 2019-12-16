"""
Script that cleans up temporary files generated during a :py:mod:`grid` simulation run.

"""

if __name__ == "__main__":

    import os

    os.system("rm testing_input*")
    os.system("rm training_input*")
    os.system("rm forming_input*")
    os.system("rm *_train*")
    os.system("rm main_exec_res*")
    os.system("rm gridconfig/*.pyc")
    os.system("rm *.pyc")
    os.system("rm slurm*")
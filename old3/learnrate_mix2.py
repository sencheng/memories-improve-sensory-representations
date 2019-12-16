import os, time, itertools

# typelist = ['o14', 'o18']
typelist = ['o18']
framelist = [50]
# rotlist = ['t', 'o']
rotlist = ['t']

for typ, fram, rot in itertools.product(typelist, framelist, rotlist):
    os.system("sbatch batpre learnrate_mix_pre_ex2.py {} {} {} &".format(typ, fram, rot))
    # os.system("srun python learnrate_pre_ex2.py {} {} {} ".format(typ, fram, rot))

    # time.sleep(2)
    # PATH = "../results/lro_{}{}{}/".format(typ, fram, rot)
    # if not os.path.isfile(PATH + "sfa1.p"):
    #     print("In " + PATH + ", sfa1 was not created. Aborting.")
    #     continue

    time.sleep(2)

    eps_list = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    for dim_id in [1]:
        for ei, eps in enumerate(eps_list):
            os.system("sbatch batlearnrate learnrate_mix_ex2.py {} {} {} {} {} {} &".format(dim_id, ei, eps, typ, fram, rot))
            # os.system("srun python learnrate_ex2.py {} {} {} {} {} {} &".format(dim_id, ei, eps, typ, fram, rot))

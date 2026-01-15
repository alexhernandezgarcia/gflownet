import pickle
import matplotlib.pyplot as plt


if __name__=="__main__":
    use_fake_crystal = False
    use_no_comp_crystal = False
    use_crystal  = True

    filepath = "/network/projects/crystalgfn/catalyst/gflownet/logs/crystalgfn/8477498/2026-01-14_16-11-32_861501/eval/samples/gfn_samples.pkl"
    # "/network/projects/crystalgfn/catalyst/gflownet/logs/crystalgfn/8475366/2026-01-14_09-40-36_455376/eval/samples/gfn_samples.pkl"
    # "/network/projects/crystalgfn/catalyst/gflownet/logs/crystalgfn/8475366/2026-01-14_09-40-36_455376/eval/samples/gfn_samples.pkl"
    with open(filepath, 'rb') as handle:
        data = pickle.load(handle)


    if use_fake_crystal:
        x_coord_opt1 = [data["x"][i][2][0] for i in range(len(data["x"])) if data["x"][i][1][0] == 1]
        x_coord_opt2 = [data["x"][i][2][0] for i in range(len(data["x"])) if data["x"][i][1][0] == 2]
        y_coord_opt1 = [data["x"][i][2][1] for i in range(len(data["x"])) if data["x"][i][1][0] == 1]
        y_coord_opt2 = [data["x"][i][2][1] for i in range(len(data["x"])) if data["x"][i][1][0] == 2]
        breakpoint()
        fig, ax = plt.subplots()
        plt.title("Option 1 (136) - cube samples")
        ax.set_aspect("equal")
        hist, xbins, ybins, im = ax.hist2d(x_coord_opt1,y_coord_opt1,bins=15)
        for i in range(len(ybins)-1):
            for j in range(len(xbins)-1):
                ax.text(xbins[j]+0.05/1.5,ybins[i]+0.05/1.5, int(hist.T[i,j]), 
                        color="w", ha="center", va="center")
        plt.xlabel('coordinate 1 of the cube')
        plt.ylabel('coordinate 2 of the cube')
        plt.savefig('scripts/tmp_opt1.png')
        plt.clf()
        fig, ax = plt.subplots()
        plt.title("Option 2 (221) - cube samples")
        ax.set_aspect("equal")
        hist, xbins, ybins, im = ax.hist2d(x_coord_opt2,y_coord_opt2,bins=15)
        for i in range(len(ybins)-1):
            for j in range(len(xbins)-1):
                ax.text(xbins[j]+0.05/1.5,ybins[i]+0.05/1.5, int(hist.T[i,j]), 
                        color="w", ha="center", va="center")
        plt.xlabel('coordinate 1 of the cube')
        plt.ylabel('coordinate 2 of the cube')
        plt.savefig('scripts/tmp_opt2.png')
    elif use_no_comp_crystal:
        x_coord_opt1 = [data["x"][i][2][0] for i in range(len(data["x"])) if data["x"][i][1][2] == 136]
        x_coord_opt2 = [data["x"][i][2][0] for i in range(len(data["x"])) if data["x"][i][1][2] == 221]
        y_coord_opt1 = [data["x"][i][2][2] for i in range(len(data["x"])) if data["x"][i][1][2] == 136]
        y_coord_opt2 = [data["x"][i][2][2] for i in range(len(data["x"])) if data["x"][i][1][2] == 221]
        
        fig, ax = plt.subplots()
        plt.title("Option 1 (136) - cube samples")
        ax.set_aspect("equal")
        hist, xbins, ybins, im = ax.hist2d(x_coord_opt1,y_coord_opt1,bins=15)
        for i in range(len(ybins)-1):
            for j in range(len(xbins)-1):
                ax.text(xbins[j]+0.05/1.5,ybins[i]+0.05/1.5, int(hist.T[i,j]), 
                        color="w", ha="center", va="center")
        plt.xlabel('coordinate 1 of the cube')
        plt.ylabel('coordinate 2 of the cube')
        plt.savefig('scripts/tmp_opt1.png')
        plt.clf()
        fig, ax = plt.subplots()
        plt.title("Option 2 (221) - cube samples")
        ax.set_aspect("equal")
        hist, xbins, ybins, im = ax.hist2d(x_coord_opt2,y_coord_opt2,bins=15)
        for i in range(len(ybins)-1):
            for j in range(len(xbins)-1):
                ax.text(xbins[j]+0.05/1.5,ybins[i]+0.05/1.5, int(hist.T[i,j]), 
                        color="w", ha="center", va="center")
        plt.xlabel('coordinate 1 of the cube')
        plt.ylabel('coordinate 2 of the cube')
        plt.savefig('scripts/tmp_opt2.png')
    elif use_crystal:
        x_coord_opt1 = [data["proxy"][i][-6] for i in range(len(data["proxy"])) if int(data["proxy"][i][-7]) == 136] # data["proxy"][0][-4]
        x_coord_opt2 = [data["proxy"][i][-6] for i in range(len(data["proxy"])) if int(data["proxy"][i][-7]) == 221]
        y_coord_opt1 = [data["proxy"][i][-4] for i in range(len(data["proxy"])) if int(data["proxy"][i][-7]) == 136] # data["proxy"][0][-4]
        y_coord_opt2 = [data["proxy"][i][-4] for i in range(len(data["proxy"])) if int(data["proxy"][i][-7]) == 221]

        fig, ax = plt.subplots(figsize=(8,8))
        plt.title("Space group 136 - samples")
        ax.set_aspect("equal")
        hist, xbins, ybins, im = ax.hist2d(x_coord_opt1,y_coord_opt1,bins=10)
        offset = 5.0
        for i in range(len(ybins)-1):
            for j in range(len(xbins)-1):
                ax.text(xbins[j]+offset,ybins[i]+offset, int(hist.T[i,j]), 
                        color="w", ha="center", va="center")
        plt.xlabel('1st lattice parameter')
        plt.ylabel('3rd lattice parameter')
        plt.savefig('scripts/tmp_full_crystal_136.png')
        plt.clf()
        fig, ax = plt.subplots(figsize=(8,8))
        plt.title("Space group 221 - samples")
        ax.set_aspect("equal")
        offset2 = 3.0
        hist, xbins, ybins, im = ax.hist2d(x_coord_opt2,y_coord_opt2,bins=10)
        for i in range(len(ybins)-1):
            for j in range(len(xbins)-1):
                ax.text(xbins[j]+offset,ybins[i]+offset, int(hist.T[i,j]), 
                        color="b", ha="center", va="center")
        plt.xlabel('1st lattice parameter')
        plt.ylabel('3rd lattice parameter')
        plt.savefig('scripts/tmp_full_crystal_221.png')

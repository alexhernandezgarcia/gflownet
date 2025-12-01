import pickle
import matplotlib.pyplot as plt


if __name__=="__main__":
    filepath = "/network/projects/crystalgfn/catalyst/gflownet/logs/grid/corners/2025-11-24_13-40-56_577304/eval/samples/gfn_samples.pkl"
    uniform_larger_lr = "/network/projects/crystalgfn/catalyst/gflownet/logs/grid/corners/2025-11-20_13-23-09_221535/eval/samples/gfn_samples.pkl"
    with open(filepath, 'rb') as handle:
        data = pickle.load(handle)

    x_coord_opt1 = [data["x"][i][2][0] for i in range(len(data["x"])) if data["x"][i][1][0] == 1]
    x_coord_opt2 = [data["x"][i][2][0] for i in range(len(data["x"])) if data["x"][i][1][0] == 2]
    y_coord_opt1 = [data["x"][i][2][1] for i in range(len(data["x"])) if data["x"][i][1][0] == 1]
    y_coord_opt2 = [data["x"][i][2][1] for i in range(len(data["x"])) if data["x"][i][1][0] == 2]
    
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
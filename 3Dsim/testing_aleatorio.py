import matplotlib.pyplot as plt
from matplotlib.patches import Arc

if __name__ == '__main__':

    fig, ax = plt.subplots()
    #plt.plot(100*xs, 100*ys, '.')
    circ = plt.Circle((0,0), 0.1, color='red', fill=0,lw=4)
    ax.add_patch(circ)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.title("Radial Scatter")
    plt.tight_layout()
    plt.show()

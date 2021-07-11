import numpy as np
import matplotlib.pyplot as plt

def time(file_path):
    cpu_time = {}
    gpu_time = {}

    f = open(file_path,"r")
    lines = f.readlines()
    for i in range(len(lines)):  
        line = lines[i]
        words = line.split()
        if not (words): continue
        elif words[0] == "CPU":
            cpu_time[int(words[3])] = float(lines[i+2].split()[1][2:-1])
        elif words[0] == "GPU":
            gpu_time[int(words[3])] = float(lines[i+2].split()[1][2:-1])
    
    f.close()
    return cpu_time, gpu_time



if __name__ == "__main__":
    cpu_time, gpu_time = time("./time.txt")

    x = np.array(list(cpu_time.keys()))
    y1 = np.array(list(cpu_time.values()))
    y2 = np.array(list(gpu_time.values()))
    m1, b1 = np.polyfit(x, y1, 1)
    print("CPU c1=%f c2=%f(1e7)"%(b1, m1*1e7))
    m2, b2 = np.polyfit(x, y2, 1)
    print("GPU c1=%f c2=%f(1e7)"%(b2, m2*1e7))

    plt.plot(x, y1, 'rv')
    plt.plot(x, y2, 'go')
    plt.plot(x, m1*x + b1, 'r-', label = "CPU")
    plt.plot(x, m2*x + b2, 'g--', label = "GPU")
    plt.xlabel("Vector Length")
    plt.ylabel("Time/seconds")
    plt.title("CPU vs GPU comparison")
    plt.legend()
    plt.savefig("./plot_time.png")

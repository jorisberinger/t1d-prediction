import json
import pandas
import matplotlib.pyplot as plt


def analyzeFile(filename):

    file = open(filename)
    res = json.load(file)
    analyze(res, filename)
def analyze(json, filename):
    data = json['data']
    count = 0
    for x in data:
        if(x[0] == 0 or x [1] == 0):
            count += 1

    data = pandas.DataFrame(data)
    print("number of results: " + str(len(data)))
    print("number of zeros: " + str(count))

    plt.plot(data[0], label="Standard", alpha=0.6)
    plt.plot(data[1], label="Adv", alpha=0.6)
    plt.grid(color="#cfd8dc")
    plt.legend()
    plt.title("error plot data: " + filename)
    plt.savefig("errorPlot.png", dpi=600)

    plt.figure()
    plt.boxplot([data[0], data[1]], labels=["standard", "adv"])
    plt.grid(color="#cfd8dc")
    plt.title("Boxplot Comparison data: " + filename)
    plt.savefig("boxplot.png", dpi=600)


if __name__ == '__main__':
    analyzeFile("result.json")
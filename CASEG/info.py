import os
from marissa.toolbox.tools import tool_general

if __name__ == "__main__":
    # return the used / necessary packages, output must be manually inspected and corrected
    # this solution is used instead of pip as more unnecessary modules are currently installed
    path = r"D:\ECRC_AG_CMR\3 - Promotion\Entwicklung\marissa"
    p, v = tool_general.get_python_packages(path)

    with open(os.path.join(path, "raw_package_overview.txt"), "w+") as wfile:
        wfile.write("package" + "\t" + "version" + "\n")
        for i in range(len(p)):
            wfile.write(p[i] + "\t" + v[i] + "\n")
    wfile.close()

    with open(os.path.join(path, "raw_requirements.txt"), "w+") as wfile:
        wfile.write("package" + "\t" + "version" + "\n")
        for i in range(len(p)):
            if v[i] == "":
                wfile.write(p[i] + "\n")
            else:
                wfile.write(p[i] + "==" + v[i] + "\n")
    wfile.close()
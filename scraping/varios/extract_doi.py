import sys  

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Forma de uso: {} entrada salida".format(sys.argv[0]))
        exit()
    out = open(sys.argv[2], "w")
    with open(sys.argv[1]) as f:
        for line in f:
            res = line.split(",")
            if res[9] == "" and res[7] == "":
                continue
            out.write("{},{}\n".format(res[9], res[7]))
    out.close()

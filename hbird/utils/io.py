

def read_file_set(path:str):
    with open(path, "r") as f:
        file_set = [x.strip() for x in f.readlines()]
    return file_set
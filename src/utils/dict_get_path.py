

def get_at(a:dict, path):
    # path is string or list of string
    if isinstance(path, str):
        return a[path]
    else:
        for p in path:
            a = a[p]
        return a
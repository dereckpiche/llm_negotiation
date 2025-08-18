
def get_from_nested_dict(a:dict, path) -> any:
    # path is string or list of string
    try:
        if isinstance(path, str):
            return a[path]
        else:
            for p in path:
                a = a[p]
            return a
    except Exception:
        return None

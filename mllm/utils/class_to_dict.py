def class_to_dict(obj):
    """
    Convert a class instance to a dictionary.

    Args:
        obj: The class instance to convert.

    Returns:
        dict: A dictionary representation of the class instance.
    """
    if isinstance(obj, dict):
        return obj
    return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}

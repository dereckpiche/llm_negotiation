import uuid


def generate_short_id() -> int:
    """
    Generates a short unique ID for tracking adapter versions.

    Returns:
        int: An 8-digit integer ID.
    """
    return int(str(uuid.uuid4().int)[:8])

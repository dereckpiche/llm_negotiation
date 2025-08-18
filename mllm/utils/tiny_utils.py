def find_subsequence(seq, subseq):
    """Return the index of the first occurrence of subseq in seq, or -1 if not found."""
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i:i+m] == subseq:
            return i
    return -1


import uuid
def generate_short_id() -> int:
    """
    Generates a short unique ID for tracking adapter versions.

    Returns:
        int: An 8-digit integer ID.
    """
    return int(str(uuid.uuid4().int)[:8])
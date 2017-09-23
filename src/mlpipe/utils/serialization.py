from ..pipeline import Pipe

def save(pipe, filename):
    """
    Utility function to save a pipe. This function is simply a shortcut to `pipe._save`.

    Args:
        pipe: a pipe instance
        filename: the filename to save the serialized pipe as

    Returns:
        the filename
    """
    pipe._save(filename)
    return filename


def load(filename):
    """
    Utility function to load a pipe. This function is simply a shortcut to `Pipe._load`.

    Args:
        filename: the location of the serialized pipe

    Returns:
        the pipe instance
    """

    return Pipe._load(filename)




import time

"""def get_kwarg(kwargs, name, default=None):
    if kwargs is None:
        kwargs = {}
    if name not in kwargs and default is None:
        raise AssertionError(f"'{name}' expected and not given")
    return kwargs.get(name, default)"""

def print_time(message:str, tabs:int=0):
    """
    Prints the time and a message in the form of:
    
    <time> <message>
    
    Example: 08:58:23 Hello World!
    """
    curr_time = time.strftime("%H:%M:%S")
    print(f"{' ' * tabs*4}{curr_time} {message}")

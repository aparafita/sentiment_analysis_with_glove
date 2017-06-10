# From http://code.activestate.com/recipes/134892/
# "getch()-like unbuffered character reading from stdin
#  on both Windows and Unix (Python recipe)"

# Getch is a standard Python recipe to request a single byte to the user 
# without having to type in linebreak each time

# It is extended to accept a string to print (as with the input() method)
# and to print the byte received from the user

class _Getch:
    """Gets a single character from standard input.  Does not echo to the screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self, prompt=None): 
        if prompt:
            print(prompt, end='')

        x = self.impl()
        print(x)

        return x

class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()


getch = _Getch()
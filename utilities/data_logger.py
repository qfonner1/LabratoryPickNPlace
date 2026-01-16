import sys

class Logger:
    """Redirects prints to both terminal and a file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w",encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)  # print to terminal
        self.log.write(message)       # save to file
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
"""
Errors and Exceptions in the
Current Package
"""


class WrongVariableType(Exception):
    pass


class WrongPythonException(Exception):
    """Raised when Python version is not the right one"""
    
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
        print("WrongPythonException: " + self.message + \
              ". Trying to run: " + \
              str(self.expression))
    
    def __str__(self):
        return repr(self.expression)


if __name__ == '__main__':
    try:
        raise WrongPythonException(1, 'Wrong Python version')
    except:
        pass

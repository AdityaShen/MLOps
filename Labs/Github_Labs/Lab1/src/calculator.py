def fun1(x, y):
    """
    Returns the average of two numbers.
    Args:
        x (int/float): First number.
        y (int/float): Second number.
    Returns:
        float: Average of x and y.
    Raises:
        ValueError: If x or y is not a number.
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    
    return (x + y) / 2


def fun2(x, y):
    """
    Returns the absolute difference between two numbers.
    Args:
        x (int/float): First number.
        y (int/float): Second number.
    Returns:
        int/float: Absolute difference of x and y.
    Raises:
        ValueError: If x or y is not a number.
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    
    return abs(x - y)


def fun3(x, y):
    """
    Raises x to the power of y.
    Args:
        x (int/float): Base.
        y (int/float): Exponent.
    Returns:
        int/float: x raised to the power y.
    Raises:
        ValueError: If x or y is not a number.
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    
    return x ** y


def fun4(x, y, z):
    """
    Returns the maximum of three numbers.
    Args:
        x (int/float): First number.
        y (int/float): Second number.
        z (int/float): Third number.
    Returns:
        int/float: Maximum of x, y, and z.
    Raises:
        ValueError: If any input is not a number.
    """
    if not all(isinstance(v, (int, float)) for v in (x, y, z)):
        raise ValueError("All inputs must be numbers.")
    
    return max(x, y, z)


# Example usage:
# f1_op = fun1(4, 6)      # average
# f2_op = fun2(10, 3)     # absolute difference
# f3_op = fun3(2, 3)      # power
# f4_op = fun4(f1_op, f2_op, f3_op)

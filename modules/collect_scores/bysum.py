def bysumfunc(list_of_scores):
    """
    Returns the sum of a list of numbers. 
    
    Parameters: 
    list_of_scores: The list of numbers of which the sum is to be computed. 
    
    Returns:
    sum: The sum of the list of numbers. 
    
    """
    sum = 0
    for score in list_of_scores:
        sum += score
    return sum
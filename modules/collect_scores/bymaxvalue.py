def bymaxvaluefunc(list_of_scores):
    """
    Returns max value from a list of numbers. 
    
    Parameters: 
    list_of_scores: The list of numbers of which twice the sum is to be computed. 
    
    Returns:
    maxvalue: The maximum value in a the list of numbers. 
    
    """
    maxvalue = 0
    for score in list_of_scores:
        if score > maxvalue:
            maxvalue = score
    return maxvalue
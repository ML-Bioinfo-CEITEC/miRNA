def bymeanfunc(list_of_scores): 
    """
    Returns the mean of a list of numbers. 
    
    Parameters: 
    list_of_scores: The list of numbers of which the mean is to be computed. 
    
    Returns:
    mean: The mean of the list of numbers. 
    
    """
    sum = 0
    for score in list_of_scores:
        sum += score
    mean = sum / len(list_of_scores)
    return mean 
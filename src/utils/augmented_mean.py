import numpy as np

def augmented_mean(data: list):

    print(data)
    
    if all(isinstance(i, (int, float, type(None))) for i in data):
        filtered_data = [i for i in data if i is not None]
        if not filtered_data:
            return None
        return sum(filtered_data) / len(filtered_data)
    
    if all(isinstance(i, list) for i in data):
        max_length = max(len(sublist) for sublist in data)
        means = []
        for i in range(max_length):
            elements = [sublist[i] for sublist in data if i < len(sublist) and isinstance(sublist[i], (int, float)) and sublist[i] is not None]
            elements = [e for e in elements if not e==None]
            
            if not elements:
                means.append(None)
            else:
                means.append(sum(elements) / len(elements))
        return means
    
    else:
        return [None] * len(data)
    



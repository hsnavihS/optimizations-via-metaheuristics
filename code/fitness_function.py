from basic_face_recognition import fr
 
 
def fitness_function(scale_factor):
    '''
    Universal fitness function for all search algorithms.
    If there is a label mismatch, return zero.
    Otherwise, return the confidence value.
    '''
 
    person_name, confidence = fr(scale_factor)
    return confidence, person_name
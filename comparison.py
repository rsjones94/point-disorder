
class BidirectionalDict(dict):
    """
    A dictionary-like object that maps relationships to bidirectional keys. That is,
    entries are accessed with a key [(i,j)], and such a key is assumed to be equivalent to [(j,i)]
    """
    pass
    # override methods for inserting and retrieving from dict
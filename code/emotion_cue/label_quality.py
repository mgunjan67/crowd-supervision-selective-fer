"""Fixed seven-class FER+ strict-majority mapping, independent of model outputs."""
import numpy as np
from .constants import CLASS_NAMES
VOTE_NAMES=('neutral','happiness','surprise','sadness','anger','disgust','fear','contempt','unknown','NF')
CANONICAL=('neutral','happy','surprise','sad','angry','disgust','fear',None,None,None)

def majority_label(votes):
    values=np.asarray(votes)
    if values.shape!=(10,) or not np.isfinite(values).all() or (values<0).any() or not np.equal(values,np.floor(values)).all():
        raise ValueError('Expected ten nonnegative integer vote counts')
    total=int(values.sum())
    winner=int(values.argmax())
    count=int(values[winner])
    eligible=total>0 and count*2>total and CANONICAL[winner] is not None
    return (CLASS_NAMES.index(CANONICAL[winner]) if eligible else None),count,total

def training_label(original, majority, partition):
    if original not in range(7): raise ValueError('Invalid original category')
    if partition not in ('Training','PublicTest','PrivateTest'): raise ValueError('Invalid partition')
    if majority is not None and majority not in range(7): raise ValueError('Invalid majority category')
    return majority if partition=='Training' and majority is not None else original

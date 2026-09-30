import pytest
from emotion_cue.label_quality import majority_label, training_label

def test_majority_maps_to_canonical_order():
    assert majority_label([0,7,0,0,3,0,0,0,0,0])==(3,7,10)
    assert majority_label([6,0,0,0,0,0,0,0,4,0])==(4,6,10)

@pytest.mark.parametrize('column,expected',[(0,4),(1,3),(2,6),(3,5),(4,0),(5,1),(6,2)])
def test_every_ferplus_category_maps_explicitly(column,expected):
    votes=[0]*10; votes[column]=6; votes[8]=4
    assert majority_label(votes)[0]==expected

@pytest.mark.parametrize('votes',[
    [5,5,0,0,0,0,0,0,0,0], [4,3,3,0,0,0,0,0,0,0],
    [0]*10, [0,0,0,0,0,0,0,6,4,0], [0,0,0,0,4,0,0,0,0,6]])
def test_ambiguous_or_out_of_scope_votes_not_forced(votes):
    assert majority_label(votes)[0] is None

def test_all_votes_count_in_denominator():
    assert majority_label([0,4,0,0,0,0,0,0,3,3])[0] is None

def test_only_training_targets_change():
    assert training_label(0,3,'Training')==3
    assert training_label(0,None,'Training')==0
    assert training_label(0,3,'PublicTest')==0
    assert training_label(0,3,'PrivateTest')==0

@pytest.mark.parametrize('votes',[[0]*9,[-1]+[0]*9,[.1]+[0]*9])
def test_malformed_votes_rejected(votes):
    with pytest.raises(ValueError): majority_label(votes)

from emotion_cue.data_contract import deterministic_split_indices


def test_split_is_deterministic_and_disjoint():
    train_a, val_a = deterministic_split_indices(100, 0.1, 42)
    train_b, val_b = deterministic_split_indices(100, 0.1, 42)
    assert (train_a, val_a) == (train_b, val_b)
    assert len(train_a) == 90
    assert len(val_a) == 10
    assert set(train_a).isdisjoint(val_a)
    assert sorted(train_a + val_a) == list(range(100))


def test_different_seed_changes_membership():
    assert deterministic_split_indices(40, 0.2, 1) != deterministic_split_indices(
        40, 0.2, 2
    )

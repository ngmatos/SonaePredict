from sklearn.model_selection import train_test_split


def get_sample(data: object, target: object) -> object:
    print('Using Random Split for evaluating estimator performance')
    return train_test_split(data, target, test_size=0.3, random_state=42)

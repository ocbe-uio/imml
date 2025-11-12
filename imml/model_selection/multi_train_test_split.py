import numpy as np
from sklearn.model_selection import train_test_split


def multi_train_test_split_Xs(*args, **kwargs):
    """
    Split multi-modal datasets and labels into train and test sets.

    Similar to sklearn's `train_test_split
    <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>`_, but works with
    lists of arrays/data (Xs) and single arrays (y). Ensures that all X in a Xs get the same train/test split indices.

    Parameters
    ----------
    *args : list of array-likes or array-like
        Variable number of inputs to split. Can be:
        - Lists of arrays (Xs): Multi-modal data where each element is a modality.
        - Single arrays (y): Labels.

    **kwargs : dict
        Additional keyword arguments to pass to sklearn's train_test_split.
    Returns
    -------
    tuple
        Splitting results in the same order as inputs:
        - For each list input (Xs): (list_train, list_test)
        - For each array input (y): (array_train, array_test)

    Example
    --------
    >>> import numpy as np
    >>> from imml.model_selection import multi_train_test_split_Xs
    >>> Xs = [np.random.rand(100, 10), np.random.rand(100, 20)]
    >>> y = np.random.randint(0, 2, 100)
    >>> Xs_train, Xs_test, y_train, y_test = multi_train_test_split_Xs(Xs, y, train_size=0.7, random_state=42)
    """
    output = []

    for Xs in args:
        if isinstance(Xs, list):
            train_list = []
            test_list = []

            if ('random_state' not in kwargs) or (kwargs["random_state"] is None):
                random_state = np.random.randint(0, 2 ** 31 - 1)
                kwargs = {**kwargs, 'random_state': random_state}
            else:
                kwargs = kwargs

            for X in Xs:
                X_train, X_test = train_test_split(X, **kwargs)
                train_list.append(X_train)
                test_list.append(X_test)

            output.extend([train_list, test_list])

        else:
            X_train, X_test = train_test_split(Xs, **kwargs)
            output.extend([X_train, X_test])

    return tuple(output)

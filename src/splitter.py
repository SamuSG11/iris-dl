from sklearn.model_selection import train_test_split

class DatasetSplitter:
    """
    Classe per split del dataset in Train / Validation / Test (70-15-15)
    """

    def __init__(self, test_size=0.30, val_size=0.50, random_state=42, stratify=True):
        """
        test_size = quota iniziale di split (default 0.30 -> 30% temp set)
        val_size = quota del temp set che diventa validation (default 0.50)
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.stratify = stratify

    def split(self, X, y):
        """
        Restituisce:
        X_train, X_val, X_test, y_train, y_val, y_test
        """

        strat1 = y if self.stratify else None

        # 1. Train / Temp (70 / 30)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=strat1
        )

        strat2 = y_temp if self.stratify else None

        # 2. Temp → Validation / Test (15 / 15)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=strat2
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
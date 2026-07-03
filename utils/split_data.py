from sklearn.model_selection import train_test_split


def create_train_valid_split(df):

    train_df, valid_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True)
    )
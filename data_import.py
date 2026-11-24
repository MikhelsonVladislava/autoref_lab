import pandas as pd

def get_train_test_gazeta(link=None, test_link=None):

    dataset = None
    if link is None:
        link1 = "hf://datasets/IlyaGusev/gazeta@refs/convert/parquet/default/train/0000.parquet"
        link2 = "hf://datasets/IlyaGusev/gazeta@refs/convert/parquet/default/train/0001.parquet"

        dataset = pd.concat([pd.read_parquet(link1, columns=['text', 'summary']), pd.read_parquet(link2, columns=['text', 'summary'])])
        dataset.to_parquet('train.parquet')
    else:
        dataset = pd.read_parquet(link, columns=['text', 'summary'])

    if test_link is None:
        test_link = "hf://datasets/IlyaGusev/gazeta@refs/convert/parquet/default/test/0000.parquet"
    
    test_dataset = pd.read_parquet(test_link, columns=['text', 'summary'])
    test_dataset.to_parquet('test.parquet')
    
    return dataset, test_dataset
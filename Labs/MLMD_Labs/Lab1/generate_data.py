from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import os

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name='target')
data = pd.concat([X, y], axis=1)

train, remainder = train_test_split(data, train_size=0.6, random_state=42)
eval_data, serving = train_test_split(remainder, test_size=0.5, random_state=42)

for name, df in [('train', train), ('eval', eval_data), ('serving', serving)]:
    path = os.path.join('data', name)
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, 'data.csv'), index=False)
    print(f'{name}: {len(df)} rows saved')
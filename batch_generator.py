import numpy as np
import pandas as pd

def generate_batch(data, batch_size, part_correct=0.5):
    num_correct_indexes = int(batch_size * part_correct)
    true_data = data[data['label'] == 1].sample(num_correct_indexes)
    false_data = data[data['label'] == 0].sample(batch_size - num_correct_indexes)
    return pd.concat((true_data, false_data), axis=0, ignore_index=True).sample(frac=1)
import pandas as pd

def write_split(df, folds, split_name, invert=False):
    selection = df.fold_num.isin(folds)
    selection = ~selection if invert else selection
    df_selected = df[selection][['seq', 'mean_value']]
    print(split_name, df_selected.shape)
    print(df_selected.head(), '\n')
    df_selected.to_csv(f'WTC11_{split_name}.tsv', sep='\t', index=False)

df = pd.read_csv('WTC11_folds.tsv', sep='\t')
df.columns = ['seq_id', 'seq', 'mean_value', 'fold_num', 'rev'][0:len(df.columns)]
df = df[df.rev == 0]
# 
# df = df[['seq', 'mean_value']]
print('folds', df.shape)
print(df.head(), '\n')
write_split(df, [1], 'test')
write_split(df, [2], 'valid')
write_split(df, [1, 2], 'train', invert=True)



def permute_columns_subgroup(data, columns, subgroup, rng):
    for column in columns:
        mask = subgroup.representation
        data.loc[mask, (column,)] = rng.permutation(data.loc[mask, (column,)])


def negate_columns_subgroup(data, columns, subgroup):
    for column in columns:
        mask = subgroup.representation
        data.loc[mask, (column,)] = -data.loc[mask, (column,)]


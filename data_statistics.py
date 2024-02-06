import pandas as pd
from matplotlib import pyplot as plt

from general import load_info


__all__ = (
    'calculate_line',
    'calculate_diagnose_bars',
    'calculate_density'
)


def calculate_line(df: pd.DataFrame):
    print('- - Calculating lines')
    all_diagnoses = df['diagnose'].unique()
    print('Column', end='')
    for name in df.columns:
        if any((
            name in {'sex', 'age', 'diagnose'},
            '/' in name
        )):
            continue
        print(f' {name}', end='')
        df.sort_values(name, inplace=True, ignore_index=True)
        f, ax = plt.subplots(figsize=(19.20, 10.80))
        diagnoses = {i: ([], []) for i in all_diagnoses}
        for row_id, series in df.iterrows():
            if series[name] is None:
                continue
            lists = diagnoses[series['diagnose']]
            lists[0].append(row_id)
            lists[1].append(series[name])
        for diagnose, (x, y) in diagnoses.items():
            ax.scatter(x, y, label=diagnose, s=11)

        ax: plt.Axes
        ax.set_title(name)
        name = name.replace(' ', '_').replace('/', '_')
        plt.legend(all_diagnoses)
        plt.savefig(f"graphs\\line\\{name}.png")
        plt.close('all')
    print()


def calculate_diagnose_bars(df: pd.DataFrame) -> pd.Series:
    print('- - Calculating diagnose bars')
    plt.close()
    ax = plt.subplot()
    values = df['diagnose'].value_counts()
    ax.bar(x=values.index, height=values.values)
    ax.set_ylabel('Количество')
    ax.set_xlabel('Диагноз')
    plt.setp(ax.get_xticklabels(), rotation=10, ha='right')
    plt.tight_layout()
    plt.savefig('graphs\\diagnose_bars.png')
    return values


def calculate_density(df: pd.DataFrame):
    raise NotImplementedError()
    print('- - Calculating density bars')
    print('Column', end='')
    for name in df.columns:
        print(f' {name}', end='')
        ax = df[name].plot(kind='kde')
        ax: plt.Axes
        ax.set_title(name)
        name = name.replace(' ', '_').replace('/', '_')
        plt.savefig(f"graphs\\density\\{name}.png")
        plt.close('all')
    print()


def main():
    df = load_info()
    calculate_diagnose_bars(df)
    calculate_line(df)
    # calculate_density(df)


if __name__ == '__main__':
    main()

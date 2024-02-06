import pandas as pd
from openpyxl import load_workbook, Workbook

__all__ = ('load_info',)


def load_info() -> pd.DataFrame:
    wb: Workbook = load_workbook(filename='ПАТТЕРН ОБЕЗЛИЧ..xlsx')
    columns = [
        'sex', 'age',
        'IN T', 'IN P', 'IN P1/P', 'IN P2/P', 'IN P3/P', 'IN P1', 'IN P2', 'IN P3',
        'OUT T', 'OUT P', 'OUT P1/P', 'OUT P2/P', 'OUT P3/P', 'OUT P1', 'OUT P2', 'OUT P3',
        'SUM T', 'SUM P', 'SUM P1/P', 'SUM P2/P', 'SUM P3/P', 'SUM P1', 'SUM P2', 'SUM P3',
        'diagnose'
    ]
    iterator = wb.worksheets[0]
    iterator = (row for row in iterator.iter_rows(min_row=3, min_col=0, max_col=len(columns)))
    iterator = ([i.value for i in row] for row in iterator)
    df = pd.DataFrame(iterator, columns=columns)
    return df

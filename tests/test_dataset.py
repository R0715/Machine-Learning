"""用真实交接数据检验日期边界、标签及导出，使用 unittest 无额外测试依赖。"""
from pathlib import Path
from decimal import Decimal
import csv
import unittest
import numpy as np
import pandas as pd
from src.config import Config
from src.dataset import build_dataset
from src.features import build_features
from build_dataset import read_table

ROOT=Path(__file__).resolve().parents[1]


class DatasetChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices=read_table(ROOT/'outputs/tables/clean_prices.csv')
        cls.features=read_table(ROOT/'outputs/tables/raw_features.csv')
        cls.data=build_dataset(cls.prices,cls.features,Config())

    def test_counts_and_no_overlap(self):
        sets=[set(getattr(self.data,f'X_{s}').index) for s in ('train','val','test')]
        self.assertEqual([len(s) for s in sets],[3465,242,251])
        for i in range(3):
            for j in range(i+1,3):
                self.assertFalse(sets[i]&sets[j])

    def test_every_exported_label_independent_decimal(self):
        with (ROOT/'outputs/tables/clean_prices.csv').open(encoding='utf-8-sig') as f:
            records=list(csv.DictReader(f))
        expected={records[i]['date']:(Decimal(records[i+1]['close'])/Decimal(records[i]['close'])-1,
                                       records[i+1]['date']) for i in range(len(records)-1)}
        for split in ('train','val','test'):
            folder=ROOT/'dataset'/split
            y=read_table(folder/f'y_{split}.csv')
            meta=read_table(folder/f'metadata_{split}.csv')
            for date,value in y.y_true_ret.items():
                reference,target=expected[str(date.date())]
                self.assertAlmostEqual(value,float(reference),places=13)
                self.assertEqual(meta.loc[date,'target_date'],target)

    def test_exports_preserve_B_values(self):
        for split in ('train','val','test'):
            x=read_table(ROOT/'dataset'/split/f'X_{split}.csv')
            y=read_table(ROOT/'dataset'/split/f'y_{split}.csv')
            pd.testing.assert_frame_equal(x,self.features.loc[x.index],check_exact=False,atol=1e-14,rtol=1e-12)
            self.assertTrue(x.index.equals(y.index))
            self.assertEqual(x.shape[1],17)
            self.assertTrue(np.isfinite(x.to_numpy()).all())

    def test_excluded_rows_account_for_every_input_row(self):
        excluded=read_table(ROOT/'dataset/excluded_rows.csv')
        assigned=set().union(*(set(getattr(self.data,f'X_{s}').index) for s in ('train','val','test')))
        self.assertFalse(assigned & set(excluded.index))
        self.assertEqual(assigned | set(excluded.index),set(self.prices.index))
        self.assertEqual(excluded.loc['2024-10-10','reason'],'label_crosses_train_val_boundary')
        self.assertEqual(excluded.loc['2026-10-23','reason'],'end_date_no_next_execution')

    def test_strict_target_date_boundaries(self):
        meta=self.data.metadata
        self.assertTrue((meta.loc[self.data.X_train.index,'target_date']<pd.Timestamp('2024-10-11')).all())
        self.assertTrue((meta.loc[self.data.X_val.index,'target_date']<=pd.Timestamp('2025-10-10')).all())
        self.assertEqual(meta.loc[self.data.X_test.index,'target_date'].min(),pd.Timestamp('2025-10-13'))
        self.assertEqual(meta.loc[self.data.X_test.index,'target_date'].max(),pd.Timestamp('2026-10-23'))

    def test_future_changes_do_not_change_historical_training(self):
        changed=self.prices.copy()
        changed.loc[changed.index>'2025-10-10',['open','high','low','close']]*=5
        revised=build_dataset(changed,build_features(changed),Config())
        for split in ('train','val'):
            pd.testing.assert_frame_equal(getattr(self.data,f'X_{split}'),getattr(revised,f'X_{split}'),check_exact=False,atol=1e-13,rtol=1e-11)
            pd.testing.assert_series_equal(getattr(self.data,f'y_{split}'),getattr(revised,f'y_{split}'))

    def test_horizon_five_does_not_invent_end_labels(self):
        data=build_dataset(self.prices,self.features,Config(horizon=5))
        self.assertEqual(len(data.X_train),3461)
        self.assertEqual(len(data.X_val),238)
        self.assertEqual(len(data.X_test),251)
        self.assertEqual(data.y_test.isna().sum(),4)
        self.assertTrue(data.y_test.iloc[-4:].isna().all())
        self.assertTrue((data.metadata.loc[data.X_val.index,'target_date']<=pd.Timestamp('2025-10-10')).all())

    def test_missing_feature_date_rejected(self):
        with self.assertRaisesRegex(ValueError,'完全对齐'):
            build_dataset(self.prices,self.features.drop(self.features.index[100]),Config())

    def test_interior_missing_value_rejected(self):
        features=self.features.copy()
        features.iloc[100,0]=np.nan
        with self.assertRaisesRegex(ValueError,'预热结束'):
            build_dataset(self.prices,features,Config())

    def test_infinity_rejected(self):
        features=self.features.copy()
        features.iloc[100,0]=np.inf
        with self.assertRaisesRegex(ValueError,'无穷值'):
            build_dataset(self.prices,features,Config())

    def test_future_target_feature_rejected(self):
        features=self.features.copy()
        features['target_close']=self.prices.close.shift(-1)
        with self.assertRaisesRegex(ValueError,'标签'):
            build_dataset(self.prices,features,Config())

    def test_invalid_horizon_rejected(self):
        with self.assertRaisesRegex(ValueError,'正整数'):
            build_dataset(self.prices,self.features,Config(horizon=0))


if __name__=='__main__':
    unittest.main()

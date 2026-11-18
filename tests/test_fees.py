import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from src.config import Config
from src.backtest import run_backtest, transaction_fees, rebalance_notional


class FeeTests(unittest.TestCase):
    def setUp(self):
        self.cfg=Config(cutoff='2025-01-02',end='2025-01-08')
        self.dates=pd.bdate_range(self.cfg.cutoff,self.cfg.end,name='date')
        self.prices=pd.DataFrame({'open':100.,'close':100.},index=self.dates)

    def run_weights(self,weights,cfg=None):
        return run_backtest(self.prices,cfg or self.cfg,pd.Series(weights,index=self.dates))

    def test_hand_calculated_small_and_large_orders(self):
        self.assertAlmostEqual(transaction_fees(10000,'BUY',self.cfg)['total_fees'],5.1)
        self.assertAlmostEqual(transaction_fees(10000,'SELL',self.cfg)['total_fees'],10.1)
        self.assertAlmostEqual(transaction_fees(100000,'BUY',self.cfg)['total_fees'],31)
        self.assertAlmostEqual(transaction_fees(100000,'SELL',self.cfg)['total_fees'],81)
        self.assertEqual(transaction_fees(0,'SELL',self.cfg)['total_fees'],0)

    def test_full_buy_reserves_cash_and_does_not_retrade(self):
        result=self.run_weights([1.]*5)
        expected=100000/1.00031
        self.assertAlmostEqual(result.trades.iloc[0].notional,expected,places=8)
        self.assertAlmostEqual(result.metrics['final_equity'],expected,places=8)
        self.assertEqual(len(result.trades),1)
        self.assertTrue((result.equity.cash>=0).all())
        self.assertEqual(result.trades.iloc[0].stamp_tax,0)

    def test_full_roundtrip_flat_market_loses_exact_fees(self):
        result=self.run_weights([1,0,0,0,0])
        buy=100000/1.00031
        expected=buy*(1-.00081)
        self.assertAlmostEqual(result.metrics['final_equity'],expected,places=8)
        self.assertAlmostEqual(100000-expected,result.metrics['total_fees'],places=8)
        self.assertEqual(list(result.trades.side),['BUY','SELL'])
        self.assertTrue((result.trades.trade_date>result.trades.signal_date).all())

    def test_minimum_commission_full_buy(self):
        result=self.run_weights([1.]*5,replace(self.cfg,initial_cash=1000))
        expected=(1000-5)/1.00001
        self.assertAlmostEqual(result.trades.iloc[0].notional,expected,places=8)
        self.assertEqual(result.trades.iloc[0].commission,5)

    def test_half_weight_is_after_fees(self):
        result=self.run_weights([.5]*5)
        self.assertEqual(len(result.trades),1)
        self.assertAlmostEqual(result.trades.iloc[0].actual_weight_after,.5)

    def test_no_trade_has_no_fees(self):
        result=self.run_weights([0]*5)
        self.assertTrue(result.trades.empty)
        self.assertEqual(result.metrics['total_fees'],0)
        self.assertEqual(result.metrics['final_equity'],100000)

    def test_unaffordable_tiny_account(self):
        result=self.run_weights([1]*5,replace(self.cfg,initial_cash=4))
        self.assertTrue(result.trades.empty)
        self.assertEqual(result.metrics['final_equity'],4)
        self.assertEqual(rebalance_notional(0,4,0,self.cfg),0)
        self.assertEqual(rebalance_notional(0,5/(1-.00051),0,self.cfg),0)

    def test_zero_fee_compatibility(self):
        cfg=replace(self.cfg,commission_rate=0,min_commission=0,transfer_fee_rate=0,stamp_tax_rate=0)
        result=self.run_weights([1,0,.5,1,0],cfg)
        self.assertEqual(result.metrics['final_equity'],100000)
        self.assertEqual(result.metrics['total_fees'],0)

    def test_random_orders_against_independent_cash_ledger(self):
        rng=np.random.default_rng(7)
        for _ in range(1000):
            cash=float(rng.uniform(0,100000));hold=float(rng.uniform(0,100000));w=float(rng.uniform(0,1))
            signed=rebalance_notional(cash,hold,w,self.cfg)
            if signed==0:continue
            n=abs(signed)
            # 独立直接应用课程费率，不调用收费函数来验证自身。
            expected_fee=max(5,n*.0003)+n*.00001+(n*.0005 if signed<0 else 0)
            new_cash=cash-signed-expected_fee
            new_hold=hold+signed
            self.assertGreaterEqual(new_cash,-1e-8)
            self.assertGreaterEqual(new_hold,-1e-8)
            self.assertAlmostEqual(new_cash+new_hold,cash+hold-expected_fee,places=7)
            self.assertAlmostEqual(new_hold/(new_cash+new_hold),w,places=11)

    def test_bad_fee_config_rejected(self):
        with self.assertRaisesRegex(ValueError,'税费参数'):
            self.run_weights([1]*5,replace(self.cfg,commission_rate=-.01))


if __name__=='__main__':unittest.main()

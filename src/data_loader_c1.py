from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class TwoStreamDataset(Dataset):
    def __init__(self, df, window_size=21):
        self.window_size = window_size
        self.day_avg = df['DayAvgPrice'].values
        self.intraday_std = df['IntradayStd'].values
        self.volume = df['Volume'].values
        self.log_profit = df['Log_Profit'].values
        self.day_of_week = df['day_of_week'].values
        self.day_of_year = df['day_of_year'].values
        self.day_avg_diff = df['DayAvgPrice_diff'].values
        self.day_avg_2diff = df['DayAvgPrice_2diff'].values
        self.lambda_C1 = df['lambda_C1'].values
        self.lambda_C2 = df['lambda_C2'].values
        self.lambda_C3 = df['lambda_C3'].values
        self.lambda_T1 = df['lambda_T1'].values
        self.lambda_T2 = df['lambda_T2'].values
        self.lambda_T3 = df['lambda_T3'].values
        self.poly_1 = df['POLY_1'].values
        self.poly_2 = df['POLY_2'].values
        self.poly_3 = df['POLY_3'].values
        self.dap_1 = df['DAP_1'].values
        self.dap_2 = df['DAP_2'].values
        self.dap_3 = df['DAP_3'].values
        self.dap_4 = df['DAP_4'].values
        self.dap_5 = df['DAP_5'].values
        self.dap_6 = df['DAP_6'].values
        self.dap_7 = df['DAP_7'].values
        self.dap_10 = df['DAP_10'].values
        self.open = df['open'].values
        self.high = df['high'].values
        self.low = df['low'].values
        self.close = df['close'].values
        self.parkinson_vol = df['parkinson_vol'].values
        self.parkinson_vol_ma5 = df['parkinson_vol_ma5'].values
        self.parkinson_vol_ma20 = df['parkinson_vol_ma20'].values
        self.parkinson_vol_diff1 = df['parkinson_vol_diff1'].values
        self.parkinson_vol_lag1 = df['parkinson_vol_lag1'].values
        self.day_avg_roll5 = df['DayAvgPrice_roll5'].values
        self.day_avg_roll10 = df['DayAvgPrice_roll10'].values
        self.day_avg_roll20 = df['DayAvgPrice_roll20'].values
        self.day_avg_ema5 = df['DayAvgPrice_ema5'].values
        self.day_avg_ema20 = df['DayAvgPrice_ema20'].values
        self.intraday_std_roll5 = df['IntradayStd_roll5'].values
        self.intraday_std_roll10 = df['IntradayStd_roll10'].values
        self.intraday_std_roll20 = df['IntradayStd_roll20'].values
        self.intraday_std_ema5 = df['IntradayStd_ema5'].values
        self.intraday_std_ema20 = df['IntradayStd_ema20'].values
        self.log_ret_1 = df['log_ret_1'].values
        self.log_ret_2 = df['log_ret_2'].values
        self.log_ret_3 = df['log_ret_3'].values
        self.log_ret_4 = df['log_ret_4'].values
        self.log_ret_5 = df['log_ret_5'].values
        self.log_ret_6 = df['log_ret_6'].values
        self.log_ret_7 = df['log_ret_7'].values
        self.log_ret_10 = df['log_ret_10'].values
        self.mean_w = df['mean_w'].values
        self.std_w = df['std_w'].values
        self.z_w = df['z_w'].values
        self.q10_w = df['q10_w'].values
        self.q90_w = df['q90_w'].values
        self.slope_w = df['slope_w'].values
        self.vol_w = df['vol_w'].values
        self.vol_z = df['vol_z'].values
        self.dow_sin = df['dow_sin'].values
        self.dow_cos = df['dow_cos'].values
        self.moy_sin = df['moy_sin'].values
        self.moy_cos = df['moy_cos'].values
        self.rsi14 = df['rsi14'].values
        self.macd = df['macd'].values
        self.macd_signal = df['macd_signal'].values
        self.macd_hist = df['macd_hist'].values
        self.bb_low = df['bb_low'].values
        self.bb_lmid = df['bb_mid'].values
        self.bb_up = df['bb_up'].values
        self.atr14 = df['atr14'].values
        self.phi_hilbert = df['phi_hilbert'].values
        self.dphi_hilbert = df['dphi_hilbert'].values
        self.stft_energy_low = df['stft_energy_low'].values
        self.stft_energy_mid = df['stft_energy_mid'].values
        self.stft_energy_high = df['stft_energy_high'].values
        self.gk_sigma = df['gk_sigma'].values
        self.rs_sigma = df['rs_sigma'].values
        self.yz_sigma = df['yz_sigma'].values
        self.adx = df['adx'].values
        self.chop = df['chop'].values
        self.kalm_slope = df['kalm_slope'].values
        self.rv20 = df['rv20'].values
        self.vol_of_vol = df['vol_of_vol'].values
        self.bb_pct_b = df['bb_pct_b'].values
        self.bb_bandwidth = df['bb_bandwidth'].values
        self.ret_overnight = df['ret_overnight'].values
        self.ret_intraday = df['ret_intraday'].values
        self.corr_ret_dlogvol = df['corr_ret_dlogvol'].values
        self.c_week_real = df['c_week_real'].values
        self.c_month_real = df['c_month_real'].values
        self.c_quarter_real = df['c_quarter_real'].values
        self.c_year_real = df['c_year_real'].values
        self.c_week_imag = df['c_week_imag'].values
        self.c_month_imag = df['c_month_imag'].values
        self.c_quarter_imag = df['c_quarter_imag'].values
        self.c_year_imag = df['c_year_imag'].values
        self.real_t = df['real_time'].values
        self.imag_t = df['imag_time'].values
        self.days_since_prev = df['days_since_prev'].values

        self.df = df.reset_index(drop=True)

        self.target = df['Target'].values

    def __len__(self):
        return len(self.day_avg) - self.window_size

    def __getitem__(self, idx):
        slice_ = slice(idx, idx + 1 + self.window_size)
        real_feats = np.column_stack([
            self.day_avg[slice_],
            self.intraday_std[slice_],
            self.volume[slice_],
            self.day_of_week[slice_],
            self.day_of_year[slice_],
            self.log_profit[slice_],
            self.day_avg_diff[slice_],
            self.day_avg_2diff[slice_],
            self.lambda_C1[slice_],
            self.lambda_C2[slice_],
            self.lambda_C3[slice_],
            self.lambda_T1[slice_],
            self.lambda_T2[slice_],
            self.lambda_T3[slice_],
            # self.poly_1[slice_],
            # self.poly_2[slice_],
            # self.poly_3[slice_],
            # self.dap_1[slice_],
            # self.dap_2[slice_],
            # self.dap_3[slice_],
            # self.dap_4[slice_],
            # self.dap_5[slice_],
            # self.dap_6[slice_],
            # self.dap_7[slice_],
            # self.dap_10[slice_],
            # self.open[slice_],
            # self.high[slice_],
            # self.low[slice_],
            # self.close[slice_],
            self.parkinson_vol[slice_],
            self.parkinson_vol_ma5[slice_],
            self.parkinson_vol_ma20[slice_],
            self.parkinson_vol_diff1[slice_],
            self.parkinson_vol_lag1[slice_],
            # self.day_avg_roll5[slice_],
            # self.day_avg_roll10[slice_],
            # self.day_avg_roll20[slice_],
            # self.day_avg_ema5[slice_],
            # self.day_avg_ema20[slice_],
            self.intraday_std_roll5[slice_],
            self.intraday_std_roll10[slice_],
            self.intraday_std_roll20[slice_],
            self.intraday_std_ema5[slice_],
            self.intraday_std_ema20[slice_],
            self.log_ret_1[slice_],
            self.log_ret_2[slice_],
            self.log_ret_3[slice_],
            self.log_ret_4[slice_],
            self.log_ret_5[slice_],
            self.log_ret_6[slice_],
            self.log_ret_7[slice_],
            self.log_ret_10[slice_],
            self.mean_w[slice_],
            self.std_w[slice_],
            self.z_w[slice_],
            self.q10_w[slice_],
            self.q90_w[slice_],
            self.slope_w[slice_],
            self.vol_w[slice_],
            self.vol_z[slice_],
            self.dow_sin[slice_],
            self.dow_cos[slice_],
            self.moy_sin[slice_],
            self.moy_cos[slice_],
            self.rsi14[slice_],
            self.macd[slice_],
            self.macd_signal[slice_],
            self.macd_hist[slice_],
            self.bb_low[slice_],
            self.bb_lmid[slice_],
            self.bb_up[slice_],
            self.atr14[slice_],
            self.phi_hilbert[slice_],
            self.dphi_hilbert[slice_],
            self.stft_energy_low[slice_],
            self.stft_energy_mid[slice_],
            self.stft_energy_high[slice_],
            self.gk_sigma[slice_],
            self.rs_sigma[slice_],
            self.yz_sigma[slice_],
            self.adx[slice_],
            self.chop[slice_],
            self.kalm_slope[slice_],
            self.rv20[slice_],
            self.vol_of_vol[slice_],
            self.bb_pct_b[slice_],
            self.bb_bandwidth[slice_],
            self.ret_overnight[slice_],
            self.ret_intraday[slice_],
            self.corr_ret_dlogvol[slice_],
            self.days_since_prev[slice_],
        ])  # shape: (window_size, num_real_features)


        x_ct_real = np.column_stack([
            self.c_week_real[slice_],
            self.c_month_real[slice_],
            self.c_quarter_real[slice_],
            self.c_year_real[slice_],
            self.real_t[slice_],
        ])  # shape: (window_size, 5)

        x_ct_imag = np.column_stack([
            self.c_week_imag[slice_],
            self.c_month_imag[slice_],
            self.c_quarter_imag[slice_],
            self.c_year_imag[slice_],
            self.imag_t[slice_],
        ])  # shape: (window_size, 5)

        y = self.target[idx + self.window_size]

        if len(self.df) < 500:

            feats_dates_start = self.df['date'].iloc[slice_].tolist()[:3]
            feats_dates_end = self.df['date'].iloc[slice_].tolist()[-3:]
            target_date = self.df['date'].iloc[idx + self.window_size]
            #print(f"Фичи: {feats_dates_start} ... {feats_dates_end}, Таргет: {target_date}")


        return {
            'real_feats': torch.tensor(real_feats, dtype=torch.float32),
            'complex_time_real': torch.tensor(x_ct_real, dtype=torch.float32),
            'complex_time_imag': torch.tensor(x_ct_imag, dtype=torch.float32),
            'target': torch.tensor(y, dtype=torch.float32),
            "row_idx": idx
        }

def load_dataloader(df, window_size, batch_size, shuffle, drop_last=False):
    dataset = TwoStreamDataset(df, window_size=window_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return loader
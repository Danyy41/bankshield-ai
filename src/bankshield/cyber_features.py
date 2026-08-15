"""Causal cyber-risk feature engineering (Phase 2).

For every transaction, aggregate that customer's login history into a
handful of risk features -- but only ever looking at logins that
happened *strictly before* the transaction's own timestamp. This is the
same causality discipline Phase 1 applies to `transaction_velocity_24h`
and `amount_to_avg_ratio`: a feature must only use information that
would actually have been available at the moment the transaction
occurred, or it isn't a legitimate feature -- it's leakage.

Features produced (see config.CYBER_FEATURE_COLUMNS):

- `cyber_failed_logins_1h` -- failed login attempts by this customer in
  the hour before the transaction.
- `cyber_login_count_24h` -- total login attempts (success + failure) in
  the trailing 24h.
- `cyber_minutes_since_last_login` -- gap between the transaction and the
  customer's most recent login (capped; see `NO_RECENT_LOGIN_MINUTES`).
- `cyber_new_device_recent` -- whether the most recent login (within
  `CYBER_RECENT_LOGIN_LOOKBACK_HOURS`) was from an unfamiliar device.
- `cyber_unusual_country_recent` -- same, for an unfamiliar country.
- `cyber_recent_suspicious_auth` -- composite flag: several recent failed
  attempts *and* the most recent login was from a new device/location --
  i.e. the account-takeover shape (failed attempts, then a suspicious
  success) rather than either signal alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bankshield import config

NO_RECENT_LOGIN_MINUTES = 7 * 24 * 60  # sentinel: one week, used when there's no prior login


def add_cyber_features(transactions_df: pd.DataFrame, logins_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `transactions_df` with cyber_* columns appended."""
    df = transactions_df.copy().reset_index(drop=True)
    txn_ts = df["timestamp"].to_numpy(dtype="datetime64[ns]")

    logins = logins_df.sort_values("timestamp")
    login_ts_all = logins["timestamp"].to_numpy(dtype="datetime64[ns]")
    login_fail_all = ~logins["login_success"].to_numpy()
    login_new_device_all = logins["new_device"].to_numpy()
    login_new_location_all = logins["new_location"].to_numpy()
    login_customer_all = logins["customer_id"].to_numpy()

    n = len(df)
    failed_1h = np.zeros(n, dtype=int)
    login_count_24h = np.zeros(n, dtype=int)
    minutes_since_last = np.full(n, NO_RECENT_LOGIN_MINUTES, dtype=float)
    new_device_recent = np.zeros(n, dtype=bool)
    unusual_country_recent = np.zeros(n, dtype=bool)

    one_hour = np.timedelta64(1, "h")
    lookback_window = np.timedelta64(config.CYBER_RECENT_LOGIN_LOOKBACK_HOURS, "h")
    count_window = np.timedelta64(config.CYBER_LOGIN_COUNT_WINDOW_HOURS, "h")

    for customer_id, group_idx in df.groupby("customer_id").indices.items():
        cust_login_mask = login_customer_all == customer_id
        cust_login_ts = login_ts_all[cust_login_mask]
        cust_login_fail = login_fail_all[cust_login_mask]
        cust_new_device = login_new_device_all[cust_login_mask]
        cust_new_location = login_new_location_all[cust_login_mask]

        for row_i in group_idx:
            t = txn_ts[row_i]
            pos = np.searchsorted(cust_login_ts, t, side="left")
            if pos == 0:
                continue  # no prior login for this customer at all

            prior_ts = cust_login_ts[:pos]

            failed_1h[row_i] = int(np.sum(cust_login_fail[:pos][prior_ts >= t - one_hour]))
            login_count_24h[row_i] = int(np.sum(prior_ts >= t - count_window))

            last_idx = pos - 1
            gap_minutes = (t - cust_login_ts[last_idx]) / np.timedelta64(1, "m")
            if gap_minutes <= config.CYBER_RECENT_LOGIN_LOOKBACK_HOURS * 60:
                minutes_since_last[row_i] = gap_minutes
                new_device_recent[row_i] = bool(cust_new_device[last_idx])
                unusual_country_recent[row_i] = bool(cust_new_location[last_idx])

    recent_suspicious_auth = (failed_1h >= 2) & (new_device_recent | unusual_country_recent)

    df["cyber_failed_logins_1h"] = failed_1h
    df["cyber_login_count_24h"] = login_count_24h
    df["cyber_minutes_since_last_login"] = minutes_since_last
    df["cyber_new_device_recent"] = new_device_recent
    df["cyber_unusual_country_recent"] = unusual_country_recent
    df["cyber_recent_suspicious_auth"] = recent_suspicious_auth

    return df

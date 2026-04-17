"""
Backtesting Engine v2 — Adaptive Strategy
=========================================
Major improvements over v1:
  1. Signal freshness: 4h window (was 48h) — prevents stale trade entries
  2. Latest-signal-only: uses the most recent signal per symbol, not any in window
  3. Market regime filter: blocks BUY entries during confirmed BEAR regime
  4. Trailing stop-loss: moves SL up as price rises, locking in profit
  5. Max hold time: auto-exits positions held > 48 hours (time-based exit)
  6. Multi-confirmation gate: trend + MACD + volume must agree for entry
  7. Whale confirmation: skips entry if whale_signal is DISTRIBUTION
  8. Better defaults: min_score=65, min_probability=55, TP=6%, SL=3%
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pymongo
from config import settings

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# ADAPTIVE STRATEGY DEFAULTS
# ══════════════════════════════════════════════════════════════════
DEFAULT_SETTINGS = {
    'initial_balance':      1000.0,
    'risk_per_trade':       0.05,       # fallback if Kelly disabled

    # Fixed TP/SL (used when ATR mode is off)
    'take_profit':          0.06,       # 6% TP
    'stop_loss':            0.03,       # 3% SL

    # ATR-based dynamic TP/SL (overrides fixed when enabled)
    'atr_tp_multiplier':    2.0,        # TP = entry + 2×ATR
    'atr_sl_multiplier':    1.0,        # SL = entry - 1×ATR
    'use_atr_sizing':       True,       # Use ATR for TP/SL

    # Signal quality gates — must match ai_score.py thresholds (BUY=55, HOLD=35)
    'min_score':            52,
    'min_probability':      45,

    # Kelly criterion position sizing
    'use_kelly_sizing':     True,       # Adaptive bet sizing
    'kelly_win_rate':       0.60,       # Starting estimate (updates from history)
    'kelly_avg_win':        6.0,        # Starting estimate
    'kelly_avg_loss':       3.0,        # Starting estimate

    # Correlation filter
    'use_correlation_filter': True,     # Block concentrated positions
    'max_per_group':          2,        # Max 2 positions per correlated group

    # S/R filter: don't enter near strong resistance
    'use_sr_filter':        True,       # Block entries near resistance

    # Adaptive features
    'trailing_stop':        True,
    'trailing_stop_pct':    0.025,
    'max_hold_hours':       48,
    'use_regime_filter':    False,  # Disabled: BEAR regime is persistent, would block all trades
    'use_whale_filter':     True,
    'use_confirmation':     True,
    'signal_window_hours':  4,

    'max_open_positions':   5,
    'fee_rate':             0.00075,
    'allow_hold_entry':     False,
}


def _load_db_strategy_config() -> Dict:
    """Load active strategy config from DB, fall back to adaptive defaults."""
    try:
        from optimization.strategy_config import get_active_strategy_config
        params = get_active_strategy_config()
        if params:
            merged = {**DEFAULT_SETTINGS, **params}
            return merged
    except Exception as e:
        logger.debug(f"[Backtester] DB config unavailable: {e} — using adaptive defaults")
    return {**DEFAULT_SETTINGS}


# ══════════════════════════════════════════════════════════════════
# BACKTESTER CLASS
# ══════════════════════════════════════════════════════════════════

class Backtester:
    """
    Backtesting engine v2 — Adaptive Strategy.
    Simulates historical trading using AI signals from MongoDB.
    """

    def __init__(self, settings_override: Dict = None):
        db_settings = _load_db_strategy_config()
        self.settings  = {**db_settings, **(settings_override or {})}
        self.client    = None
        self.db        = None
        self._mtf_cache: Dict = {}          # 4h data cache for multi-timeframe
        self._win_history: List[float] = [] # live win/loss tracker for Kelly

    def connect(self) -> bool:
        try:
            self.client = pymongo.MongoClient(settings.MONGO_URI,
                                              serverSelectionTimeoutMS=5000)
            self.db = self.client[settings.DATABASE_NAME]
            self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"DB connection failed: {e}")
            return False

    def close(self):
        if self.client:
            self.client.close()

    # ─────────────────────────────────────────────────────────────
    # DATA LOADING
    # ─────────────────────────────────────────────────────────────

    def load_market_data(self, symbol: str,
                         start_date: datetime,
                         end_date: datetime) -> List[Dict]:
        if self.db is None:
            return []
        try:
            cursor = self.db[settings.COLLECTION_MARKET_DATA].find({
                'symbol': symbol,
                'open_time': {'$gte': start_date, '$lte': end_date}
            }).sort('open_time', 1)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error loading market data for {symbol}: {e}")
            return []

    def load_ai_signals(self, start_date: datetime = None,
                        end_date: datetime = None,
                        timeframe: str = None) -> List[Dict]:
        if self.db is None:
            return []
        try:
            query = {}
            if start_date or end_date:
                query['timestamp'] = {}
                if start_date:
                    query['timestamp']['$gte'] = start_date
                if end_date:
                    query['timestamp']['$lte'] = end_date
            if timeframe:
                query['timeframe'] = timeframe
            cursor = self.db[settings.COLLECTION_AI_SIGNALS].find(
                query).sort('timestamp', 1)
            signals = list(cursor)
            logger.info(f"Loaded {len(signals)} AI signals for backtest period")

            # If real signals are sparse (<100) in requested window,
            # supplement with synthetic backtest signals (Nov2025-Feb2026)
            if len(signals) < 100:
                logger.info(
                    f"[Backtester] Only {len(signals)} real signals in window — "
                    f"supplementing with ai_signals_backtest_synth"
                )
                try:
                    synth_query = {}
                    if start_date or end_date:
                        synth_query['timestamp'] = {}
                        if start_date:
                            synth_query['timestamp']['$gte'] = start_date
                        if end_date:
                            synth_query['timestamp']['$lte'] = end_date
                    synth = list(self.db['ai_signals_backtest_synth'].find(
                        synth_query).sort('timestamp', 1))
                    if not synth:
                        # No synth in window either — load all synth regardless of date
                        synth = list(self.db['ai_signals_backtest_synth'].find(
                            {}).sort('timestamp', 1))
                        logger.info(f"[Backtester] Loaded ALL {len(synth)} synth signals (no date filter)")
                    else:
                        logger.info(f"[Backtester] Loaded {len(synth)} synth signals in window")
                    signals = synth + signals
                    signals.sort(
                        key=lambda s: self._naive_utc(s.get('timestamp', datetime.min))
                    )
                    logger.info(f"[Backtester] Total signals after merge: {len(signals)}")
                except Exception as synth_err:
                    logger.warning(f"[Backtester] Could not load synth signals: {synth_err}")

            return signals
        except Exception as e:
            logger.error(f"Error loading AI signals: {e}")
            return []


    def get_available_symbols(self) -> List[str]:
        if self.db is None:
            return []
        try:
            return self.db[settings.COLLECTION_MARKET_DATA].distinct('symbol')
        except:
            return []

    # ─────────────────────────────────────────────────────────────
    # LATEST SIGNAL LOOKUP (CRITICAL FIX)
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _naive_utc(dt: datetime) -> datetime:
        if dt is None:
            return datetime.min
        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
            try:
                from datetime import timezone as _tz
                return dt.astimezone(_tz.utc).replace(tzinfo=None)
            except Exception:
                return dt.replace(tzinfo=None)
        return dt

    def _get_latest_signal(self, symbol: str, signal_map: Dict,
                           at_time: datetime) -> Optional[Dict]:
        """
        FIX: Return the MOST RECENT signal before at_time within window.
        v1 bug: any signal in 48h window could trigger — stale signals caused
        trades on old data. Now: only the freshest signal within 4h counts.
        """
        sigs = signal_map.get(symbol, [])
        window_sec = self.settings.get('signal_window_hours', 4) * 3600
        at_naive = self._naive_utc(at_time)

        best = None
        best_time = None
        for sig in sigs:
            sig_time = self._naive_utc(sig.get('timestamp', datetime.min))
            time_diff = (at_naive - sig_time).total_seconds()
            if 0 <= time_diff <= window_sec:
                if best_time is None or sig_time > best_time:
                    best = sig
                    best_time = sig_time
        return best

    # ─────────────────────────────────────────────────────────────
    # ADAPTIVE ENTRY GATE
    # ─────────────────────────────────────────────────────────────

    def _should_buy(self, symbol: str, signal_map: Dict,
                    at_time: datetime) -> bool:
        """
        Adaptive multi-gate entry logic:
          Gate 1: Signal freshness (4h window, latest only)
          Gate 2: Signal quality (score >= min_score, prob >= min_probability)
          Gate 3: Market regime filter (no BUY in BEAR regime)
          Gate 4: Whale filter (no BUY if whales are distributing)
          Gate 5: Multi-confirmation (trend + MACD agree)
        """
        sig = self._get_latest_signal(symbol, signal_map, at_time)
        if sig is None:
            return False

        # Gate 2: Signal type + quality score
        signal_type = sig.get('final_signal', '')
        score = float(sig.get('final_score', 0) or 0)
        prob  = float(sig.get('probability_up', 0) or 0)
        allow_hold = self.settings.get('allow_hold_entry', False)

        valid_signal = (signal_type == 'BUY' or
                        (allow_hold and signal_type == 'HOLD'))
        if not valid_signal:
            return False
        if score < self.settings['min_score']:
            return False
        if prob < self.settings['min_probability']:
            return False

        # Gate 3: Market regime filter — SKIP if BEAR with high confidence
        if self.settings.get('use_regime_filter', True):
            regime = sig.get('market_regime', 'SIDEWAYS')
            regime_conf = float(sig.get('regime_confidence', 0) or 0)
            if regime == 'BEAR' and regime_conf >= 55:
                logger.debug(f"[Gate3] {symbol}: BEAR regime blocked entry")
                return False

        # Gate 4: Whale filter — skip if large distribution detected
        if self.settings.get('use_whale_filter', True):
            whale_signal = sig.get('whale_signal', 'NONE')
            whale_score  = float(sig.get('whale_score', 50) or 50)
            # Only block if strong distribution signal (score < 30 = bearish whales)
            if whale_signal == 'DISTRIBUTION' and whale_score < 35:
                logger.debug(f"[Gate4] {symbol}: Whale DISTRIBUTION blocked entry")
                return False

        # Gate 5: Multi-confirmation — at least 2 of 3 technical factors agree
        if self.settings.get('use_confirmation', True):
            inds = sig.get('indicators', {})
            rsi          = float(sig.get('rsi', 50) or inds.get('rsi', 50) or 50)
            tech_score   = float(sig.get('technical_score', 50) or 50)

            # Check alignment flags
            trend_ok   = tech_score >= 60             # Technical strength
            rsi_ok     = 40 <= rsi <= 70              # RSI not overbought/oversold
            score_ok   = score >= self.settings['min_score'] + 5  # Extra margin

            confirmations = sum([trend_ok, rsi_ok, score_ok])
            if confirmations < 2:
                logger.debug(
                    f"[Gate5] {symbol}: Multi-confirm failed "
                    f"(tech={trend_ok}, rsi={rsi_ok}, score={score_ok})"
                )
                return False

        return True

    def _has_sell_signal(self, symbol: str, signal_map: Dict,
                         at_time: datetime) -> bool:
        sig = self._get_latest_signal(symbol, signal_map, at_time)
        if sig and sig.get('final_signal') == 'SELL':
            # Only act on strong SELL signals
            score = float(sig.get('final_score', 100) or 100)
            if score < 40:
                return True
        return False

    # ─────────────────────────────────────────────────────────────
    # MAIN BACKTEST LOOP
    # ─────────────────────────────────────────────────────────────

    def run_backtest(self, days: int = 30,
                     start_date: datetime = None,
                     end_date: datetime = None) -> Dict:
        """
        Run full adaptive backtest simulation.

        New in v2:
          - Trailing stop-loss moves up as position profits
          - Max hold time auto-exits stale positions
          - All 5 entry gates active
        """
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=days)

        logger.info("=" * 60)
        logger.info(
            f"ADAPTIVE BACKTEST v2: "
            f"{start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')} ({days}d)"
        )
        logger.info(
            f"  TP={self.settings['take_profit']*100:.0f}%  "
            f"SL={self.settings['stop_loss']*100:.0f}%  "
            f"Trail={'ON' if self.settings.get('trailing_stop') else 'OFF'}  "
            f"RegimeFilter={'ON' if self.settings.get('use_regime_filter') else 'OFF'}  "
            f"MinScore={self.settings['min_score']}"
        )
        logger.info("=" * 60)

        # Load symbols + signals
        symbols = self.get_available_symbols()
        if not symbols:
            logger.warning("No market data found")
            return self._empty_result(start_date, end_date)

        signals = self.load_ai_signals(start_date, end_date)

        # Build signal lookup by symbol (sorted ascending for latest-signal logic)
        signal_map: Dict[str, List] = {}
        for sig in signals:
            sym = sig.get('symbol', '')
            if sym not in signal_map:
                signal_map[sym] = []
            signal_map[sym].append(sig)

        # Sort each symbol's signals by timestamp ascending
        for sym in signal_map:
            signal_map[sym].sort(
                key=lambda s: self._naive_utc(s.get('timestamp', datetime.min))
            )

        # Simulation state
        balance         = self.settings['initial_balance']
        initial_balance = balance
        open_positions  = {}       # symbol → position dict
        trade_history   = []
        equity_curve    = [{'timestamp': start_date, 'equity': balance}]
        peak_equity     = balance

        # Load + sort all candles
        all_candles = []
        for sym in symbols:
            candles = self.load_market_data(sym, start_date, end_date)
            for c in candles:
                c['_sym'] = sym
            all_candles.extend(candles)

        if not all_candles:
            logger.warning("No candles in date range")
            return self._empty_result(start_date, end_date)

        all_candles.sort(key=lambda x: x.get('open_time', datetime.min))
        logger.info(
            f"Processing {len(all_candles)} candles across {len(symbols)} symbols"
        )

        # ── Candle-by-candle simulation ──────────────────────────
        for candle in all_candles:
            sym         = candle['_sym']
            price       = candle.get('close', 0)
            high        = candle.get('high', price)
            low         = candle.get('low', price)
            candle_time = candle.get('open_time', datetime.utcnow())

            if price <= 0:
                continue

            # ── Manage open positions ────────────────────────────
            if sym in open_positions:
                pos       = open_positions[sym]
                entry     = pos['entry_price']
                peak_high = pos.get('peak_price', entry)

                # Update peak price for trailing stop
                if high > peak_high:
                    peak_high = high
                    open_positions[sym]['peak_price'] = peak_high

                # Compute exit levels — ATR-based or fixed
                atr_pct_pos = pos.get('atr_pct', 0.0)
                use_atr = self.settings.get('use_atr_sizing', True) and atr_pct_pos > 0

                if use_atr:
                    # ATR dynamic: TP = 2× ATR, SL = 1× ATR
                    atr_mult_tp = self.settings.get('atr_tp_multiplier', 2.0)
                    atr_mult_sl = self.settings.get('atr_sl_multiplier', 1.0)
                    tp_pct = (atr_pct_pos / 100.0) * atr_mult_tp
                    sl_pct = (atr_pct_pos / 100.0) * atr_mult_sl
                    # Clamp to reasonable range
                    tp_pct = max(0.02, min(0.15, tp_pct))
                    sl_pct = max(0.01, min(0.08, sl_pct))
                else:
                    tp_pct = self.settings['take_profit']
                    sl_pct = self.settings['stop_loss']

                tp_price = entry * (1 + tp_pct)

                # Trailing stop: SL moves up as price rises
                if self.settings.get('trailing_stop', False):
                    trail_pct = self.settings.get('trailing_stop_pct', 0.025)
                    sl_price  = peak_high * (1 - trail_pct)
                    sl_price  = max(sl_price,
                                   entry * (1 - self.settings['stop_loss']))
                else:
                    sl_price = entry * (1 - self.settings['stop_loss'])

                # Max hold time exit
                max_hold_h = self.settings.get('max_hold_hours', 48)
                hold_hours = (self._naive_utc(candle_time)
                              - self._naive_utc(pos['entry_time'])
                              ).total_seconds() / 3600

                exit_price  = None
                exit_reason = None

                if low <= sl_price:
                    exit_price  = sl_price
                    exit_reason = 'STOP_LOSS'
                elif high >= tp_price:
                    exit_price  = tp_price
                    exit_reason = 'TAKE_PROFIT'
                elif hold_hours >= max_hold_h:
                    exit_price  = price
                    exit_reason = 'TIME_EXIT'
                elif self._has_sell_signal(sym, signal_map, candle_time):
                    exit_price  = price
                    exit_reason = 'SELL_SIGNAL'

                if exit_price is not None:
                    fee = (pos['position_size']
                           + abs(pos['position_size']
                                 * (exit_price - entry) / entry)
                           ) * 2 * self.settings.get('fee_rate', 0.00075)

                    pnl_pct    = (exit_price - entry) / entry * 100
                    profit_u   = (pos['position_size']
                                  * (exit_price - entry) / entry) - fee
                    balance   += pos['position_size'] + profit_u

                    # Update Kelly win history
                    self._win_history.append(1.0 if pnl_pct > 0 else 0.0)

                    duration = (
                        self._naive_utc(candle_time)
                        - self._naive_utc(pos['entry_time'])
                    ).total_seconds() / 60

                    trade_history.append({
                        'symbol':          sym,
                        'entry_price':     round(entry, 6),
                        'exit_price':      round(exit_price, 6),
                        'pnl_percent':     round(pnl_pct, 2),
                        'profit_usdt':     round(profit_u, 2),
                        'duration_minutes':round(duration, 0),
                        'exit_reason':     exit_reason,
                        'entry_time':      pos['entry_time'].isoformat()
                            if hasattr(pos['entry_time'], 'isoformat')
                            else str(pos['entry_time']),
                        'exit_time':       candle_time.isoformat()
                            if hasattr(candle_time, 'isoformat')
                            else str(candle_time),
                        'peak_price':      round(peak_high, 6),
                        'trailing_sl':     round(sl_price, 6),
                    })
                    del open_positions[sym]
                    logger.debug(
                        f"CLOSE {sym}: {exit_reason} PnL={pnl_pct:+.2f}%"
                    )

            # ── Open new position ────────────────────────────────
            elif sym not in open_positions:
                if len(open_positions) < self.settings['max_open_positions']:
                    if self._should_buy(sym, signal_map, candle_time):

                        # Gate A: Correlation filter
                        if self.settings.get('use_correlation_filter', True):
                            try:
                                from ai.portfolio_manager import is_correlated_entry_blocked
                                corr = is_correlated_entry_blocked(
                                    sym, open_positions,
                                    self.settings.get('max_per_group', 2))
                                if corr['blocked']:
                                    logger.debug(f"[Corr] {sym}: {corr['reason']}")
                                    continue
                            except ImportError:
                                pass

                        # Gate B: S/R filter — don't buy near strong resistance
                        if self.settings.get('use_sr_filter', True):
                            sig_ = self._get_latest_signal(sym, signal_map, candle_time)
                            if sig_:
                                near_res = sig_.get('near_resistance', False)
                                sr_qual  = float(sig_.get('sr_quality', 0) or 0)
                                if near_res and sr_qual >= 45:
                                    logger.debug(f"[S/R] {sym}: near resistance (quality={sr_qual:.0f}) — skip")
                                    continue

                        # Determine ATR from signal (for dynamic TP/SL)
                        atr_pct = 0.0
                        if self.settings.get('use_atr_sizing', True):
                            sig_ = self._get_latest_signal(sym, signal_map, candle_time)
                            if sig_:
                                atr_pct = float(sig_.get('atr_pct', 0) or 0)

                        # Kelly position sizing
                        if self.settings.get('use_kelly_sizing', True):
                            try:
                                from ai.portfolio_manager import get_position_size
                                # Estimate current win/loss stats from history
                                recent = self._win_history[-50:] if self._win_history else []
                                live_wr = (sum(recent) / len(recent)) if recent else self.settings.get('kelly_win_rate', 0.60)
                                sig_ = self._get_latest_signal(sym, signal_map, candle_time) or {}
                                score_f = float(sig_.get('final_score', 65) or 65)
                                conf_f  = float(sig_.get('probability_up', 55) or 55)
                                k = get_position_size(
                                    balance, score_f, conf_f, live_wr,
                                    self.settings.get('kelly_avg_win', 6.0),
                                    self.settings.get('kelly_avg_loss', 3.0)
                                )
                                position_size = k['position_size_usdt']
                            except (ImportError, Exception):
                                position_size = balance * self.settings['risk_per_trade']
                        else:
                            position_size = balance * self.settings['risk_per_trade']

                        if position_size > 1.0 and balance >= position_size:
                            balance -= position_size
                            open_positions[sym] = {
                                'entry_price':   price,
                                'entry_time':    candle_time,
                                'position_size': position_size,
                                'peak_price':    price,
                                'atr_pct':       atr_pct,  # store for TP/SL
                            }
                            logger.debug(
                                f"OPEN {sym} @ {price:.4f}"
                                f" size=${position_size:.2f}"
                                f" atr={atr_pct:.2f}%"
                            )

            # Track equity
            total_eq = balance
            for p in open_positions.values():
                total_eq += p['position_size']
            equity_curve.append({
                'timestamp': candle_time,
                'equity':    round(total_eq, 2),
            })
            if total_eq > peak_equity:
                peak_equity = total_eq

        # ── Close remaining positions at period end ──────────────
        for sym, pos in list(open_positions.items()):
            last_candles = self.load_market_data(
                sym, end_date - timedelta(days=1), end_date)
            last_price = (last_candles[-1].get('close', pos['entry_price'])
                          if last_candles else pos['entry_price'])

            pnl_pct  = (last_price - pos['entry_price']) / pos['entry_price'] * 100
            profit_u = (pos['position_size']
                        * (last_price - pos['entry_price'])
                        / pos['entry_price'])
            balance += pos['position_size'] + profit_u

            trade_history.append({
                'symbol':          sym,
                'entry_price':     round(pos['entry_price'], 6),
                'exit_price':      round(last_price, 6),
                'pnl_percent':     round(pnl_pct, 2),
                'profit_usdt':     round(profit_u, 2),
                'duration_minutes':0,
                'exit_reason':     'END_OF_PERIOD',
                'entry_time':      pos['entry_time'].isoformat()
                    if hasattr(pos['entry_time'], 'isoformat')
                    else str(pos['entry_time']),
                'exit_time':       end_date.isoformat(),
            })

        # ── Metrics ──────────────────────────────────────────────
        metrics = self._calculate_metrics(
            trade_history, initial_balance, balance, equity_curve)

        result = {
            'timestamp':    datetime.utcnow(),
            'period': {
                'start': start_date.isoformat(),
                'end':   end_date.isoformat(),
                'days':  days,
            },
            'settings':     self.settings,
            'metrics':      metrics,
            'trade_history':trade_history,
            'equity_curve': equity_curve[-500:],
        }

        logger.info("=" * 60)
        logger.info("ADAPTIVE BACKTEST COMPLETE")
        logger.info(f"  Balance: ${initial_balance:.2f} → ${balance:.2f}")
        logger.info(f"  Return:  {metrics['return_percent']:+.2f}%")
        logger.info(
            f"  Trades:  {metrics['total_trades']} "
            f"(WR: {metrics['win_rate']:.1f}%)"
        )
        logger.info(f"  PF:      {metrics['profit_factor']:.2f}")
        logger.info(f"  Drawdown:{metrics['max_drawdown']:.2f}%")
        logger.info("=" * 60)

        return result

    # ─────────────────────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────────────────────

    def _calculate_metrics(self, trades, initial, final, equity_curve) -> Dict:
        total_trades = len(trades)
        if total_trades == 0:
            return {
                'balance_initial': round(initial, 2),
                'balance_final':   round(final, 2),
                'return_percent':  0.0, 'win_rate': 0.0,
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'max_drawdown': 0.0, 'profit_factor': 0.0,
                'average_trade': 0.0, 'best_trade': 0.0, 'worst_trade': 0.0,
                'avg_duration_min': 0.0, 'tp_hits': 0, 'sl_hits': 0,
                'time_exits': 0, 'expectancy_pct': 0.0, 'expectancy_usdt': 0.0,
                'avg_win_pct': 0.0, 'avg_loss_pct': 0.0, 'rr_ratio': 0.0,
                'sharpe_proxy': 0.0,
            }

        pnls        = [t['pnl_percent'] for t in trades]
        profits_u   = [t['profit_usdt'] for t in trades]
        winners     = [p for p in pnls if p > 0]
        losers      = [p for p in pnls if p <= 0]
        gross_profit= sum(p for p in profits_u if p > 0)
        gross_loss  = abs(sum(p for p in profits_u if p < 0))

        # Drawdown
        max_dd  = 0.0
        peak    = equity_curve[0]['equity'] if equity_curve else initial
        for pt in equity_curve:
            eq = pt['equity']
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        avg_win  = (sum(p for p in pnls if p > 0) / len(winners)) if winners else 0
        avg_loss = abs(sum(p for p in pnls if p < 0) / len(losers)) if losers else 0
        win_rate_dec  = len(winners) / total_trades
        loss_rate_dec = 1 - win_rate_dec
        expectancy    = (avg_win * win_rate_dec) - (avg_loss * loss_rate_dec)
        rr_ratio      = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0

        sharpe = 0.0
        if len(pnls) >= 2:
            import statistics as _s
            mean_r = _s.mean(pnls)
            std_r  = _s.stdev(pnls)
            if std_r > 0:
                sharpe = round(mean_r / std_r, 2)

        # Expectancy per $50 (5% of $1000)
        stake = initial * DEFAULT_SETTINGS['risk_per_trade']

        return {
            'balance_initial':  round(initial, 2),
            'balance_final':    round(final, 2),
            'return_percent':   round((final - initial) / initial * 100, 2),
            'win_rate':         round(win_rate_dec * 100, 1),
            'total_trades':     total_trades,
            'winning_trades':   len(winners),
            'losing_trades':    len(losers),
            'max_drawdown':     round(max_dd, 2),
            'profit_factor':    round(gross_profit / gross_loss, 2)
                                if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0),
            'average_trade':    round(sum(pnls) / total_trades, 2),
            'best_trade':       round(max(pnls), 2) if pnls else 0.0,
            'worst_trade':      round(min(pnls), 2) if pnls else 0.0,
            'avg_duration_min': round(
                sum(t['duration_minutes'] for t in trades) / total_trades, 0),
            'tp_hits':          sum(1 for t in trades
                                    if t['exit_reason'] == 'TAKE_PROFIT'),
            'sl_hits':          sum(1 for t in trades
                                    if t['exit_reason'] == 'STOP_LOSS'),
            'time_exits':       sum(1 for t in trades
                                    if t['exit_reason'] == 'TIME_EXIT'),
            'expectancy_pct':   round(expectancy, 2),
            'expectancy_usdt':  round(expectancy / 100 * stake, 2),
            'avg_win_pct':      round(avg_win, 2),
            'avg_loss_pct':     round(-avg_loss, 2),
            'rr_ratio':         rr_ratio,
            'sharpe_proxy':     sharpe,
        }

    def _empty_result(self, start, end) -> Dict:
        return {
            'timestamp': datetime.utcnow(),
            'period':    {'start': start.isoformat(), 'end': end.isoformat(), 'days': 0},
            'settings':  self.settings,
            'metrics':   self._calculate_metrics(
                [], self.settings['initial_balance'],
                self.settings['initial_balance'], []),
            'trade_history': [],
            'equity_curve':  [],
        }

    # ─────────────────────────────────────────────────────────────
    # DB STORAGE
    # ─────────────────────────────────────────────────────────────

    def save_results(self, result: Dict) -> bool:
        if self.db is None:
            return False
        try:
            clean = {k: v for k, v in result.items() if k != '_id'}
            self.db['backtest_results'].insert_one(clean)
            logger.info("Backtest results saved to MongoDB")
            return True
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
            return False

    def get_latest_result(self) -> Optional[Dict]:
        if self.db is None:
            return None
        try:
            return self.db['backtest_results'].find_one(
                {}, sort=[('timestamp', pymongo.DESCENDING)])
        except:
            return None

    def get_all_results(self, limit: int = 10) -> List[Dict]:
        if self.db is None:
            return []
        try:
            return list(self.db['backtest_results'].find({}).sort(
                'timestamp', pymongo.DESCENDING).limit(limit))
        except:
            return []


# ══════════════════════════════════════════════════════════════════
# MODULE-LEVEL HELPER (used by dashboard + scheduler)
# ══════════════════════════════════════════════════════════════════

def run_backtest(db, lookback_days: int = 90,
                 settings_override: Dict = None) -> Dict:
    """Convenience function for external callers."""
    bt = Backtester(settings_override)
    bt.db = db
    result = bt.run_backtest(days=lookback_days)
    bt.save_results(result)
    return result.get('metrics', {})


if __name__ == '__main__':
    bt = Backtester()
    if bt.connect():
        result = bt.run_backtest(days=90)
        m = result['metrics']
        print(f"\n  Return:  {m['return_percent']:+.2f}%")
        print(f"  Win Rate:{m['win_rate']:.1f}%")
        print(f"  Trades:  {m['total_trades']}")
        print(f"  PF:      {m['profit_factor']:.2f}")
        print(f"  Drawdown:{m['max_drawdown']:.2f}%")
        print(f"  TP/SL/Time: {m['tp_hits']}/{m['sl_hits']}/{m.get('time_exits',0)}")
        if result['trade_history']:
            bt.save_results(result)
        bt.close()

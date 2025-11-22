# Naporrr
оціни промт на створення професійного бектесту для торгового бота : Нижче — фіналізована концепція та детальний дизайн модуля бектестингу (ФАЗА 1: REST API) саме під вашого бота, з урахуванням усіх вимог: виключення розміру позиції/плеча/ліміту одночасних угод з оптимізації, анти-repainting, максимальна точність, адаптивність під різні пари, без хардкоду, і селективні перевірки через WebSocket у межах 10 ГБ.

Мета: описати функції (лише текст) і взаємодію між компонентами так, щоб реалізація utils/backtest.py була прямолінійною та відповідала архітектурі вашого бота.

1) Загальна ідея та межі ФАЗИ 1
- Джерело даних: REST API Bybit (OHLCV 1m + допоміжні сервіси). Немає повного запису WebSocket, щоб економити місце. 
- Точність: максимально можлива при REST-джерелі завдяки:
  - Генерації синтетичного orderbook та trades з OHLCV (адаптивно під символ, волатильність, обсяг).
  - Повторному використанню реальних модулів аналізу вашого бота: ImbalanceAnalyzer, VolumeAnalyzer, SignalGenerator, RiskManager, щоб логіка сигналів і TP/SL була ідентичною live-боту.
  - Anti-repainting кроковому симулюванню: сигнал формується на початку свічки, SL/TP/час утримання — із минулої волатильності, виконання — по інтрабарному шляху.
- Виходи: журнали і звіти у logs/backtest/ (CSV трейди, крива капіталу, JSON метрики, журнали).
- Оптимізація параметрів: тільки ті, що впливають на якість сигналу/закриття (приклади: пороги composite, ваги факторів, мінімальні SL/TP тощо). Виключаємо base_order_pct, leverage, max_open_positions.
- Обмежений WS-контроль: короткі калібрувальні сесії (селективні вибірки) для 2-3 пар у вузькі вікна часу (напр. 10–30 хв) для верифікації синтетичної генерації (≤10 ГБ сумарно, ротація).

2) Структура модуля та ролі функцій (лише текст)
Умовна структура utils/backtest.py (класи та ключові методи з описами):

BacktestEngine
- Опис: Головний оркестратор бектесту. Керує сценаріями quick/full/grid, розбиває періоди, координує завантаження даних, симуляцію, підрахунок метрик, збереження результатів і (за потреби) оптимізацію параметрів.
- Основні методи:
  - configure_from_settings(): читає settings.backtest і фіксує конфіг (символи, період, джерела, режими).
  - run_quick_test(): швидкий прогін на обраному періоді й підмножині пар для sanity-check.
  - run_full_test(): повний прогін на всіх обраних символах і періоді; агрегує результати по символах і загалом.
  - run_parameter_sweep(param_grid): перебір комбінацій оптимізованих параметрів з обмеженням на ітерації; зберігає результати кожної комбінації; обирає найкращі.
  - run_walk_forward(): опційно виконує walk-forward розбиття (in-sample/validation) без repainting.
  - _test_on_symbol(symbol, time_ranges, params): запускає повноцінну симуляцію на одному символі по одному або кількох підперіодах.
  - _save_run_artifacts(run_id, outputs): записує CSV/JSON/лог-файли; ротує за політикою з settings.backtest.

BacktestConfig
- Опис: Не-Pydantic об’єкт (внутрішній), що тримає зліпок конфігурації на момент запуску бектесту (щоб всі допоміжні сервіси мали стабільні параметри під час прогону).
- Поля: список символів, період, таймфрейм, режим генерації синтетичних даних, моделі slippage/latency, список параметрів для оптимізації, політики зберігання, таргет-метрики тощо.

HybridDataProvider (режим REST)
- Опис: Відповідає за завантаження OHLCV із REST (та кешування на диск/у пам’ять), підготовку даних до синтетичної генерації.
- Основні методи:
  - load_ohlcv(symbol, start, end, timeframe): батчове завантаження 1m свічок з REST (врахування rate limit), валідація цілісності (без пропусків, відсортовано, нормалізовані timestamps).
  - get_symbol_stats(ohlcv): обчислює базові характеристики символу для подальшої адаптації (середня волатильність, середній обсяг, приблизний спред у bps, частка трендових/флетових відрізків).
  - cache_policy(): стратегія кешування свічок на локальний диск (компресія), щоб мінімізувати повторні запити.

SyntheticMarketBuilder
- Опис: Генерує реалістичний синтетичний порядок подій всередині хвилини на базі OHLCV; від нього залежить робота ваших аналізаторів.
- Основні методи:
  - build_intrabar_path(candle, symbol_profile): генерує порядок подій O→H→L→C або O→L→H→C із невеликою стохастикою; повертає часові точки і ціни (інтрапойнти) у межах хвилини.
  - synthesize_orderbook(candle, symbol_profile): будує на кожному інтрапойнті книжку з 50 рівнями (агреговано), адаптивно до волатильності/обсягу (розподіл ліквідності за рівнями, спред залежно від поточної волатильності).
  - synthesize_trades(candle, symbol_profile): генерує набір угод з розумним розподілом розмірів (Pareto), напрямку (залежно від руху ціни), і часткою великих угод (для tape patterns).
  - calibrate_from_ws_stats(ws_snapshot_stats): приймає агреговані метрики з Версифікатора WS (нижче) і підкручує параметри генератора (спред-модель, коефіцієнти розподілу обсягів, частку великих трейдів) для зменшення розбіжностей.

SyntheticFeedRunner
- Опис: Подає синтетичні orderbook і trades у DataStorage так, як це робить ваш Collector, щоб ImbalanceAnalyzer/VolumeAnalyzer/SignalGenerator працювали на "реалістичній" стрічці.
- Основні методи:
  - reset_storage_for_symbol(symbol): готує DataStorage для символу (retention_seconds, max_depth тощо) відповідно до settings.websocket/data_retention.
  - feed_intrabar(symbol, intrabar_point): на кожному інтрапойнті викликає storage.update_order_book(...) і storage.add_trade(...) у потрібній кількості, з часовими мітками відповідно до інтрапойнту.
  - finalize_candle(symbol): закінчує хвилину, забезпечуючи стабільний стан для наступної.

NoRepaintSimulator
- Опис: Ядро анти-repainting. Точно задає, які дані "доступні" у кожній точці часу.
- Основні методи:
  - step_begin_candle(symbol, candle_index): на початку хвилини створює контекст тільки з минулих даних; забороняє доступ до high/low/close поточної свічки для сигналу.
  - compute_signal(symbol): викликає VolumeAnalyzer.compute(...) та ImbalanceAnalyzer.compute(...) на синтетичних даних, потім SignalGenerator.generate(...) з spread_bps=з поточного best_bid/ask; повертає сигнал HOLD/BUY/SELL та strength; готує factors для валідації.
  - derive_sl_tp_and_lifetime(signal, entry_price, past_vol_data): використовує RiskManager.calc_sl_tp(...) та RiskManager.get_adaptive_lifetime_seconds(...), передаючи volatility_data з останнього "минулого" розрахунку VolumeAnalyzer (тобто за даними до початку поточної хвилини).
  - simulate_execution(symbol, order_intent): моделює виконання лімітного входу: якщо intrabar-прайс торкається limit — філ; якщо ні і минув fallback_after_sec — філ маркетом із адаптивним slippage. Параметри: reprice_step_bps, passive_improve_bps, fallback_after_sec — читаються з settings.execution, але бектест оперує лише ймовірнісною/правдоподібною імітацією цих ефектів, без реальних API-дзвінків.
  - simulate_position_lifecycle(symbol, position, intrabar_stream): перевіряє в кожному інтрапойнті умови SL/TP/REVERSE/TIME_EXIT, застосовує CloseReasonDetector для узгодженості причин як у live.
  - finish_candle(symbol): якщо позиція відкрита й не закрита — продовжує на наступну хвилину; якщо досягнуто TIME_EXIT — закриває з відповідною причиною.

ExecutionModel
- Опис: Реалістична модель виконання без WebSocket:
  - Entry: сигнал на open свічки; ліміт виставляється близько до best_bid/best_ask (пасивне поліпшення з settings.execution.passive_improve_bps), шанс філу залежить від діапазону (H–L), частки часу досягнення ціни, обсягу.
  - Fallback: якщо не виконано до fallback_after_sec — маркет-філ на найближчому інтрапойнті з адаптивним slippage (залежно від волатильності, синтетичної глибини).
  - Exit: SL/TP перевіряються по інтрапойнтах і виконуються з невеликою толерантністю (slippage-толеранс від волатильності). REVERSE/opp_signal — за фактом нового сигналу у майбутніх хвилинах із дотриманням min_position_hold_time_sec.
- Основні методи:
  - prob_fill_limit(order, intrabar_slice): ймовірнісна оцінка філу ліміт-ордера на базі доступних інтрапойнтів і синтетичної глибини.
  - apply_slippage(side, price, vol_profile): повертає скориговану ціну з урахуванням волатильності, "спреду" та квазі-імпакту.

PnLCalculator
- Опис: Обчислює PnL у масштабі, незалежному від розміру позиції (універсально для порівняння параметрів). В опціях — перерахунок у %, у базовій валюті або в “на 1 контракт”.
- Основні методи:
  - pnl_on_close(position, exit_price): повертає PnL в абсолюті та у % від entry_price.
  - aggregate_equity_curve(trade_sequence): формує криву капіталу при фіксованому базовому номіналі (щоб порівнюваність була збережена і не залежала від base_order_pct/leverage).

MetricsAnalyzer
- Опис: Формує всі метрики якості, ризику й операційні; підсумки по символах і загалом.
- Основні методи:
  - compute_basic_metrics(trades): total_trades, win_rate, avg_win/avg_loss, total_pnl_pct, profit_factor, largest_win/loss.
  - compute_risk_metrics(equity_curve): max_drawdown, sharpe/sortino/calmar, recovery_factor, risk_of_ruin (оцінка).
  - compute_operational_metrics(trades): середній час утримання, частота трейдів, рахунки TP_HIT/SL_HIT/TIME_EXIT/REVERSE, серії win/loss.
  - symbol_comparison(results): ранжування символів, виявлення outliers.
  - objective_score(metrics, targets): агрегований скор для оптимізації за пріоритетами settings.backtest.target_metrics.

ReportWriter
- Опис: Записує результати у logs/backtest: CSV/JSON та компактний текстовий звіт, без важкої графіки (файли невеликого розміру).
- Основні методи:
  - write_trades_csv(run_id, symbol, trades): компактний CSV із основними полями (timestamp, side, entry/exit, sl/tp, reason, pnl%).
  - write_equity_curve(run_id, symbol, curve_points): CSV/JSON для кривої капіталу.
  - write_metrics_json(run_id, symbol, metrics): усі метрики в JSON.
  - write_summary(run_id, all_symbols_metrics): агрегований підсумок по символах; рекомендації.
  - rotate_old_runs(policy): очищення за clear_data_storage_days/keep_last_n_runs.

ParameterOptimizer (опційно в межах Phase 1)
- Опис: Невеликий модуль перебору параметрів (grid або random), строго виключаючи параметри розміру позиції/плеча/ліміту угод.
- Основні методи:
  - build_param_grid(optimizable_params): формує сітку або випадкові вибірки.
  - evaluate_combo(params): викликає BacktestEngine._test_on_symbol для набору символів; повертає метрики та objective_score.
  - select_best(results): обирає найкращі комбінації і зберігає їх у logs/backtest/optimization.

WSSelectiveVerifier (калібратор у 10 ГБ)
- Опис: Короткочасні записи через WebSocket для 2–3 символів, 10–30 хв за сесію, лише агреговані статистики для калібрування синтетики (без повного бектесту по WS).
- Основні методи:
  - record_sample(symbols, duration_min): підписка на public orderbook/trades; збір агрегатів: середній спред, розподіл розмірів трейдів, частка великих угод, глибина top-5 рівнів, імбеланс-волатильність.
  - summarize_and_store(run_id): запис компактного calibration.json (десятки-ста КБ, не ГБ).
  - compare_with_synthetic(symbol_profile, ws_summary): розрахунок відхилень; рекомендує корекції параметрів SyntheticMarketBuilder.
  - enforce_disk_budget(max_gb=10): видаляє найстаріші сесії та зберігає лише агрегати.

3) Анти-repainting: контракт поведінки по крокам
- Початок хвилини (t = open):
  - Доступні тільки історичні дані до цієї хвилини.
  - Формуємо синтетичний intrabar-потік для поточної хвилини, але сигнал рахуємо ДО подачі цих інтраподій у аналізатори.
  - VolumeAnalyzer/ImbalanceAnalyzer працюють на стані storage, який містить лише минуле.
  - SignalGenerator.generate повертає HOLD/BUY/SELL і strength.
- Entry:
  - Розрахунок SL/TP та max_lifetime на базі волатильності з минулих хвилин (range/ATR/std з VolumeAnalyzer).
  - Вхід спершу як ліміт (пасивне поліпшення ціни), далі — якщо intrabar не торкнувся — fallback до market за execution.fallback_after_sec з адаптивним slippage.
- Інтрабар (всередині хвилини):
  - Подаємо у DataStorage синтетичний orderbook/trades відповідно до intrabar_path.
  - На кожному інтрапойнті перевіряємо: чи торкнулись SL/TP з толерансом, чи настав REVERSE (новий сигнал в майбутніх хвилинах з дотриманням min_position_hold_time_sec), чи час вийти за TIME_EXIT.
- Кінець хвилини:
  - Якщо позиція відкрита — переносимо на наступну хвилину; сигнал наступної хвилини знову рахуємо тільки на минулих даних, без доступу до її high/low/close.
- Ретельні перевірки:
  - Entry — тільки по open (для відтворюваності).
  - SL/TP — в межах досяжності інтрадіапазоном (не допускаємо неможливих цін).

4) Що саме оптимізуємо (і що ні)
- Не оптимізуємо: trading.base_order_pct, trading.leverage, risk.max_open_positions, trading.base_order_usdt — вони під вашим ручним контролем.
- Оптимізуємо (приклади, без хардкоду — задається у settings.backtest): 
  - signals: weight_imbalance, weight_momentum, composite_thresholds.strength_{2..5}, smoothing_alpha, hold_threshold, max_imbalance_contradiction, volatility_filter_threshold.
  - risk: min_sl_pct, min_tp_pct, tpsl_ratio_{high,medium,low}_winrate, base_position_lifetime_minutes, low/high_volatility_lifetime_multiplier.
  - imbalance: smoothing_factor, historical_window_minutes, historical_samples, long_term_smoothing.
  - volume: momentum_windows, momentum_weights, short_window_sec (адаптивний).
  - spread: max_spread_threshold_bps.
- Механіка: параметри вносяться в тимчасову сесію Settings для прогону, щоб ImbalanceAnalyzer/VolumeAnalyzer/SignalGenerator/RiskManager бачили їх як “продакшн” значення.

5) Метрики і цілі (для ранжування параметрів)
- Базові: total_trades, win_rate, avg_win/avg_loss, profit_factor, total_pnl_pct, largest_win/loss.
- Ризик: max_drawdown, sharpe, sortino, calmar, recovery_factor.
- Операційні: середня тривалість, trades_per_day, TP_HIT/SL_HIT/TIME_EXIT/REVERSE частки, серії win/loss, середня відстань SL/TP від entry у bps.
- Цілі (з settings.backtest.target_metrics): мін-значення/макс-ліміти; агрегований score для порівняння комбінацій.

6) Робота з кількома парами (адаптивність, без хардкоду)
- Для кожного символу HybridDataProvider рахує профіль символу (волатильність, типовий спред, середні обсяги) і передає SyntheticMarketBuilder.
- Параметри генерації синтетики (спред, розподіл глибини, частка великих угод, інтенсивність трейдів) масштабуються під символ-профіль.
- Параметри сигналів та ризику — спільні (unified) або індивідуально підібрані: режим визначається BacktestSettings (optimize_per_symbol, optimize_unified). Для unified шукаємо набір, що працює прийнятно для всіх пар — підхід robust/median.
- Жодних жорстко прошитих символів: усі діапазони/коефіцієнти з settings.backtest та (за потреби) з профілю символу.

7) Запис результатів і управління диском
- Структура: logs/backtest/runs/{run_id}/ з файлами:
  - trades_{symbol}.csv (компактно)
  - equity_{symbol}.csv або .json
  - metrics_{symbol}.json
  - summary.json (агрегація по всіх)
  - calibration.json (якщо була WS-сесія)
  - backtest.log (тільки ключові події)
- Політика зберігання:
  - Авто-очищення за clear_data_storage_days або keep_last_n_runs.
  - Калібраційні WS-сесії зберігають ТІЛЬКИ агрегати; raw-сировину видаляємо; перевищення ліміту 10 ГБ запускає ротацію найстаріших записів.

8) Налаштування BacktestSettings (у settings.py) — опис полів
- enable_backtest: вмикає/вимикає бекграунд-цикли бектесту/оптимізації.
- backtest_cycle_hrs: періодичність автозапуску (години).
- backtest_start_time: бажаний час старту в добі (UTC), напр. “03:00”.
- data_source: “rest_api” (для ФАЗИ 1).
- data_lookback_days: глибина історії у днях (наприклад 30–90).
- data_timeframe: “1m”.
- optimization_mode: “grid”/“random” (у ФАЗІ 1 доцільні ці, без важких баєс).
- optimization_iterations: максимум комбінацій/ітерацій.
- optimize_per_symbol: оптимізувати параметри окремо на кожен символ (True/False).
- optimize_unified: шукати параметри, однаково добрі для всіх пар (True/False).
- auto_apply_params: автоматично застосовувати знайдені параметри до settings (з вимогою порога покращення).
- auto_apply_threshold_improvement: мінімальне покращення метрики (наприклад +10% Sharpe або +X% total_pnl_pct) для автозастосування.
- require_manual_approval: вимагати ручне підтвердження застосування (безпека).
- validation_split: частка даних out-of-sample (наприклад 0.3).
- walk_forward_enabled: вмикає walk-forward (True/False).
- walk_forward_window_days/step_days: параметри вікон та кроку.
- strict_repainting_check: строгі анти-repainting інваріанти (True).
- intrabar_simulation: увімкнути інтрабарну симуляцію (True).
- use_realistic_slippage: враховувати slippage (True).
- slippage_model: “adaptive” (або “fixed”, “volume_based”).
- target_metrics: цільові пороги якості (мінімальні для Sharpe, win_rate, тощо).
- generate_html_report/generate_plots: у ФАЗІ 1 — False/мінімально, щоб економити місце.
- save_equity_curve/save_trade_log: True (корисні для аналізу).
- report_path: каталог звітів.
- notify_on_completion/notify_on_better_params: нотифікації (через notifier).
- notify_threshold_improvement: поріг, коли варто сповіщати.
- auto_cleanup/keep_last_n_runs/cleanup_interval_days: політики очищення.
- max_memory_gb/checkpoint_every_n_iterations/enable_progress_bar: контроль ресурсів.
- debug_mode/save_intermediate_results/log_level_backtest: діагностика.
- calibration_ws_enabled: чи дозволено короткі WS-калібрації.
- calibration_ws_duration_min: тривалість кожної сесії (наприклад 15 хв).
- calibration_ws_symbols: список символів для калібрування (за замовчуванням 2–3).
- disk_budget_gb: загальний бюджет диска під калібрацію (10 ГБ).
- auto_correct_param: якщо True — дозволяє автокорекцію внутрішніх коефіцієнтів SyntheticMarketBuilder на базі ws-агрегатів (без зміни торгових налаштувань).

9) Потік виконання одного повного тесту (узгоджена послідовність)
- Для кожного символу:
  1) Завантажити OHLCV 1m за період (REST) з перевіркою цілісності.
  2) Обчислити профіль символу (волатильність, типовий спред, обсяг).
  3) Ініціалізувати DataStorage/аналізатори.
  4) Для кожної хвилини t:
     - NoRepaintSimulator.step_begin_candle(t) — сигнал тільки на минулих даних.
     - Якщо сигнал BUY/SELL зі strength ≥ entry_signal_min_strength:
       - derive_sl_tp_and_lifetime із минулої волатильності; сформувати намір ордера (ліміт з поліпшенням).
     - Запустити SyntheticFeedRunner на інтрапойнтах хвилини, симулюючи orderbook/trades.
     - ExecutionModel: спробувати філ ліміт-ордера; якщо ні — fallback до market за налаштуваннями.
     - Якщо позиція відкрита — на кожному інтрапойнті перевіряти SL/TP/REVERSE/TIME_EXIT; використати CloseReasonDetector, щоб привести причину до тих самих правил, що й у live.
  5) По завершенню періоду — розрахунок метрик, збереження результатів.
- Якщо optimization_mode ввімкнено — повторити з різними наборами параметрів; обрати найкращі.
- Якщо calibration_ws_enabled — запустити коротку WS-сесію (мінімальні агрегати), оновити параметри синтетики і повторити короткий тест для перевірки корекції.

10) Реалістичність і обмеження
- Комісії: враховуються фіксовано (наприклад 0.055% taker, 0.02–0.04% maker) у моделі виконання; задається в BacktestSettings.
- Slippage: адаптивний — вищий при високій волатильності та низькій синтетичній глибині.
- Заповнення ліміту: залежить від intrabar-досяжності ціни та синтетичної глибини; є параметри для калібрування (через WSSelectiveVerifier).
- Intrabar-порядок: використовується обережний O→H→L→C чи O→L→H→C з невеликим шумом; гарантує фізичну досяжність high/low та запобігає “неможливим” exіt-цінам.
- Важливо: REST-дані не дають справжній тік-потік; тому WSSelectiveVerifier використовується як регулярний "калібратор" параметрів синтетики у межах 10 ГБ.

11) Висновок і відповідність вашим вимогам
- Параметри позиції, плече, ліміт відкритих угод — виключені з оптимізації; результати PnL нормуються, щоб бути не залежними від розміру позиції.
- Підхід є професійним та адаптованим саме під вашу архітектуру: ми повторно використовуємо реальні модулі вашого бота й формуємо їм вхід так, ніби це WS-стрічка, але з REST-даних.
- Даних REST достатньо для старту; для підвищення точності передбачено легкі WS-вибірки (калібрація) під жорсткий ліміт 10 ГБ.
- Repainting усунутий: сигнал на open, SL/TP і lifetime з минулої волатильності; інтрабарний рух моделюється; перевірки неможливих виходів.
- Налаштування для управління бектестом описані (enable_backtest, backtest_cycle_hrs, auto_correct_param, clear_data_storage_days, calibration_ws_*), що дає повну автоматизацію й контроль.
- Адаптивність без хардкоду: профілі символів, параметри генерації синтетики, спільні або індивідуальні параметри стратегії — все керується конфігом і статистикою символів.

Якщо погоджуєтесь із цією концепцією, наступний крок — зафіксувати список конкретних полів BacktestSettings і перший набір optimizable params для запуску “quick test”, а також сценарії run_quick_test і run_full_test у вигляді реальних функцій.

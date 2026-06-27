# Falcon 存档 — 2026-06-27 深夜

## 当前状态
- **版本**: V0.3.1 (统一框架 + Russell + Hybrid + OOS验证)
- **结论**: alpha真实, 80/20 Hybrid是最优配置
- **下一步**: V0.4实盘系统(分钟级止损+新闻流+Analyst监控)

## 文件路径

### 脚本 (scripts/falcon/)
| 文件 | 用途 | 状态 |
|------|------|------|
| `falcon_v03.py` | **主脚本** — 统一回测, --universe spx/r2k/both | ✅ 生产 |
| `falcon_v03_engine.py` | 共享引擎(PIT rank + 灵活调仓 + Futu成本) | ✅ 生产 |
| `falcon_hybrid.py` | 简单Hybrid(SPX+R2K固定比例加权) | ✅ 生产 |
| `falcon_v02_fast.py` | V0.2快速网格搜索(旧, 保留参考) | 📦 归档 |
| `falcon_pipeline.py` | V0.1/V0.2原始管线(旧, 保留参考) | 📦 归档 |
| `falcon_regime.py` | Regime过滤测试(旧) | 📦 归档 |
| `falcon_optimize.py` | 动态止损优化(旧) | 📦 归档 |
| `falcon_reality_check.py` | 前视偏差审计 | 📦 归档 |
| `falcon_v03_backtest.py` | V0.3 SPX专用(已被falcon_v03.py替代) | 📦 可删 |
| `falcon_v03_russell.py` | V0.3 R2K专用(已被falcon_v03.py替代) | 📦 可删 |
| `fetch_all_fmp.py` | FMP全量数据拉取(40季度历史) | ✅ 工具 |
| `fetch_historical_fmp.py` | FMP历史拉取(早期版本) | 📦 可删 |
| `fetch_russell_data.py` | Russell 2000数据拉取 | ✅ 工具 |

### 数据 (data/falcon/)
| 文件 | 内容 | 大小 |
|------|------|------|
| `features_v02.parquet` | SPX 80列特征矩阵(476只×753天) | 核心 |
| `russell_prices.json` | R2K日K数据(691只) | 49MB |
| `fmp_ratios_historical.json` | SPX FMP Ratios(40季度) | 18.7MB |
| `fmp_ratios_russell.json` | R2K FMP Ratios | 13.4MB |
| `fmp_key_metrics.json` | SPX Key Metrics | 17.0MB |
| `fmp_metrics_russell.json` | R2K Key Metrics | 12.2MB |
| `fmp_financial_growth.json` | SPX Growth | 14.7MB |
| `fmp_growth_russell.json` | R2K Growth | 7.8MB |
| `analyst_historical.json` | SPX Analyst | 4.9MB |
| `fmp_analyst_russell.json` | R2K Analyst | 2.3MB |
| `falcon_v03_unified.csv` | 统一回测结果(64×2组合) | 输出 |

### 配置 (config/)
| 文件 | 内容 |
|------|------|
| `universe_sp500.csv` | 501只SPX ticker(Wikipedia) |
| `active_weights.yaml` | Mercurius旧权重(待更新) |

## 最优参数固定

### SPX配置
```yaml
universe: spx
tickers: 476
weights: {fund_ratio: 0.7, analyst: 0.2, fund_metric: 0.1, tech: 0.0}
strategy: fixed
hold_days: 30
stop_loss: -0.15
bear_alloc: 0.50
execution: T+1 Limit Order
cost_model: futu_tiered
```

### R2K配置
```yaml
universe: r2k
tickers: 512 (有FMP数据)
weights: {fund_ratio: 0.5, fund_metric: 0.3, fund_growth: 0.2, tech: 0.0}
strategy: fixed
hold_days: 10
stop_loss: -0.15
bear_alloc: 0.30
execution: T+1 Limit Order
cost_model: futu_tiered
```

### Hybrid配置
```yaml
type: simple_fixed_ratio
spx_weight: 0.80
r2k_weight: 0.20
# 各自用各自最优参数, 日收益加权
regime_proxy: {spx: SPY_MA200, r2k: IWM_MA200}  # 待实现
```

## 回测结果速查

| | SPX | R2K | Hybrid 80/20 |
|---|---|---|---|
| 全样本Sharpe | 1.676 | 0.818 | 1.689 |
| 全样本MaxDD | 16.9% | 19.0% | 15.1% |
| 牛市Sharpe | 2.206 | 1.235 | 2.254 |
| 熊市Sharpe | -0.194 | 0.439 | -0.037 |
| 熊市DD | 12.4% | 19.0% | 11.1% |
| OOS Sharpe(24H2) | 1.836 | 1.492 | — |
| 通过率 | 22/64 | 47/64 | — |
| SPX-R2K相关性 | — | — | 0.327 |

## 待办 (优先级排序)

### P0 — 进实盘前必须做
- [ ] OOS验证: 用2024 H2数据做样本外测试 ✅ (已通过)
- [ ] 分钟级止损: Alpaca WebSocket实时监控持仓价格
- [ ] 盘前评分清单: 每天收盘后自动生成买/卖/限价单

### P1 — 实盘后第一周做
- [ ] 新闻流: FMP+Massive叠加+去重+FinBERT打分
- [ ] Analyst变化检测: 每小时轮询FMP, 检测评级变动
- [ ] 双Regime proxy: SPX用SPY_MA200, R2K用IWM_MA200

### P2 — 实盘稳定后优化
- [ ] 自动下单: Alpaca Paper Trading API
- [ ] R2K精选池: 只保留≥2家分析师覆盖的ticker
- [ ] Futu实盘接入: 从Alpaca Paper切换到Futu OpenD

### 不做
- ❌ 三维网格搜索(Hybrid权重×VIX×regime) — 过拟合风险
- ❌ 三套Paper并行 — 一套够用, 归因靠记录
- ❌ Insider/DCF因子 — 数据质量差, 已验证无增量

## API数据源状态
| API | 端点 | 状态 | 用途 |
|-----|------|------|------|
| Massive | api.massive.com/v2/aggs | ✅ | 日K+分钟K |
| FMP | /stable/ratios | ✅ | 基本面(20个) |
| FMP | /stable/key-metrics | ✅ | 关键指标(23个) |
| FMP | /stable/financial-growth | ✅ | 增长(18个) |
| FMP | /stable/analyst-estimates | ✅ | 分析师(3个) |
| FMP | /stable/news/stock | ✅ | 新闻(待接入) |
| Alpaca | Paper Trading API | ✅ | 模拟交易(待接入) |
| FMP | /v3/analyst-estimates | ❌ 404 | 用/stable/替代 |
| FMP | insider-trading | ⚠️ 500条限制 | 结构性缺陷, 已排除 |

## 关键教训
1. **PIT > 静态**: FMP用最新数据回测历史 = 31%虚假alpha
2. **执行模式很重要**: T+1 Limit比T收盘好13%(隔夜漂移)
3. **FMP付费数据=alpha核心**: Fund占70%权重, 技术特征占0%
4. **小盘alpha形态不同**: 分析师覆盖是质量筛选, Fixed比Signal好
5. **Hybrid不提升Sharpe, 只降波动**: 80/20 DD降1.8pp
6. **OOS必须验证**: Sharpe 2.35看起来漂亮, OOS 1.84才是真实水平

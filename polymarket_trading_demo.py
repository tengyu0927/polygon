"""
Polymarket 基础交易示例（官方 Python SDK：polymarket-client）
=============================================================

本文件演示如何在 Polymarket（全球最大的预测市场）上用官方 Python SDK 完成
从「查询行情」到「下单 / 撤单 / 查持仓」的完整交易流程。

⚠️ 安全提示
-----------
- 默认开启「干跑模式」(DRY_RUN=1)，只做只读操作（查市场、查余额、查订单、查持仓），
  不会真正下单 / 撤单。请先跑一遍看效果，确认无误后再关闭干跑。
- 下单会消耗真实资金（pUSD）。请务必使用小额（如 1~10 美元）测试。
- 私钥等同于资产所有权，绝不要提交到 git 或公开分享。

安装
----
    pip install polymarket-client

环境变量（推荐写进 .env 或 shell 配置，不要硬编码在代码里）
----------------------------------------------------------
    POLYMARKET_PRIVATE_KEY    # 你的钱包私钥（0x 开头的 64 位 hex）
    POLYMARKET_WALLET_ADDRESS # 你的 Polymarket 钱包地址（个人资料里可看到）
    POLYMARKET_DRY_RUN        # 1=干跑（默认，只读），0=真实交易

运行
----
    python polymarket_trading_demo.py
"""

from __future__ import annotations

import asyncio
import os
import time
from decimal import Decimal

# 官方 SDK 的客户端：
#   - AsyncPublicClient : 只读公开数据（行情、市场、订单簿）
#   - AsyncSecureClient : 需私钥，可交易 / 查账户
# 同时存在同步版本 PublicClient / SecureClient，本示例统一用 async 版本。
from polymarket import (
    AsyncPublicClient,
    AsyncSecureClient,
)

# USDC（Polymarket 的抵押品，也称 pUSD）在小数上有 6 位精度。
# 余额接口返回的是「最小单位」的整数，除以 1e6 才是美元金额。
USDC_DECIMALS = 6

# 读取配置；干跑模式默认开启。
PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY")
WALLET_ADDRESS = os.getenv("POLYMARKET_WALLET_ADDRESS")
DRY_RUN = os.getenv("POLYMARKET_DRY_RUN", "1") == "1"


# ---------------------------------------------------------------------------
# 0. 认证：创建交易客户端（AsyncSecureClient）
# ---------------------------------------------------------------------------
async def create_secure_client() -> AsyncSecureClient:
    """用私钥 + 钱包地址创建可交易的客户端。

    私钥用于对订单做签名，钱包地址用于定位你的 Polymarket 账户。
    创建过程中 SDK 会自动推导出 CLOB API 凭证，并确保钱包已就绪。
    """
    if not PRIVATE_KEY or not WALLET_ADDRESS:
        raise SystemExit(
            "缺少环境变量：请先设置 POLYMARKET_PRIVATE_KEY 和 "
            "POLYMARKET_WALLET_ADDRESS 后再运行。"
        )

    client = await AsyncSecureClient.create(
        private_key=PRIVATE_KEY,
        wallet=WALLET_ADDRESS,
    )
    print(f"[认证] 已登录钱包: {WALLET_ADDRESS}")
    return client


# ---------------------------------------------------------------------------
# 1. 只读行情：查市场（无需私钥，用公开客户端即可）
# ---------------------------------------------------------------------------
async def demo_list_markets() -> None:
    """列出活跃市场，并打印前几条。

    说明：
    - AsyncPublicClient 不需要私钥，任何数据都是公开可查的。
    - list_markets() 返回一个「分页器」，用 iter_items() 逐条迭代。
    - 可用参数：closed=False（只看未结束的）、page_size、tag_id 等。
    """
    async with AsyncPublicClient() as client:
        # closed=False 表示只看还在交易中的市场
        markets = client.list_markets(closed=False)

        print("\n[行情] 当前活跃市场（前 5 条）：")
        count = 0
        async for market in markets.iter_items():
            # 每个 market 是 Market 对象，常用字段：
            #   question             问题文本
            #   condition_id         市场唯一 ID（查持仓时用到）
            #   outcomes             结果（yes/no 两个 token，.yes.token_id）
            #   metrics.liquidity_num 流动性（美元）
            #   metrics.volume_num    成交量（美元）
            print(
                f"  - {market.question} | "
                f"流动性 ${float(market.metrics.liquidity_num or 0):,.0f} | "
                f"成交量 ${float(market.metrics.volume_num or 0):,.0f}"
            )
            count += 1
            if count >= 5:
                break


async def get_one_active_market():
    """随便抓取一个活跃市场，作为后续下单的「标的」示例。"""
    async with AsyncPublicClient() as client:
        markets = client.list_markets(closed=False)
        first_page = await markets.first_page()  # Page 对象，.items 是市场列表
        if not first_page.items:
            raise SystemExit("没有找到活跃市场。")
        market = first_page.items[0]
        print(f"\n[标的] 选择市场: {market.question}")
        return market


# ---------------------------------------------------------------------------
# 2. 查余额
# ---------------------------------------------------------------------------
async def demo_balance(client: AsyncSecureClient) -> None:
    """查询账户的 USDC（pUSD）余额。

    get_balance_allowance(asset_type="COLLATERAL") 返回 BalanceAllowance：
      balance     余额（最小单位整数，除以 1e6 得美元）
      allowances  已授权的额度（给交易所合约的授权，下单前需有足够授权）
    """
    allowance = await client.get_balance_allowance(asset_type="COLLATERAL")
    balance_usd = Decimal(allowance.balance) / Decimal(10**USDC_DECIMALS)
    print(f"\n[余额] 可用 USDC: ${balance_usd:,.2f}")


# ---------------------------------------------------------------------------
# 3. 市价单（market order）：立刻以对手价成交
# ---------------------------------------------------------------------------
async def demo_market_buy(client: AsyncSecureClient, token_id: str) -> None:
    """市价买入：花约 1 USDC 买入某个结果 token。

    参数说明（place_market_order）：
      token_id   要交易的「结果 token」ID（market.outcomes.yes.token_id 等）
      side       "BUY" 买入 / "SELL" 卖出
      amount     买入时 = 要花的美元金额（不含手续费的本金）
      max_spend  可选，包含手续费在内的总花费上限（防止超预算）
      max_price  可选，成交均价上限（防止买贵了）
      order_type "FAK"（部分成交即撤）/ "FOK"（要么全成要么不成），默认 FAK

    卖出时不用 amount，改用：
      shares     要卖出的份额数
      min_price  成交的最低价下限

    返回 OrderResponse：
      ok         是否被接受
      order_id   订单 ID
      （若失败）message 错误信息
    """
    if DRY_RUN:
        print("\n[市价买入] 干跑模式：跳过真实下单。真实调用示例：")
        print("  await client.place_market_order(")
        print("      token_id=token_id, side='BUY', amount='1',")
        print("      max_spend='1', order_type='FAK')")
        return

    # 小额测试：最多花 1 USDC（含手续费）
    response = await client.place_market_order(
        token_id=token_id,
        side="BUY",
        amount="1",        # 想买 1 美元
        max_spend="1",     # 含手续费总共不超过 1 美元
        order_type="FAK",  # 立即成交，未成交部分自动撤销
    )

    if not response.ok:
        # 下单被拒绝时，response.message 会说明原因（余额不足 / 市场已结束等）
        print(f"[市价买入] 下单失败: {response.message}")
        return

    print(f"[市价买入] 已提交，订单 ID: {response.order_id}")

    # 成交后需要在链上「结算」，等待结算完成：
    hashes = await client.wait_for_order_fill_settlement(response)
    print(f"[市价买入] 结算完成，交易哈希: {hashes}")


# ---------------------------------------------------------------------------
# 4. 限价单（limit order）：指定价格挂单，等别人来成交
# ---------------------------------------------------------------------------
async def demo_limit_buy(client: AsyncSecureClient, token_id: str) -> None:
    """限价买入：以指定价格挂单。

    参数说明（place_limit_order）：
      price      你愿意成交的价格（0~1，例如 0.50 = 50 美分）
      size       要买入的份额数量
      side       "BUY" / "SELL"
      post_only  是否只做 maker（挂单不主动吃单），默认 False
      expiration Unix 时间戳（秒）；不传 = GTC 一直有效；
                 传了 = GTD 到点自动过期（须至少比当前晚 3 分钟）

    价格必须符合市场的最小价格步长（tick size），份额要满足最小下单量。
    """
    if DRY_RUN:
        print("\n[限价买入] 干跑模式：跳过真实挂单。真实调用示例：")
        print("  await client.place_limit_order(")
        print("      token_id=token_id, price='0.50', size='10', side='BUY')")
        return

    # 5 分钟后过期（GTD 订单），避免挂单一直挂着
    expiration = int(time.time()) + 300

    response = await client.place_limit_order(
        token_id=token_id,
        price="0.50",   # 50 美分买入
        size="10",      # 买 10 份
        side="BUY",
        expiration=expiration,
    )

    if not response.ok:
        print(f"[限价买入] 挂单失败: {response.message}")
        return

    print(f"[限价买入] 已挂单，订单 ID: {response.order_id}")


# ---------------------------------------------------------------------------
# 5. 查询 / 撤单
# ---------------------------------------------------------------------------
async def demo_list_and_cancel_orders(client: AsyncSecureClient) -> None:
    """列出当前未成交的挂单，并把它们全部撤掉（仅干跑关闭时执行撤单）。

    list_open_orders() 返回分页器，元素是 OpenOrder，常用字段：
      id / side / price / size_matched / status ...
    cancel_order(order_id=...) 撤单笔；cancel_orders(order_ids=[...]) 撤多笔；
    cancel_all() 一键全撤。
    """
    print("\n[挂单] 当前未成交订单：")
    orders = client.list_open_orders()
    open_ids: list[str] = []
    async for order in orders.iter_items():
        open_ids.append(order.id)
        print(
            f"  - {order.id[:12]}... | {order.side} | "
            f"价格 {order.price} | 状态 {order.status}"
        )

    if not open_ids:
        print("  （无未成交订单）")
        return

    if DRY_RUN:
        print("[挂单] 干跑模式：跳过撤单。")
        return

    result = await client.cancel_orders(order_ids=open_ids)
    print(
        f"[挂单] 已撤销 {len(result.canceled)} 笔，"
        f"未撤销 {len(result.not_canceled)} 笔"
    )


# ---------------------------------------------------------------------------
# 6. 查持仓
# ---------------------------------------------------------------------------
async def demo_positions(client: AsyncSecureClient, condition_id: str) -> None:
    """列出你在某个市场（condition_id）里的持仓。

    list_positions(market=[condition_id]) 返回分页器，元素是 Position。
    每个 Position 对应一个结果 token 的持仓，常用字段：
      token_id / size（持有份额）/ avg_price（成本价）等。
    """
    print("\n[持仓] 该市场下的持仓：")
    positions = client.list_positions(market=[condition_id])
    async for position in positions.iter_items():
        print(f"  - token: {position.token_id[:12]}... | 份额: {position.size}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
async def main() -> None:
    mode = "干跑模式，只读" if DRY_RUN else "真实模式，会下单"
    print("=" * 60)
    print(f"Polymarket 交易示例（{mode}）")
    print("=" * 60)

    # 只读部分：不需要私钥
    await demo_list_markets()
    market = await get_one_active_market()

    # 需要私钥的交易部分
    client = await create_secure_client()
    try:
        await demo_balance(client)

        # 拿到该市场的 yes/no token_id 和 condition_id
        yes_token_id = market.outcomes.yes.token_id
        no_token_id = market.outcomes.no.token_id
        condition_id = market.condition_id
        if yes_token_id is None or no_token_id is None or condition_id is None:
            raise SystemExit("该市场缺少 token_id / condition_id，无法演示下单。")

        await demo_market_buy(client, yes_token_id)         # 市价买入 yes
        await demo_limit_buy(client, yes_token_id)          # 限价买入 yes
        await demo_list_and_cancel_orders(client)           # 列挂单 / 撤单
        await demo_positions(client, condition_id)          # 查持仓
    finally:
        # 用完关闭客户端，释放底层连接
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())

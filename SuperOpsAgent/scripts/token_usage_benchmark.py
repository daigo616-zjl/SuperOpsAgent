"""Token 占用对比基准：短期记忆压缩路径 vs 全量历史路径。

模拟一段 20 轮运维排障会话，两条路径使用相同的轮次内容：
- baseline：每轮 prompt 携带全部历史消息（无压缩）。
- compressed：走真实 ShortTermMemory（Redis 滑动窗口 + 滚动摘要），
  拼装方式与 rag_agent_service._start_turn 一致。

两条路径都真实调用 chat model，统计 provider 返回的 prompt_tokens；
压缩旁路的摘要模型调用（input+output）单独记账，最后给出净节省。
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402

from app.config import config  # noqa: E402
from app.core.llm_factory import LLMFactory  # noqa: E402
from app.memory.short_term import (  # noqa: E402
    ShortTermMemory,
    SummaryGenerator,
    window_messages_to_langchain,
)

SYSTEM_PROMPT = "你是运维排障助手，基于对话历史回答用户问题。"

# 20 轮确定性对话（user, assistant），模拟真实排障的长度分布
TURNS = [
    ("我们订单服务今晚 8 点开始频繁超时，网关日志里全是 504，先从哪查？",
     "504 说明上游超时。建议顺序：1) 看网关到订单服务的 upstream 延迟分布；2) 看订单服务自身 CPU/线程池；3) 看下游 DB 与缓存。先确认是整体变慢还是部分接口。"),
    ("网关侧看到订单服务的 P99 从 200ms 涨到了 8s，但 CPU 才 40%。",
     "CPU 不高但延迟暴涨，常见原因是等下游：连接池耗尽、DB 锁等待、或 GC 停顿。先查线程 dump 里大量线程 BLOCKED/WAITING 在哪个点。"),
    ("线程 dump 显示 120 个 Tomcat 线程里 100 个 WAITING 在 getConnection。",
     "基本确定是数据库连接池耗尽。看下 HikariPool 的 active/pending 指标，以及有没有慢查询占住连接不放。"),
    ("Hikari active=20 满了，pending 队列 90 多。慢查询日志里有条库存扣减 SQL 跑了 6 秒。",
     "锁热点 SQL 占住了连接。看这条 SQL 的执行计划，大概率是索引失效走了全表扫，或者行锁等待。库存表更新频繁，优先确认事务里有没有包住远程调用。"),
    ("执行计划显示全表扫，stock 表 2000 万行，idx_sku 索引在但没走。",
     "索引失效常见原因：隐式类型转换（sku 字段 varchar 传了数字）、函数包裹列、或统计信息过期。先看这条 SQL 的参数类型是否与列类型一致。"),
    ("确实是隐式转换，代码里把 Long 拼成了字符串传参。但为什么以前没这个问题？",
     "以前没暴露可能是因为数据量小全表扫也快，或走了另一条执行计划。20 天前 stock 表有过大批量导入的话，统计信息变化会让优化器改选计划。"),
    ("修复了类型转换，超时降下来了。但我想知道这 6 秒的慢查询为什么没触发告警？",
     "查三点：1) 慢查询阈值配置是否高于 6s；2) 告警规则的持续时间窗口是否太长被静默；3) 告警通知渠道是否正常送达。很多'没告警'其实是被抑制规则吞了。"),
    ("阈值设的是 5 秒，但通知走了钉钉机器人，机器人 token 上个月过期了。",
     "这就是根因了。告警链路要加自监控：机器人发送失败也应有兜底通知（比如邮件或短信），并且定期发心跳消息验证通道活性。"),
    ("换一个问题：我们 nightly 全量备份任务最近经常跑到天亮还没结束，之前 2 小时就完了。",
     "备份变慢先看 IO 和锁：1) 备份期间是否有大批量写入任务撞在一起；2) 是否改成了一致性快照模式导致更长的事务持有；3) 磁盘 IO 利用率是否被打满。"),
    ("检查发现是上周上线的报表任务和备份撞在同一时段，把 buffer pool 打满了。",
     "资源排期冲突。给报表任务加执行窗口或资源组限制，错峰到备份完成后；同时给备份加超时与失败告警，别让它静默跑十几个小时。"),
    ("错峰后备份恢复 2 小时。下一个问题：K8s 里有个 Pod 反复 OOMKilled，limit 设了 4Gi。",
     "OOMKilled 是容器内存超 limit。先看是 JVM 堆内还是堆外：如果是容器里跑 Java，确认 -XX:MaxRAMPercentage 与容器 limit 匹配，堆外 native 内存往往是被忽略的大头。"),
    ("Java 应用，heap 用了 2.8G，但容器内存 3.9G，快顶到 limit。",
     "差距 1G+ 是堆外：metaspace、线程栈、DirectBuffer、JNI。用 pmap/NMT 看 native 分配，重点怀疑 DirectBuffer 和 gzip 类 native 缓存。"),
    ("NMT 显示 DirectBuffer 占了 800M，代码里有大量 Netty 调用。",
     "Netty 默认用池化 DirectBuffer，上限由 -XX:MaxDirectMemorySize 控制。显式设一个合理上限（如 1G）让它受控，同时确认有没有 ByteBuf 泄漏（开启泄漏检测 -Dio.netty.leakDetection.level=paranoid 短期观察）。"),
    ("设了 MaxDirectMemorySize=1G 后稳定了。再问个容量规划问题：大促流量预计是日常 5 倍，连接池怎么算？",
     "连接池不是越大越好。按公式：connections = (core_count * 2) + effective_spindle_count 起步，再按单请求 DB 耗时和 QPS 推导。5 倍流量先压测，看 DB 侧 max_connections 和锁竞争，别只堆应用侧连接数。"),
    ("压测发现 DB 先到瓶颈，QPS 3 万时 CPU 90%。读写分离能把读流量分走多少？",
     "取决于读写比。先统计：如果读占 80%，读写分离理论上能卸掉大部分读压力。但要注意复制延迟——读己之写的场景不能路由到从库，库存扣减这类强一致读必须走主库。"),
    ("读写比大概 85:15。从库延迟平时 100ms 以内，大促时会不会放大？",
     "会。大促时大批量写入会造成复制风暴，延迟可能到秒级。对策：1) 从库并行复制调优；2) 业务侧对延迟敏感的读加 fallback 主库逻辑；3) 监控延迟并动态摘除落后过多的从库。"),
    ("好，读写分离方案定了。最后一个问题：这套排查过程怎么沉淀成团队的知识库？",
     "把每次故障按固定模板沉淀：现象、排查路径、根因、修复、预防措施。关键是可检索——按服务名、错误码、症状打标签，让下次 'getConnection WAITING' 这种关键词能直接搜到这次的处理结论。"),
    ("模板定了，标签也打了。但新人说搜出来的文档太多，不知道先看哪篇。",
     "检索结果要排序和瘦身：按故障类型匹配度排序、同类文档合并、每篇开头放 TL;DR。更进一步可以让 AI 助手先消化文档再给答案，新人用自然语言问就行，不用自己翻文档。"),
    ("这个想法好，我们正打算做运维知识库 + RAG 问答。向量库选型有什么建议？",
     "看数据规模和运维场景特点：错误码、服务名这类精确词需要 BM25 或混合检索兜底，纯向量对专有名词召回差。建议混合检索 + 重排序，元数据按服务/组件过滤。"),
    ("明白了，混合检索 + 重排序。今天聊的这些我先整理成故障复盘文档。",
     "复盘文档记得把今天的量化数据带上：P99 从 200ms 到 8s 的现象、连接池 20/90 的水位、DirectBuffer 800M 的构成。有数字的复盘对下次容量规划最有用。"),
]


def usage_tokens(resp) -> tuple[int, int]:
    um = getattr(resp, "usage_metadata", None) or {}
    return um.get("input_tokens", 0), um.get("output_tokens", 0)


class CountingSummarizer(SummaryGenerator):
    """记录摘要模型的 token 开销"""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    async def summarize(self, old_summary: str, messages: list[dict]) -> str:
        model = self._get_model()
        resp = await model.ainvoke(self._prompt(old_summary, messages))
        in_t, out_t = usage_tokens(resp)
        self.calls += 1
        self.input_tokens += in_t
        self.output_tokens += out_t
        content = resp.content
        if isinstance(content, list):
            text = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        else:
            text = str(content)
        return text.strip()


async def main() -> None:
    model = LLMFactory.create_qwen_chat_model(
        model=config.dashscope_model,
        temperature=0,
        streaming=False,
        max_tokens=32,
        enable_thinking=False,
    )
    summarizer = CountingSummarizer()
    memory = ShortTermMemory(summarizer=summarizer)
    session_id = f"token-bench-{int(time.time())}"

    if not await memory.available():
        print("FATAL: Redis 短期记忆不可用，无法对比", file=sys.stderr)
        sys.exit(1)
    await memory.clear(session_id)

    rows = []
    base_total = comp_total = 0

    for i, (question, answer) in enumerate(TURNS, start=1):
        # ---- baseline：全量历史 ----
        base_msgs = [SystemMessage(content=SYSTEM_PROMPT)]
        for prev_q, prev_a in TURNS[: i - 1]:
            base_msgs.append(HumanMessage(content=prev_q))
            base_msgs.append(AIMessage(content=prev_a))
        base_msgs.append(HumanMessage(content=question))
        resp = await model.ainvoke(base_msgs)
        b_in, _ = usage_tokens(resp)
        base_total += b_in

        # ---- compressed：Redis 滑动窗口 + 滚动摘要 ----
        summary, window = await memory.build_context(session_id)
        comp_msgs = [SystemMessage(content=SYSTEM_PROMPT)]
        if summary:
            comp_msgs.append(SystemMessage(content=f"[历史对话摘要]\n{summary}"))
        comp_msgs.extend(window_messages_to_langchain(window))
        comp_msgs.append(HumanMessage(content=question))
        resp = await model.ainvoke(comp_msgs)
        c_in, _ = usage_tokens(resp)
        comp_total += c_in

        await memory.append_turn(session_id, question, answer)
        rows.append((i, b_in, c_in, len(window), len(summary)))

    await memory.clear(session_id)

    print(f"\n{'turn':>4} {'baseline':>10} {'compressed':>11} {'win_msgs':>9} {'summary_len':>12}")
    for i, b, c, w, s in rows:
        print(f"{i:>4} {b:>10} {c:>11} {w:>9} {s:>12}")

    summary_overhead = summarizer.input_tokens + summarizer.output_tokens
    saved = base_total - comp_total
    net_saved = saved - summary_overhead
    print("\n==== 汇总（prompt tokens，来自 provider usage）====")
    print(f"轮数:                {len(TURNS)}")
    print(f"baseline 总量:       {base_total}")
    print(f"compressed 总量:     {comp_total}")
    print(f"prompt 节省:         {saved}  ({saved / base_total * 100:.1f}%)")
    print(f"摘要模型开销:        {summary_overhead}  ({summarizer.calls} 次调用, "
          f"in {summarizer.input_tokens} / out {summarizer.output_tokens})")
    print(f"净节省:              {net_saved}  ({net_saved / base_total * 100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())

"""B 站评论：草稿生成、内容校验、发布与状态记录。

这一层只做「在 B 站视频下发一条一级评论」，不点赞、不投币、不发弹幕。
设计上参照 bilibili_autopilot 的几条硬约束：

- **预览与发布分离**：``[comment].publish_enabled = false`` 时只生成草稿并落
  ``state.json``，不发任何写请求；
- **意图先落库，POST 只调一次**：发送前把草稿与时间写进状态，超时/异常记
  ``unknown`` 并暂停评论，等人工去 B 站核对后再 ``/bili-comment-ack`` 解除；
- **额度 + 最小间隔 + 活动时段**三重闸，任一不满足都只分享不评论；
- **内容依据**：模型必须输出它引用的素材原句，程序逐条回查；对不上就回炉一次，
  仍对不上直接放弃，不修补后硬发；
- **不泄露身份**：链接、@、联系方式、求三连、私聊里的称呼一律拒绝。

评论内容会公开挂在 B 站，所以默认全关（``enabled = false``）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("plugin.bilibili_interest_share.comment")

# 平台上限是 1000 字；160 是本插件的产品选择（够说清一句，也不像刷屏）
_MAX_COMMENT_CHARS = 160
# 短评论可以没有任何事实引用（纯反应），长评论必须给得出依据
_EVIDENCE_FREE_MAX_CHARS = 60
# 与弹幕逐字重合的判定长度：≥ 这个长度的整句出现在评论里就算抄弹幕
_DANMAKU_COPY_MIN_CHARS = 12

_URL_RE = re.compile(r"(https?://|www\.|b23\.tv|bilibili\.com|BV[0-9A-Za-z]{10})", re.I)
_MENTION_RE = re.compile(r"[@＠]\S")
_DIGIT_NOISE_RE = re.compile(r"\d{6,}")

# 默认禁用的表达：推广话术 + 任何可能指向私域联系方式的写法
_DEFAULT_BLOCK_WORDS: Tuple[str, ...] = (
    "三连", "一键三连", "点赞", "投币", "收藏", "关注我", "求关注",
    "加群", "进群", "微信群", "QQ群", "私信我", "加我",
    "微信", "vx", "wx", "v信", "扣扣", "qq号",
    "下单", "优惠", "折扣", "链接在", "购买", "代购", "课程",
)
# 私聊关系/身份相关称呼不再写死在代码里：人设里的名字与别名由
# ``LlmJudge.persona_identity_terms()`` 动态提供，其余自定义词走
# ``[comment].privacy_patterns``。

# B 站返回码 → 处理策略
_FATAL_CODES: Dict[int, str] = {
    -101: "账号未登录或登录已失效",
    -111: "csrf 校验失败（bili_jct 失效）",
    -412: "请求被风控拦截",
    12061: "账号需要绑定手机号",
}
_SKIP_CODES: Dict[int, str] = {
    12009: "该视频评论区已关闭",
    12002: "评论内容被平台拒绝",
}


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _norm(text: str) -> str:
    """去掉空白与标点，只留文字与数字，用于依据回查。"""

    return re.sub(r"[\s\W_]+", "", (text or "").lower())


def _privacy_hit(term: str, text: str) -> bool:
    """隐私词命中判定：两个字符以上按子串；单字只在该字独立出现时命中。"""

    term = (term or "").strip()
    if not term:
        return False
    if len(term) >= 2:
        return term in (text or "")
    pattern = rf"(?<![\u4e00-\u9fff]){re.escape(term)}(?![\u4e00-\u9fff])"
    return re.search(pattern, text or "") is not None


@dataclass
class CommentEvidence:
    """判断/写评论时可用的素材。字段名与插件其它部分保持一致。"""

    danmaku: Sequence[str] = field(default_factory=tuple)
    comments: Sequence[str] = field(default_factory=tuple)
    subtitle: str = ""
    frames_desc: str = ""
    description: str = ""
    tags: str = ""

    def richness(self, *, min_subtitle: int = 300, min_desc: int = 80, min_danmaku: int = 20) -> int:
        """素材充足度：字幕/简介/弹幕/热评/画面 五项里满足几项。"""

        score = 0
        if len(self.subtitle or "") >= min_subtitle:
            score += 1
        if len(self.description or "") >= min_desc:
            score += 1
        if len(self.danmaku or ()) >= min_danmaku:
            score += 1
        if len(self.comments or ()) >= 3:
            score += 1
        if (self.frames_desc or "").strip():
            score += 1
        return score

    def source_text(self) -> str:
        parts: List[str] = []
        parts.extend(str(item) for item in self.danmaku or ())
        parts.extend(str(item) for item in self.comments or ())
        for value in (self.subtitle, self.frames_desc, self.description, self.tags):
            if value:
                parts.append(str(value))
        return "\n".join(parts)

    def prompt_block(self) -> str:
        lines: List[str] = []
        if self.tags:
            lines.append(f"标签：{_clip(self.tags, 120)}")
        if self.description:
            lines.append(f"简介：{_clip(self.description, 220)}")
        if self.frames_desc:
            lines.append(f"画面（抽帧识别）：{_clip(self.frames_desc, 300)}")
        if self.subtitle:
            lines.append(f"字幕节选：{_clip(self.subtitle, 400)}")
        if self.danmaku:
            lines.append("弹幕节选：" + " / ".join(_clip(str(x), 40) for x in list(self.danmaku)[:20]))
        if self.comments:
            lines.append("热门评论：" + " / ".join(_clip(str(x), 60) for x in list(self.comments)[:8]))
        return "\n".join(lines) if lines else "（没有可用素材）"


# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------


def build_comment_prompt(
    video: Any,
    evidence: CommentEvidence,
    *,
    persona_block: str = "",
    style_prompt: str = "",
    max_chars: int = _MAX_COMMENT_CHARS,
) -> str:
    """拼出写评论的 prompt。"""

    lines: List[str] = []
    if persona_block.strip():
        lines.append(persona_block.strip())
        lines.append("")
        lines.append("下面请你完全以这个角色的身份和口吻，在 B 站这个视频下面写一条评论。")
        lines.append("")
    lines.append("【视频】")
    lines.append(f"标题：{getattr(video, 'title', '')}")
    owner = getattr(video, "owner", "") or ""
    partition = getattr(video, "partition", "") or ""
    if owner or partition:
        lines.append(" ".join(x for x in (f"UP：{owner}" if owner else "", f"分区：{partition}" if partition else "") if x))
    lines.append("")
    lines.append("【可用素材】")
    lines.append(evidence.prompt_block())
    lines.append("")
    lines.append("【写评论的要求】")
    lines.append(f"- 长度不超过 {max_chars} 字，一到两句话，像普通观众看完随手留的；")
    lines.append("- 只聊素材里真实出现的内容，**不要**编造视频里的画面、台词、数据或自己的亲身经历；")
    lines.append("- 不要写“推荐大家去看”“值得一看”这类推广腔，也不要提点赞/投币/关注/加群；")
    lines.append("- 不要放任何链接、@、联系方式，也不要提私下的称呼和身份；")
    lines.append("- 不要复述弹幕原句，也不要堆网络流行语；")
    style = (style_prompt or "").strip()
    if style:
        lines.append(f"- 额外语气要求：{style}")
    lines.append("")
    lines.append("只输出一个 JSON 对象，不要输出其它文字：")
    lines.append('{"text": "评论正文", "evidence": ["从素材里原样摘抄的依据片段", "..."]}')
    lines.append(
        "evidence 里每条都必须能在上面的素材中逐字找到；"
        "如果你写的是纯感受（不涉及具体事实），evidence 可以是空数组。"
    )
    return "\n".join(lines)


def parse_draft(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出里抠出 JSON 草稿。"""

    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    body = str(data.get("text") or data.get("comment") or "").strip()
    if not body:
        return None
    evidence = data.get("evidence")
    items = [str(x).strip() for x in evidence if str(x or "").strip()] if isinstance(evidence, list) else []
    return {"text": body, "evidence": items[:5]}


# ----------------------------------------------------------------------
# 校验
# ----------------------------------------------------------------------


def validate_draft(
    draft: Dict[str, Any],
    evidence: CommentEvidence,
    *,
    max_chars: int = _MAX_COMMENT_CHARS,
    extra_block_words: Sequence[str] = (),
    privacy_words: Sequence[str] = (),
) -> Tuple[bool, str, str]:
    """返回 ``(是否通过, 原因, 清洗后的正文)``。"""

    text = str(draft.get("text") or "").strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return False, "empty", ""
    if len(text) > max_chars:
        return False, f"too_long({len(text)}>{max_chars})", text
    if _URL_RE.search(text):
        return False, "contains_link", text
    if _MENTION_RE.search(text):
        return False, "contains_mention", text
    if _DIGIT_NOISE_RE.search(text):
        return False, "contains_long_number", text

    lowered = text.lower()
    blocked = [w for w in _DEFAULT_BLOCK_WORDS if w and w.lower() in lowered]
    blocked += [w for w in extra_block_words if w and w.lower() in lowered]
    if blocked:
        return False, f"blocked_words({','.join(blocked[:3])})", text
    privacy = [w for w in privacy_words if _privacy_hit(w, text)]
    if privacy:
        return False, f"privacy_words({','.join(privacy[:3])})", text

    norm_text = _norm(text)
    # 纯感受可以没有依据；但只要写了依据，就必须逐条对得上
    claimed = [str(item) for item in draft.get("evidence") or () if str(item or "").strip()]
    source = _norm(evidence.source_text())
    for item in claimed:
        needle = _norm(item)
        if len(needle) < 2:
            return False, f"evidence_too_short({_clip(item, 12)})", text
        if needle not in source:
            return False, f"evidence_mismatch({_clip(item, 20)})", text
    if len(text) > _EVIDENCE_FREE_MAX_CHARS and not claimed:
        return False, "evidence_missing", text

    # 抄弹幕：整句搬过来算不合格
    for line in evidence.danmaku or ():
        candidate = _norm(str(line))
        if len(candidate) >= _DANMAKU_COPY_MIN_CHARS and candidate in norm_text:
            return False, "copies_danmaku", text

    return True, "ok", text


# ----------------------------------------------------------------------
# 服务
# ----------------------------------------------------------------------


class CommentService:
    """把草稿生成、校验、额度与发布串起来。

    依赖注入而不是继承插件：``config_getter`` 每次现取配置，
    ``judge`` 提供 ``generate_text`` / ``persona_block``。
    """

    def __init__(
        self,
        *,
        store: Any,
        client: Any,
        judge: Any,
        config_getter: Callable[[], Any],
    ) -> None:
        self._store = store
        self._client = client
        self._judge = judge
        self._config = config_getter
        self._lock = asyncio.Lock()

    # -- 小工具 --------------------------------------------------------

    def _cfg(self) -> Any:
        return self._config().comment

    @staticmethod
    def _in_active_hours(spec: str, now: Optional[datetime] = None) -> bool:
        text = (spec or "").strip()
        if not text:
            return True
        now = now or datetime.now()
        for chunk in text.split(","):
            chunk = chunk.strip()
            if "-" not in chunk:
                continue
            try:
                start_s, end_s = chunk.split("-", 1)
                sh, sm = (int(x) for x in start_s.strip().split(":"))
                eh, em = (int(x) for x in end_s.strip().split(":"))
            except (ValueError, TypeError):
                continue
            start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
            end = now.replace(hour=eh, minute=em, second=0, microsecond=0)
            if start <= end:
                if start <= now <= end:
                    return True
            else:  # 跨午夜
                if now >= start or now <= end:
                    return True
        return False

    def _gate_reason(self) -> str:
        """返回拦截原因；空串表示可以继续。"""

        cfg = self._cfg()
        if not cfg.enabled:
            return "disabled"
        if self._store is None or self._client is None:
            return "not_loaded"
        paused = self._store.comment_pause_reason()
        if paused:
            return f"paused({paused})"
        if not self._in_active_hours(cfg.active_hours):
            return "outside_active_hours"
        if self._store.comment_count_today() >= int(cfg.daily_limit):
            return "daily_limit"
        last = self._store.last_comment_attempt_at()
        if last and time.time() - last < float(cfg.min_interval_seconds):
            return "interval"
        return ""

    def _privacy_terms(self) -> Tuple[str, ...]:
        """评论里绝不能出现的词：人设名字/别名 + 配置里手写的隐私词。"""

        terms: List[str] = []
        try:
            terms.extend(self._judge.persona_identity_terms())
        except Exception:  # noqa: BLE001 - 人设读不到就不做额外拦截
            pass
        try:
            for item in getattr(self._cfg(), "privacy_patterns", []) or []:
                value = str(item or "").strip()
                if value and value not in terms:
                    terms.append(value)
        except Exception:  # noqa: BLE001
            pass
        return tuple(terms)

    # -- 生成 ----------------------------------------------------------

    @staticmethod
    def _transient_result(score01: float = 1.0, dry_run: bool = False) -> Dict[str, Any]:
        """瞬时故障（模型掉线/超时/空输出）：不落库，下一轮还能重试。"""

        return {
            "ok": False,
            "reason": "llm_unavailable",
            "text": "",
            "evidence": [],
            "score": round(float(score01), 3),
            "dry_run": dry_run,
            "transient": True,
        }

    async def compose(
        self,
        video: Any,
        evidence: CommentEvidence,
        *,
        score01: float = 1.0,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """生成一条评论草稿并校验，不做任何发送。"""

        cfg = self._cfg()
        persona_block = ""
        try:
            persona_block = str(self._judge.persona_block() or "")
        except Exception:  # noqa: BLE001 - 人设读取失败不影响生成
            persona_block = ""
        prompt = build_comment_prompt(
            video,
            evidence,
            persona_block=persona_block,
            style_prompt=getattr(cfg, "style_prompt", ""),
            max_chars=int(cfg.max_chars),
        )
        text = await self._judge.generate_text(prompt, max_tokens=400)
        if not (text or "").strip():
            logger.warning("评论草稿生成失败：模型没有输出 %s", getattr(video, "bvid", ""))
            return self._transient_result(score01, dry_run)
        draft = parse_draft(text)
        privacy_terms = self._privacy_terms()
        ok, reason, body = (False, "unparsable", "")
        if draft is not None:
            ok, reason, body = validate_draft(
                draft,
                evidence,
                max_chars=int(cfg.max_chars),
                extra_block_words=cfg.block_patterns,
                privacy_words=privacy_terms,
            )
        if not ok:
            # 回炉一次：把不合格原因直接反馈给模型
            retry_prompt = (
                prompt
                + f"\n\n上一次的草稿不合格，原因：{reason}。请据此重写，仍然只输出 JSON。"
            )
            text = await self._judge.generate_text(retry_prompt, max_tokens=400)
            if not (text or "").strip():
                logger.warning(
                    "评论草稿重试仍无输出 %s", getattr(video, "bvid", "")
                )
                return self._transient_result(score01, dry_run)
            draft = parse_draft(text)
            if draft is not None:
                ok, reason, body = validate_draft(
                    draft,
                    evidence,
                    max_chars=int(cfg.max_chars),
                    extra_block_words=cfg.block_patterns,
                    privacy_words=privacy_terms,
                )
        result = {
            "ok": ok,
            "reason": reason,
            "text": body,
            "evidence": list((draft or {}).get("evidence") or []),
            "score": round(float(score01), 3),
            "dry_run": dry_run,
            "transient": False,
        }
        logger.info(
            "评论草稿 %s：ok=%s reason=%s len=%s",
            getattr(video, "bvid", ""),
            ok,
            reason,
            len(body or ""),
        )
        return result

    # -- 主入口 --------------------------------------------------------

    async def maybe_comment(
        self,
        video: Any,
        evidence: CommentEvidence,
        *,
        score01: float = 1.0,
        stream_id: str = "",
    ) -> Dict[str, Any]:
        """在分享前调用：满足条件就生成（并可能发布）一条评论。"""

        cfg = self._cfg()
        bvid = str(getattr(video, "bvid", "") or "")
        if not bvid:
            return {"status": "skip", "reason": "no_bvid"}
        if self._store.get_comment(bvid):
            return {"status": "skip", "reason": "already_handled"}
        gate = self._gate_reason()
        if gate:
            return {"status": "skip", "reason": gate}
        if float(score01) < float(cfg.min_interest):
            return {"status": "skip", "reason": "below_interest"}
        if int(cfg.min_evidence) > 0 and evidence.richness() < int(cfg.min_evidence):
            return {"status": "skip", "reason": "thin_evidence"}
        if getattr(cfg, "require_frames", False) and not (evidence.frames_desc or "").strip():
            return {"status": "skip", "reason": "no_frames"}

        async with self._lock:
            draft = await self.compose(video, evidence, score01=score01)
            published = bool(getattr(cfg, "publish_enabled", False))
            if not draft.get("ok"):
                if draft.get("transient"):
                    # 模型不可用不是「内容不合格」，不能写进已处理名单，
                    # 否则该视频 30 天内都不会再被评论。
                    return {"status": "skip", "reason": "llm_unavailable"}
                self._store.mark_comment_result(
                    bvid,
                    status="rejected",
                    text=str(draft.get("text") or ""),
                    reason=str(draft.get("reason") or ""),
                    stream_id=stream_id,
                    published=False,
                )
                self._store.save()
                return {"status": "rejected", "reason": draft.get("reason")}

            if not published:
                self._store.mark_comment_result(
                    bvid,
                    status="draft",
                    text=str(draft["text"]),
                    reason="publish_disabled",
                    stream_id=stream_id,
                    published=False,
                )
                self._store.save()
                return {"status": "draft", "text": draft["text"]}

            return await self._publish(video, str(draft["text"]), stream_id=stream_id)

    async def _publish(self, video: Any, text: str, *, stream_id: str = "") -> Dict[str, Any]:
        """真实发送：先落意图，再单次 POST，再按返回码分类。"""

        bvid = str(getattr(video, "bvid", "") or "")
        csrf = ""
        try:
            csrf = str(self._client.csrf_token() or "")
        except Exception:  # noqa: BLE001
            csrf = ""
        if not csrf:
            self._store.mark_comment_result(
                bvid, status="paused", text=text, reason="no_csrf",
                stream_id=stream_id, published=False,
            )
            self._store.set_comment_pause("no_csrf")
            self._store.save()
            return {"status": "paused", "reason": "no_csrf"}

        # 意图先落库，避免中断后重发
        self._store.mark_comment_attempt(bvid, text=text, aid=int(getattr(video, "aid", 0) or 0))
        self._store.save()

        try:
            payload = await self._client.post_comment(
                int(getattr(video, "aid", 0) or 0), text, csrf=csrf
            )
        except Exception as exc:  # noqa: BLE001 - 结果未知，宁可不动
            logger.warning("评论发送结果未知 %s：%s", bvid, exc)
            self._store.mark_comment_result(
                bvid, status="unknown", text=text, reason=str(exc)[:120],
                stream_id=stream_id, published=False,
            )
            self._store.set_comment_pause("send_unknown", bvid=bvid)
            self._store.save()
            return {"status": "unknown", "reason": str(exc)[:120]}

        code = int(payload.get("code") or 0)
        rpid = str(payload.get("rpid") or "")
        if code == 0:
            # code=0 就是平台已接收；rpid 缺失不影响「已发布」的判定，
            # 否则会走到下面的 code=0 分支把它记成失败，本地与线上状态相反。
            self._store.mark_comment_result(
                bvid, status="published", text=text, rpid=rpid,
                stream_id=stream_id, published=True,
            )
            self._store.save()
            if rpid:
                logger.info("评论已发布 %s rpid=%s", bvid, rpid)
            else:
                logger.warning("评论已发布但接口未返回 rpid，请人工核对 %s", bvid)
            return {"status": "published", "rpid": rpid}

        fatal = _FATAL_CODES.get(code)
        if fatal:
            self._store.mark_comment_result(
                bvid, status="paused", text=text, reason=f"{code}:{fatal}",
                stream_id=stream_id, published=False,
            )
            self._store.set_comment_pause(f"{code}:{fatal}")
            self._store.save()
            logger.warning("评论被拦截 %s code=%s %s", bvid, code, fatal)
            return {"status": "paused", "reason": fatal, "code": code}

        skip_reason = _SKIP_CODES.get(code) or f"code={code}"
        self._store.mark_comment_result(
            bvid, status="rejected", text=text, reason=skip_reason,
            stream_id=stream_id, published=False,
        )
        self._store.save()
        return {"status": "rejected", "reason": skip_reason, "code": code}

    # -- 手工命令 ------------------------------------------------------

    def blocking_reason(self) -> str:
        """当前是否有暂停（账号级或视频级）；空串表示没有。"""

        if self._store is None:
            return ""
        paused = self._store.comment_pause_reason()
        return f"paused({paused})" if paused else ""

    def pause(self, reason: str = "manual") -> None:
        if self._store is not None:
            self._store.set_comment_pause(reason)
            self._store.save()

    def resume(self) -> None:
        if self._store is not None:
            self._store.clear_comment_pause()
            self._store.save()

    def ack(self, bvid: str) -> Dict[str, Any]:
        """人工核对未知结果后解除阻塞（该视频仍不会重发）。"""

        if self._store is None:
            return {"ok": False, "reason": "not_loaded"}
        record = self._store.get_comment(bvid)
        if not record:
            return {"ok": False, "reason": "no_record"}
        self._store.mark_comment_result(
            bvid,
            status="acked_unknown",
            text=str(record.get("text") or ""),
            reason="manual_ack",
            stream_id=str(record.get("stream_id") or ""),
            published=False,
        )
        paused_reason = self._store.comment_pause_reason()
        paused_for = self._store.comment_pause_bvid()
        kept = ""
        if paused_reason:
            if paused_for and paused_for == str(bvid):
                self._store.clear_comment_pause()
            else:
                # 别的视频造成的暂停，或账号级熔断（风控/登录失效）：不许顺手解
                kept = paused_reason
        self._store.save()
        if kept:
            return {"ok": True, "bvid": bvid, "pause_kept": kept}
        return {"ok": True, "bvid": bvid}

    def status_text(self) -> str:
        if self._store is None:
            return "评论模块未加载。"
        cfg = self._cfg()
        paused = self._store.comment_pause_reason()
        last = self._store.last_comment_attempt_at()
        last_text = (
            datetime.fromtimestamp(last).strftime("%m-%d %H:%M") if last else "从未"
        )
        lines = [
            "【B 站评论状态】",
            f"开关：{'启用' if cfg.enabled else '关闭'}"
            f" / 发布：{'开' if cfg.publish_enabled else '关（只出草稿）'}",
            f"今日：{self._store.comment_count_today()}/{int(cfg.daily_limit)} 条"
            f"，最小间隔 {int(cfg.min_interval_seconds) // 60} 分钟",
            f"上次发送：{last_text}",
            f"活动时段：{cfg.active_hours or '全天'}"
            f" / 兴趣门槛：{cfg.min_interest}",
        ]
        if paused:
            lines.append(f"⚠️ 已暂停：{paused}（核对后发 /bili-comment-ack BV号）")
        recent = self._store.recent_comments(limit=3)
        if recent:
            lines.append("")
            lines.append("最近三条：")
            for item in recent:
                lines.append(
                    f"- [{item.get('status')}] {item.get('bvid')} "
                    f"{_clip(str(item.get('text') or ''), 40)}"
                )
        return "\n".join(lines)

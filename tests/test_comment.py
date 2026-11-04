"""评论模块：校验规则、额度闸门、发布分支。"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from _helpers import load_submodule


comment_mod = load_submodule("comment")
config_mod = load_submodule("config")
store_mod = load_submodule("store")

CommentEvidence = comment_mod.CommentEvidence
CommentService = comment_mod.CommentService
build_comment_prompt = comment_mod.build_comment_prompt
parse_draft = comment_mod.parse_draft
validate_draft = comment_mod.validate_draft
CommentConfig = config_mod.CommentConfig
SeenStore = store_mod.SeenStore


class FakeVideo:
    def __init__(self, bvid: str = "BV1xx411c7mD", aid: int = 123, title: str = "一个测试视频"):
        self.bvid = bvid
        self.aid = aid
        self.title = title
        self.url = f"https://www.bilibili.com/video/{bvid}"
        self.description = "这是一个用来测试的简介，讲的是某个历史人物的一生。" * 3
        self.tags = "历史,科普,人物"
        self.owner = "某个UP"
        self.partition = "人文历史"
        self.duration = 600


def _evidence() -> Any:
    return CommentEvidence(
        danmaku=["这个UP讲得还挺细的", "第一次知道还有这回事，回头再去看一遍原片", "笑死，弹幕也太整齐了"],
        comments=["看完感觉这人真有意思", "补充一句，书里写得更细"],
        subtitle="他年轻时就在实验室里折腾各种奇怪的点子，" * 10,
        frames_desc="画面是讲解型视频，主讲人坐在书架前",
        description="这是一个用来测试的简介，讲的是某个历史人物的一生。" * 3,
        tags="历史,科普,人物",
        title="一个测试视频",
    )


# ----------------------------------------------------------------------
# 校验
# ----------------------------------------------------------------------


def test_validate_accepts_grounded_comment() -> None:
    draft = {"text": "他年轻时候那股折腾劲真挺少见，看完还挺想再翻翻书。", "evidence": ["他年轻时就在实验室里折腾各种奇怪的点子"]}
    ok, reason, body = validate_draft(draft, _evidence())
    assert ok, reason
    assert body.startswith("他年轻时候")


def test_validate_rejects_link_and_mention() -> None:
    ok, reason, _ = validate_draft({"text": "看这里 https://example.com", "evidence": []}, _evidence())
    assert not ok and reason == "contains_link"
    ok, reason, _ = validate_draft({"text": "@某人 你也来看看", "evidence": []}, _evidence())
    assert not ok and reason == "contains_mention"


def test_validate_rejects_promo_and_privacy_words() -> None:
    ok, reason, _ = validate_draft({"text": "讲得真好，求个三连", "evidence": []}, _evidence())
    assert not ok and reason.startswith("blocked_words")
    ok, reason, _ = validate_draft(
        {"text": "这个我也会讲，某个私密称呼教过我", "evidence": []},
        _evidence(),
        privacy_words=("某个私密称呼",),
    )
    assert not ok and reason.startswith("privacy_words")


def test_validate_rejects_too_long() -> None:
    ok, reason, _ = validate_draft({"text": "啊" * 200, "evidence": []}, _evidence(), max_chars=160)
    assert not ok and reason.startswith("too_long")


def test_validate_requires_evidence_for_long_comment() -> None:
    body = "这条评论很长" * 15  # 超过 60 字
    ok, reason, _ = validate_draft({"text": body, "evidence": []}, _evidence())
    assert not ok and reason == "evidence_missing"
    # 短的反应用户感受，可以不带依据
    ok, reason, _ = validate_draft({"text": "看着挺有意思的", "evidence": []}, _evidence())
    assert ok, reason


def test_validate_rejects_fabricated_evidence() -> None:
    draft = {"text": "他当年在酒馆里给朋友讲了这个故事，挺有意思。", "evidence": ["酒馆里给朋友讲故事"]}
    ok, reason, _ = validate_draft(draft, _evidence())
    assert not ok and reason.startswith("evidence_mismatch")


def test_validate_rejects_danmaku_copy() -> None:
    copied = "第一次知道还有这回事，回头再去看一遍原片"
    ok, reason, _ = validate_draft({"text": copied, "evidence": []}, _evidence())
    assert not ok and reason == "copies_danmaku"


def test_parse_draft_tolerates_code_fence() -> None:
    raw = '```json\n{"text": "还行", "evidence": []}\n```'
    draft = parse_draft(raw)
    assert draft == {"text": "还行", "evidence": []}


def test_prompt_includes_style_prompt_and_persona() -> None:
    prompt = build_comment_prompt(
        FakeVideo(),
        _evidence(),
        persona_block="你的人设是一句话介绍",
        style_prompt="像个普通网友，别用感叹号",
    )
    assert "你的人设是一句话介绍" in prompt
    assert "像个普通网友" in prompt
    assert "只输出一个 JSON 对象" in prompt


# ----------------------------------------------------------------------
# 服务：闸门与发布
# ----------------------------------------------------------------------


class FakeClient:
    def __init__(self, *, code: int = 0, rpid: str = "999", raise_exc: bool = False, csrf: str = "abc"):
        self.code = code
        self.rpid = rpid
        self.raise_exc = raise_exc
        self.csrf = csrf
        self.calls: List[Dict[str, Any]] = []

    def csrf_token(self) -> str:
        return self.csrf

    async def post_comment(self, aid: int, message: str, *, csrf: str = "") -> Dict[str, Any]:
        self.calls.append({"aid": aid, "message": message, "csrf": csrf})
        if self.raise_exc:
            raise RuntimeError("网络超时")
        return {"code": self.code, "message": "", "rpid": self.rpid if self.code == 0 else ""}


class FakeJudge:
    def __init__(
        self,
        payload: Optional[Dict[str, Any]] = None,
        identity_terms: Optional[List[str]] = None,
        empty: bool = False,
    ):
        self.payload = payload if payload is not None else {"text": "看着挺有意思，讲得也细", "evidence": []}
        self.calls = 0
        self.identity_terms = list(identity_terms or [])
        self.empty = empty
        self.budgets: List[int] = []

    def persona_block(self) -> str:
        return "人设：安静的记录者"

    def persona_identity_terms(self) -> List[str]:
        return list(self.identity_terms)

    async def generate_text(self, prompt, *, task_name: str = "", max_tokens: int = 800) -> str:
        self.calls += 1
        self.budgets.append(max_tokens)
        if self.empty:
            return ""
        return json.dumps(self.payload, ensure_ascii=False)


def _service(
    tmp_path,
    *,
    cfg: CommentConfig,
    store: Optional[SeenStore] = None,
    client: Any = None,
    judge: Any = None,
) -> tuple[CommentService, Any]:
    store = store or SeenStore(tmp_path / "state.json")
    store.load()
    ns = SimpleNamespace(comment=cfg)
    service = CommentService(
        store=store,
        client=client or FakeClient(),
        judge=judge or FakeJudge(),
        config_getter=lambda: ns,
    )
    return service, store


def _cfg(**overrides: Any) -> CommentConfig:
    base = dict(enabled=True, publish_enabled=True, active_hours="", min_evidence=0)
    base.update(overrides)
    return CommentConfig(**base)


@pytest.mark.asyncio
async def test_disabled_skips_without_generating(tmp_path) -> None:
    judge = FakeJudge()
    service, _ = _service(tmp_path, cfg=_cfg(enabled=False), judge=judge)
    result = await service.maybe_comment(FakeVideo(), _evidence())
    assert result["status"] == "skip" and result["reason"] == "disabled"
    assert judge.calls == 0


@pytest.mark.asyncio
async def test_draft_mode_never_posts(tmp_path) -> None:
    client = FakeClient()
    service, store = _service(tmp_path, cfg=_cfg(publish_enabled=False), client=client)
    result = await service.maybe_comment(FakeVideo(), _evidence())
    assert result["status"] == "draft"
    assert client.calls == []
    assert store.get_comment("BV1xx411c7mD")["status"] == "draft"


@pytest.mark.asyncio
async def test_publish_success_records_rpid(tmp_path) -> None:
    client = FakeClient(code=0, rpid="12345")
    service, store = _service(tmp_path, cfg=_cfg(), client=client)
    result = await service.maybe_comment(FakeVideo(), _evidence())
    assert result["status"] == "published"
    record = store.get_comment("BV1xx411c7mD")
    assert record["rpid"] == "12345" and record["published"] is True
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_risk_control_pauses(tmp_path) -> None:
    client = FakeClient(code=-412)
    service, store = _service(tmp_path, cfg=_cfg(), client=client)
    result = await service.maybe_comment(FakeVideo(), _evidence())
    assert result["status"] == "paused"
    assert store.comment_pause_reason().startswith("-412")


@pytest.mark.asyncio
async def test_unknown_result_pauses_and_ack_clears(tmp_path) -> None:
    client = FakeClient(raise_exc=True)
    service, store = _service(tmp_path, cfg=_cfg(), client=client)
    result = await service.maybe_comment(FakeVideo(), _evidence())
    assert result["status"] == "unknown"
    assert store.comment_pause_reason() == "send_unknown"
    ack = service.ack("BV1xx411c7mD")
    assert ack["ok"] is True
    assert store.comment_pause_reason() == ""
    # 核对过的视频不会重发
    again = await service.maybe_comment(FakeVideo(), _evidence())
    assert again["status"] == "skip" and again["reason"] == "already_handled"


@pytest.mark.asyncio
async def test_daily_limit_and_interval(tmp_path) -> None:
    store = SeenStore(tmp_path / "state.json")
    store.load()
    service, _ = _service(tmp_path, cfg=_cfg(daily_limit=2, min_interval_seconds=2700), store=store)
    now = time.time()
    store.mark_comment_attempt("BV_old", text="x", now=now - 100)
    store.mark_comment_result("BV_old", status="published", text="x", rpid="1", published=True)
    result = await service.maybe_comment(FakeVideo(bvid="BV_new1"), _evidence())
    assert result["status"] == "skip" and result["reason"] == "interval"
    # 把上次发送挪到很早，额度用满
    store._meta["comment_last_attempt_at"] = now - 99999
    store._meta["comment_day_count"] = 2
    result = await service.maybe_comment(FakeVideo(bvid="BV_new2"), _evidence())
    assert result["status"] == "skip" and result["reason"] == "daily_limit"


@pytest.mark.asyncio
async def test_below_interest_and_thin_evidence(tmp_path) -> None:
    cfg = _cfg(min_interest=0.8, min_evidence=3)
    service, _ = _service(tmp_path, cfg=cfg)
    result = await service.maybe_comment(FakeVideo(), _evidence(), score01=0.5)
    assert result["reason"] == "below_interest"
    empty = CommentEvidence(description="短简介")
    result = await service.maybe_comment(FakeVideo(), empty, score01=0.9)
    assert result["reason"] == "thin_evidence"


@pytest.mark.asyncio
async def test_rejected_draft_is_not_published(tmp_path) -> None:
    client = FakeClient()
    judge = FakeJudge(payload={"text": "求个三连支持一下", "evidence": []})
    service, store = _service(tmp_path, cfg=_cfg(), client=client, judge=judge)
    result = await service.maybe_comment(FakeVideo(), _evidence())
    assert result["status"] == "rejected"
    assert client.calls == []
    assert judge.calls == 2  # 回炉了一次
    assert store.get_comment("BV1xx411c7mD")["status"] == "rejected"


def test_config_defaults_match_plan() -> None:
    cfg = CommentConfig()
    assert cfg.enabled is False
    assert cfg.publish_enabled is False
    assert cfg.daily_limit == 10
    assert cfg.min_interval_seconds == 2700
    assert cfg.style_prompt == ""


@pytest.mark.asyncio
async def test_persona_identity_terms_are_blocked(tmp_path) -> None:
    """人设名字/别名由运行时动态拦截，代码里不写死任何称呼。"""

    client = FakeClient()
    judge = FakeJudge(payload={"text": "小助手觉得这期讲得真细，回头还想再看一遍。", "evidence": []})
    judge.identity_terms = ["小助手"]
    service, _ = _service(tmp_path, cfg=_cfg(), client=client, judge=judge)
    result = await service.maybe_comment(FakeVideo(), _evidence())
    assert result["status"] == "rejected"
    assert "privacy_words" in str(result["reason"])
    assert client.calls == []


@pytest.mark.asyncio
async def test_configured_privacy_words_are_blocked(tmp_path) -> None:
    client = FakeClient()
    judge = FakeJudge(payload={"text": "这句里带了某个私密称呼，应该被拦下来。", "evidence": []})
    service, _ = _service(
        tmp_path,
        cfg=_cfg(privacy_patterns=["某个私密称呼"]),
        client=client,
        judge=judge,
    )
    result = await service.maybe_comment(FakeVideo(), _evidence())
    assert result["status"] == "rejected"
    assert client.calls == []


# ----------------------------------------------------------------------
# 回归：审阅指出的边界
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_code_zero_without_rpid_is_published(tmp_path) -> None:
    client = FakeClient(code=0, rpid="")
    service, store = _service(tmp_path, cfg=_cfg(), client=client)
    result = await service.maybe_comment(FakeVideo(), _evidence())
    assert result["status"] == "published"
    assert store.get_comment("BV1xx411c7mD")["status"] == "published"


@pytest.mark.asyncio
async def test_llm_empty_output_is_transient(tmp_path) -> None:
    """模型掉线不能被记成「内容不合格」，否则该视频 30 天内不再尝试。"""

    judge = FakeJudge(empty=True)
    service, store = _service(tmp_path, cfg=_cfg(), judge=judge)
    result = await service.maybe_comment(FakeVideo(), _evidence())
    assert result["status"] == "skip" and result["reason"] == "llm_unavailable"
    assert store.get_comment("BV1xx411c7mD") is None
    # 空输出会带着更大的预算重试一次，仍失败才判瞬时故障
    assert judge.calls == 2
    assert judge.budgets[1] > judge.budgets[0]


def test_single_char_privacy_term_only_matches_standalone() -> None:
    ok, reason, _ = validate_draft({"text": "这条鱼养得真好，钓鱼佬狂喜", "evidence": []}, _evidence(), privacy_words=("鱼",))
    assert ok, reason
    ok, reason, _ = validate_draft({"text": "鱼！好可爱", "evidence": []}, _evidence(), privacy_words=("鱼",))
    assert not ok and reason.startswith("privacy_words")
    ok, reason, _ = validate_draft({"text": "今天小鲸鱼来了", "evidence": []}, _evidence(), privacy_words=("小鲸鱼",))
    assert not ok and reason.startswith("privacy_words")


def test_short_evidence_is_checked() -> None:
    ok, reason, _ = validate_draft(
        {"text": "这期讲得真好，回头再看一遍。", "evidence": ["原神"]}, _evidence()
    )
    assert not ok and reason.startswith("evidence_mismatch")
    ok, reason, _ = validate_draft(
        {"text": "这期讲得真好，回头再看一遍。", "evidence": ["神"]}, _evidence()
    )
    assert not ok and reason.startswith("evidence_too_short")


def test_keyword_hit_respects_ascii_boundaries() -> None:
    plugin = load_submodule("plugin")
    hit = plugin._keyword_hit
    assert not hit("OP", "stop motion animation")
    assert not hit("OP", "open world exploration")
    assert hit("OP", "新番 OP 主题曲")
    assert not hit("ED", "edited highlights")
    assert hit("官方", "官方预告片")


@pytest.mark.asyncio
async def test_ack_does_not_clear_account_level_pause(tmp_path) -> None:
    """风控/登录失效是账号级熔断，ack 一条评论不能顺手解开。"""

    store = SeenStore(tmp_path / "state.json")
    store.load()
    client = FakeClient(code=-412)
    service, _ = _service(tmp_path, cfg=_cfg(), store=store, client=client)
    await service.maybe_comment(FakeVideo(), _evidence())
    assert store.comment_pause_reason()
    result = service.ack("BV1xx411c7mD")
    assert result["ok"] is True
    assert result.get("pause_kept")
    assert store.comment_pause_reason()  # 仍然暂停


@pytest.mark.asyncio
async def test_empty_then_success_recovers(tmp_path) -> None:
    """第一次被思考挤空、重试给足预算后拿到正文 → 正常产出草稿。"""

    class FlakyJudge(FakeJudge):
        async def generate_text(self, prompt, *, task_name: str = "", max_tokens: int = 800) -> str:
            self.calls += 1
            self.budgets.append(max_tokens)
            if self.calls == 1:
                return ""
            return json.dumps(self.payload, ensure_ascii=False)

    judge = FlakyJudge()
    service, store = _service(tmp_path, cfg=_cfg(publish_enabled=False), judge=judge)
    result = await service.maybe_comment(FakeVideo(), _evidence())
    assert result["status"] == "draft"
    assert judge.budgets[1] > judge.budgets[0]


def test_title_can_be_cited_as_evidence() -> None:
    """prompt 里给了标题，模型引用标题不应该被判成证据不匹配。"""

    draft = {"text": "这个测试视频讲得挺细，回头还想再看一遍。", "evidence": ["一个测试视频"]}
    ok, reason, _ = validate_draft(draft, _evidence())
    assert ok, reason

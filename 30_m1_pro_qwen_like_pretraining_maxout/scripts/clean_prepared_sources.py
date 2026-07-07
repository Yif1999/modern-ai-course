from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CURRENT_DIR / "scripts"))

from prepare_chatlm_scale_corpus import normalize_training_text  # noqa: E402
from run_data_adapter_dry_run import (  # noqa: E402
    PREPARED_SOURCE_DIR,
    chinese_ratio,
    prepared_cache_family,
)


DEFAULT_OUTPUT_BASE_DIR = CURRENT_DIR / "data" / "cache"
REPORT_DIR = CURRENT_DIR / "outputs" / "reports"
STATUS_DIR = CURRENT_DIR / "outputs" / "status"

DOC_SEPARATOR = "<|doc_sep|>"
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
HTML_TAG_RE = re.compile(r"<[^>\n]{1,120}>")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
SHORTLINK_RE = re.compile(r"https?://t\.cn/[A-Za-z0-9]+", re.IGNORECASE)
HORIZONTAL_RULE_RE = re.compile(r"(?m)^[ \t]*(?:[-_=*~#][ \t]*){8,}$")
INLINE_HORIZONTAL_RULE_RE = re.compile(r"(?:-{8,}|={8,}|_{8,}|~{8,}|\*{8,}|#{8,})")
MENTION_RE = re.compile(r"@[^\s，。；:：,]+")
SOCIAL_GIVEAWAY_RE = re.compile(r"抽奖|转评赞|转发抽奖|关注.{0,20}转发|应援|大礼包|领取地点|开奖日期|中奖名单")
SOCIAL_COMMERCE_RE = re.compile(
    r"购买地址|截止时间|找其他商品|拍[:：]|原[:：]\d|后[:：]\d|包邮|顺丰到付|"
    r"转发.{0,20}抽\d*人|抽\d*人.{0,20}送出|送出.{0,20}(?:件|个|份)|"
    r"抢￥|券后|下单[:：]"
)
PUBLIC_NOTICE_SHORTLINK_RE = re.compile(
    r"(?:教育局|招生|录取|考生|志愿|中考|高考|报名|资格审查|考试成绩|学籍|政务|公告|通知)"
)
MEDIA_RE = re.compile(r"\.(?:jpg|jpeg|png|gif|webp|mp4|avi|mov)\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?86[- ]?)?1[3-9]\d{9}|(?:0\d{2,3}[- ]?)?\d{7,8}")
CONTACT_RE = re.compile(
    r"客服|加微信|微信号|VX[:：]|QQ[:：]|\bLINE\b|下[订单]|扫码|二维码|"
    r"商铺首页|黄页介绍|立即购买|点击下载|免费咨询|限时领取|领券|券后|"
    r"限时特惠|商务合作|新闻爆料",
    re.IGNORECASE,
)
WEAK_CONTACT_RE = re.compile(
    r"咨询热线|联系电话|联系方式|医院地址",
    re.IGNORECASE,
)
DIRECT_CONTACT_RE = re.compile(
    r"加微信|微信号[:：]?\s*[A-Za-z0-9_-]{3,}|VX[:：]\s*[A-Za-z0-9_-]{3,}|"
    r"QQ[:：]\s*\d{5,}|\bLINE\b|客服(?:微信|QQ|热线|电话)",
    re.IGNORECASE,
)
PROMO_CONTACT_CONTEXT_RE = re.compile(
    r"公司注册|代理记账|培训|课程|报名|招商|加盟|优惠|领取|"
    r"下订单|立即购买|免费咨询|限时领取|领券|券后|限时特惠|"
    r"扫码(?:报名|领取|购买|下载|咨询|加群|加微信)|"
    r"二维码(?:报名|领取|购买|下载|咨询|加群|加微信)|"
    r"下载地址|点击下载|商铺首页|黄页介绍",
    re.IGNORECASE,
)
NEGATED_PROMO_CONTEXT_RE = re.compile(
    r"(?:没有|并无|不含|不是|并非|未提供).{0,24}"
    r"(?:购买入口|客服微信|限时优惠|招商加盟|联系方式|联系电话|免费咨询|扫码报名|"
    r"领取资料|兼职接单|日结|赚钱|广告|商业推广)"
)
GAMBLING_RE = re.compile(
    r"时时彩|重庆时时彩|北京赛车|六合彩|腾讯分分彩|幸运飞艇|大发快3|"
    r"三分彩|分分彩|北京赛车|彩票平台|购彩平台|私彩|网赌|"
    r"投注平台|胆码|杀号|计划群|精准计划|"
    r"快乐购彩|购彩.{0,12}(?:平台|网站|软件|入口|注册|推荐|预测|中奖)|"
    r"(?:体彩|足彩|赛事|竞彩).{0,20}(?:投注|推荐|预测|胆码|赔率)|"
    r"(?:投注).{0,20}(?:体彩|足彩|赛事|竞彩)|"
    r"博彩.{0,12}(?:平台|网站|投注|下注|现金网|注册|彩金|赔率)|(?:平台|网站).{0,12}博彩|"
    r"(?:彩票|博彩|投注平台|赌场|娱乐城|六合彩|开奖).{0,12}赔率|"
    r"(?:彩票|博彩|六合彩|开奖).{0,12}特码|特码图|特码为|管家婆.{0,12}特码|"
    r"欲钱买|杀肖|杀尾|"
    r"彩票论坛|彩票网|网彩票|彩票注册|彩票代理|彩票(?:平台|软件|官网|预测|计划)|"
    r"太子彩票|天天彩票|报码|六肖|平特一肖|"
    r"(?:开奖现场|开奖结果|开奖).{0,8}(?:报码|开码)|"
    r"(?:真人娱乐|娱乐城|电游).{0,16}(?:注册|官网|平台|博彩|彩票|下注|赔率|百家乐|老虎机|赌场|澳门)|"
    r"(?:娱乐城|赌场).{0,12}(?:首存|存款|彩金|优惠)|"
    r"(?:彩票|体彩|福彩).{0,16}(?:赔率|精准计划|软件|平台|预测|计划|论坛|代理|官网)",
)
SEO_RE = re.compile(
    r"免责声明|相关阅读|上一篇|下一篇|网站地图|ICP备案|版权所有|本文链接|"
    r"未经授权|点击查看|更多相关|为您推荐|相关推荐|原文地址|"
    r"当前位置[:：]|返回首页|收藏本站",
)
COMMERCIAL_PROMO_RE = re.compile(
    r"本文为商业推广|广告内容提供方|长按保存下方图片|多劳多得.{0,20}(?:日结|兼职|接单|赚钱|佣金)|"
    r"日结[/／]天|兼职接单|足不出户就能赚钱|扫码报名|"
    r"官方线上教育|解释权归广告内容提供方|招商加盟|加盟热线|"
    r"付款后.{0,20}微信|领取资料"
)
BUSINESS_SERVICE_RE = re.compile(
    r"公司注册|注册地址|地址托管|代理记账|工商注册|税务注销|营业执照|"
    r"基本户|做账报税|记账公司|税务师事务所"
)
PRODUCT_SEO_RE = re.compile(
    r"货真价实|厂家|批发|报价|价格|施工|安装|定制|多少钱|哪家好|"
    r"供应|采购|客户案例|服务热线|咨询热线|联系电话|联系方式"
)
MEDICAL_PROVIDER_PROMO_RE = re.compile(
    r"(?:医院|医疗机构|门诊|专科).{0,80}"
    r"(?:规模最大|一流(?:的)?(?:检验中心|诊疗设备|专家团队|服务理念)|"
    r"最受欢迎|最值得信赖|首屈一指|专业治疗|特色医院|诊疗设备)"
)
MEDICAL_PROVIDER_PROMO_TERM_RE = re.compile(
    r"规模最大|一流(?:的)?(?:检验中心|诊疗设备|专家团队|服务理念)|"
    r"最受欢迎|最值得信赖|首屈一指|专业治疗|特色医院|诊疗设备|专家团队|服务理念"
)
STRONG_MEDICAL_PROVIDER_PROMO_RE = re.compile(r"规模最大|最受欢迎|最值得信赖|首屈一指|专业治疗|特色医院")
MEDICAL_PRODUCT_PROMO_RE = re.compile(
    r"(?:修复液|脉活修复液|瘢痕疙瘩治疗费用|静脉曲张中医疗法|祛疤).{0,260}"
    r"(?:研制成功|解决.{0,30}难题|想了解更多|跟着我阅读本文|专家介绍|彻底去掉|见解哦)"
)
MEDICAL_SERVICE_PROMO_RE = re.compile(
    r"(?:隐形矫正|正畸|牙套|口腔修复|齿科医院|牙医|成人正畸|龅牙|整牙).{0,160}"
    r"(?:哪里好|推荐|问我|联系|平台|医生口碑|全国.{0,20}医生)"
)
NEGATED_MEDICAL_PROMO_CONTEXT_RE = re.compile(
    r"(?:没有|并无|不含|不是|并非|未|避免|防止).{0,24}"
    r"(?:宣传|推荐|招揽|推广|广告|治疗服务|诊疗设备|专家团队|产品|医院)"
)
CRACKED_SOFTWARE_RE = re.compile(
    r"(?:QQExplorer|密码破解|密码暴力破解器|暴力破解器|破解工具|破解版下载|"
    r"下载.{0,24}破解版|注册码生成器|激活码生成器).{0,220}"
    r"(?:下载|破解器|找回密码|强行|安装|使用步骤|代理服务器)"
)
NEGATED_CRACKED_SOFTWARE_RE = re.compile(
    r"(?:不要|禁止|避免|不应|不能|请勿).{0,24}(?:使用|下载|安装).{0,24}"
    r"(?:破解器|破解版|破解工具|注册码|激活码|暴力破解)|"
    r"(?:不是|并非|不属于|没有|未提供).{0,24}(?:破解器|破解版下载|破解工具).{0,24}"
    r"(?:下载|安装|页面|步骤|页)?"
)
DOWNLOAD_SITE_RE = re.compile(
    r"软件分类|游戏大小|软件大小|更新时间|版本号|运行环境|下载地址|"
    r"高速下载|立即下载|点击下载|游戏下载|软件介绍|游戏介绍|"
    r"游戏截图|安装教程|绿色版|安卓版|电脑版|配置要求|解锁秘籍"
)
CONTENT_FARM_FOOTER_RE = re.compile(
    r"联盟百科是组织像一个百科全书|联盟百科不受维基媒体基金会|"
    r"Google Play徽标|隐私政策"
)
EXAM_SEO_TEMPLATE_RE = re.compile(r"欢迎光临[“\"]?.{3,140}(?:[”\"])?[^。！？]{0,80}如有问题请及时联系我")
HELP_PHRASE_RE = re.compile(r"可以帮助|能够帮助|帮助玩家|玩家可以帮助")
LEGAL_ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百千万0-9]+条")
COURT_DOC_TITLE_RE = re.compile(
    r"(?:人民法院.{0,80})?(?:民\s*事|刑\s*事|行\s*政)?\s*(?:判\s*决\s*书|裁\s*定\s*书|调\s*解\s*书)"
)
NEGATED_COURT_DOC_CONTEXT_RE = re.compile(r"(?:不是|并非|不属于).{0,16}(?:判决书|裁定书|调解书|诉讼文书)")
MOJIBAKE_RE = re.compile(
    r"学\?；|枷肫|窝[\ue000-\uf8ff]|銆€|濞变箰|鍦埚|鏄熸|暣瀹|鏂伴椈|鍙戜綔|镄勫|"
    r"(?:鐨|闂|鍙|杩|绋)[\u4e00-\u9fff]{0,8}(?:鐨|闂|鍙|杩|绋)|"
    r"\?{1,4}(?:侵鳎|矍椴|壅|舻|幕翱|悸|鞘|柩|栏腥|瘟|髁|橥|枰)|"
    r"(?:侵鳎|矍椴|壅|舻|幕翱|悸|鞘|柩|栏腥|瘟|髁|橥|枰)\?{1,4}"
)

TOPIC_SOUP_GROUPS = {
    "finance": ["A股", "新股", "股票", "基金", "净利润", "申购", "中签", "股价", "上市公司"],
    "entertainment": ["明星", "导演", "电影", "电视剧", "综艺", "票房", "演员", "娱乐圈"],
    "sports": ["中超", "女排", "火箭", "勇士", "足球", "篮球", "比赛", "世界杯"],
    "military": ["歼-", "美军", "军方", "导弹", "战机", "航母", "国防"],
    "medical": ["医院", "医生", "癌症", "抗生素", "高血压", "糖尿病", "肺部", "感染"],
    "travel": ["景区", "旅游", "游客", "旅行社", "酒店", "出境游"],
    "tech": ["互联网", "芯片", "5G", "人工智能", "智能", "手机", "运营商"],
    "education": ["大学", "学生", "教师", "考试", "学校", "课程"],
}

Matcher = Callable[[str, "SourceContext", dict[str, float], int], bool]


@dataclass(frozen=True)
class SourceContext:
    source_name: str
    source_type: str
    source_key: str
    source_kind: str

    @classmethod
    def from_row(cls, *, source_name: str, source_type: str) -> "SourceContext":
        source_key = f"{source_name} {source_type}".lower()
        return cls(
            source_name=source_name,
            source_type=source_type,
            source_key=source_key,
            source_kind=infer_source_kind(source_key),
        )


@dataclass(frozen=True)
class Rule:
    name: str
    severity: str
    scope: tuple[str, ...]
    description: str
    false_positive_risk: str
    matcher: Matcher

    def applies_to(self, ctx: SourceContext) -> bool:
        return not self.scope or ctx.source_kind in self.scope


@dataclass(frozen=True)
class RuleHit:
    name: str
    severity: str


@dataclass(frozen=True)
class Classification:
    decision: str
    flags: list[str]
    stats: dict[str, float]
    source_kind: str
    rule_hits: list[RuleHit]


class CleanPolicy:
    def decide(self, hits: list[RuleHit], ctx: SourceContext) -> str:
        if any(hit.severity == "drop" for hit in hits):
            return "drop"
        if any(hit.severity == "quarantine" for hit in hits):
            return "quarantine"
        return "keep"


POLICY = CleanPolicy()


def infer_source_kind(source_key: str) -> str:
    if "lccc" in source_key:
        return "dialogue_short"
    if any(token in source_key for token in ["dialogue", "roleplay", "game_world_chat", "hana"]):
        return "dialogue"
    if "wikipedia" in source_key or "wiki" in source_key:
        return "wiki"
    if "comment" in source_key:
        return "comment"
    if any(token in source_key for token in ["qa", "question", "answer", "belle", "zhihu"]):
        return "qa"
    if any(
        token in source_key
        for token in [
            "general_web_backbone",
            "streamed_general",
            "cci3",
            "wudao",
            "fineweb",
            "skypile",
            "chinesewebtext",
            "webtext",
        ]
    ):
        return "general_web"
    return "unknown"


def min_chars_for_source(ctx: SourceContext, default_min_chars: int) -> int:
    if ctx.source_kind in {"dialogue_short", "dialogue", "qa", "wiki"}:
        return 8
    return default_min_chars


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def compact_hash(text: str) -> str:
    compact = "".join(text.split())
    return hashlib.sha1(compact.encode("utf-8", errors="ignore")).hexdigest()


def short_text(text: str, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit] + "..."


def normalize_clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("$P$", "\n")
    text = re.sub(r"(?:\$\s*P\s*\$)+", "\n", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = ZERO_WIDTH_RE.sub("", text)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) not in {"Cc", "Cf", "Co"})
    return normalize_training_text(text)


def char_stats(text: str) -> dict[str, float]:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return {
            "length": 0,
            "chinese_ratio": 0.0,
            "punct_ratio": 0.0,
            "private_ratio": 0.0,
            "replacement_ratio": 0.0,
            "digit_ratio": 0.0,
            "latin_ratio": 0.0,
        }
    total = len(chars)
    return {
        "length": float(total),
        "chinese_ratio": sum(1 for ch in chars if "\u4e00" <= ch <= "\u9fff") / total,
        "punct_ratio": sum(1 for ch in chars if unicodedata.category(ch).startswith("P")) / total,
        "private_ratio": sum(1 for ch in chars if unicodedata.category(ch) == "Co") / total,
        "replacement_ratio": text.count("\ufffd") / total,
        "digit_ratio": sum(1 for ch in chars if ch.isdigit()) / total,
        "latin_ratio": sum(1 for ch in chars if ("A" <= ch <= "Z") or ("a" <= ch <= "z")) / total,
    }


def has_repeated_span(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 240:
        return False
    spans = [compact[i : i + 80] for i in range(0, len(compact) - 80, 40)]
    spans = [span for span in spans if len(span) >= 60]
    return len(spans) - len(set(spans)) >= 2


def low_diversity(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 500:
        return False
    unique_ratio = len(set(compact)) / max(len(compact), 1)
    char_counts = Counter(compact)
    most_common_ratio = char_counts.most_common(1)[0][1] / max(len(compact), 1)
    return unique_ratio < 0.03 or (unique_ratio < 0.05 and most_common_ratio > 0.18)


def mixed_topic_soup(text: str) -> bool:
    if len(text) < 700:
        return False
    hits = 0
    for words in TOPIC_SOUP_GROUPS.values():
        if any(word in text for word in words):
            hits += 1
    if hits < 5:
        return False
    sentence_count = len(re.findall(r"[。！？!?]", text))
    comma_count = text.count("，") + text.count(",")
    return sentence_count >= 8 or comma_count >= 30


def commercial_promo_template(text: str) -> bool:
    has_negated_promo_context = NEGATED_PROMO_CONTEXT_RE.search(text) is not None
    if COMMERCIAL_PROMO_RE.search(text) and not has_negated_promo_context:
        return True
    if (
        re.search(r"报名方式.{0,30}(?:微信|二维码|扫码|联系电话|咨询|缴费|课程|培训|学历提升)", text)
        and not has_negated_promo_context
    ):
        return True
    business_hits = len(BUSINESS_SERVICE_RE.findall(text))
    business_terms = set(BUSINESS_SERVICE_RE.findall(text))
    if business_hits >= 5 and len(business_terms) >= 3:
        return True
    if business_hits >= 3 and len(business_terms) >= 2 and re.search(
        r"可为您|本人|经验丰富|建[帐账]|记[帐账]|报税|年检|纳税筹划", text
    ):
        return True
    product_hits = len(PRODUCT_SEO_RE.findall(text))
    product_terms = set(PRODUCT_SEO_RE.findall(text))
    product_promo_terms = {
        term
        for term in product_terms
        if term
        in {
            "货真价实",
            "厂家",
            "批发",
            "客户案例",
            "服务热线",
            "咨询热线",
            "联系电话",
            "联系方式",
            "哪家好",
            "定制",
        }
    }
    product_sales_template_terms = product_promo_terms & {
        "货真价实",
        "哪家好",
        "定制",
        "客户案例",
        "服务热线",
        "咨询热线",
        "联系电话",
        "联系方式",
    }
    strong_contact = DIRECT_CONTACT_RE.search(text) is not None or re.search(
        r"立即购买|免费咨询|限时领取|领券|券后|限时特惠|下订单",
        text,
        re.IGNORECASE,
    ) is not None
    if product_hits >= 7 and len(product_terms) >= 3 and strong_contact and len(product_promo_terms) >= 2:
        return True
    if product_hits >= 6 and len(product_promo_terms) >= 2 and product_sales_template_terms and re.search(
        r"(?:货真价实|哪家好|厂家|批发|定制).{0,30}"
        r"(?:货真价实|哪家好|厂家|批发|定制)",
        text,
    ):
        return True
    return False


def medical_promo_template(text: str) -> bool:
    if NEGATED_MEDICAL_PROMO_CONTEXT_RE.search(text):
        return False
    provider_match = MEDICAL_PROVIDER_PROMO_RE.search(text)
    if provider_match and re.search(r"治疗|疾病|诊疗|专家|患者|健康", text):
        provider_window = provider_match.group(0)
        promo_terms = set(MEDICAL_PROVIDER_PROMO_TERM_RE.findall(provider_window))
        if STRONG_MEDICAL_PROVIDER_PROMO_RE.search(provider_window) or len(promo_terms) >= 2:
            return True
    if MEDICAL_PRODUCT_PROMO_RE.search(text):
        return True
    if MEDICAL_SERVICE_PROMO_RE.search(text):
        return True
    return False


def cracked_software_template(text: str) -> bool:
    if NEGATED_CRACKED_SOFTWARE_RE.search(text):
        return False
    if CRACKED_SOFTWARE_RE.search(text):
        return True
    return False


def download_site_template(text: str) -> bool:
    download_hits = len(DOWNLOAD_SITE_RE.findall(text))
    if download_hits >= 5:
        return True
    if download_hits >= 3 and re.search(r"下载地址|点击下载|游戏下载|软件下载|客户端下载地址|app下载地址", text, re.IGNORECASE):
        return True
    if download_hits >= 3 and ("下载" in text or "游戏" in text) and text.count("可以帮助") >= 4:
        return True
    return False


def machine_translation_phrase_repetition(text: str) -> bool:
    helper_hits = len(HELP_PHRASE_RE.findall(text))
    if helper_hits < 14:
        return False
    density = helper_hits / max(len(text) / 1000, 1)
    return density >= 3.0


def html_markup_noise(text: str) -> bool:
    if not HTML_TAG_RE.search(text):
        return False
    tags = HTML_TAG_RE.findall(text)
    return len(tags) >= 3


def ad_contact_template(text: str) -> bool:
    contact_hits = len(CONTACT_RE.findall(text))
    weak_contact_hits = len(WEAK_CONTACT_RE.findall(text))
    has_phone = PHONE_RE.search(text) is not None
    has_direct_contact = DIRECT_CONTACT_RE.search(text) is not None
    has_negated_promo_context = NEGATED_PROMO_CONTEXT_RE.search(text) is not None
    has_promo_context = (
        (PROMO_CONTACT_CONTEXT_RE.search(text) is not None and not has_negated_promo_context)
        or commercial_promo_template(text)
        or download_site_template(text)
    )
    has_phone_call_to_action = has_phone and re.search(
        r"联系电话|联系方式|咨询热线|服务热线|报名|免费咨询|扫码报名|加微信|QQ[:：]",
        text,
        re.IGNORECASE,
    )
    if has_direct_contact and has_promo_context:
        return True
    if has_phone_call_to_action and has_promo_context:
        return True
    if contact_hits >= 2 and has_promo_context and (has_phone_call_to_action or URL_RE.search(text)):
        return True
    if weak_contact_hits >= 2 and has_promo_context and len(SEO_RE.findall(text)) >= 2:
        return True
    return False


def dense_social_shortlinks(text: str) -> bool:
    shortlink_hits = len(SHORTLINK_RE.findall(text))
    mention_hits = len(MENTION_RE.findall(text))
    has_positive_giveaway = False
    for match in SOCIAL_GIVEAWAY_RE.finditer(text):
        prefix = text[max(0, match.start() - 12) : match.start()]
        if not any(neg in prefix for neg in ["并非", "不是", "非", "无"]):
            has_positive_giveaway = True
            break
    has_social_commerce = SOCIAL_COMMERCE_RE.search(text) is not None
    if shortlink_hits >= 2 and not has_positive_giveaway and not has_social_commerce and mention_hits == 0:
        if PUBLIC_NOTICE_SHORTLINK_RE.search(text):
            return False
    if shortlink_hits >= 2 and has_positive_giveaway:
        return True
    if shortlink_hits >= 2 and has_social_commerce:
        return True
    if shortlink_hits >= 3 and mention_hits >= 2 and len(text) < 1200:
        return True
    if shortlink_hits >= 5 and len(text) < 800:
        return True
    return False


def exam_keyword_stuffing(text: str) -> bool:
    if EXAM_SEO_TEMPLATE_RE.search(text):
        return True
    question_match = re.search(r"^(.{4,40}?[？?。]|.{4,40}?（\）)", text)
    if not question_match:
        return False
    question = re.sub(r"\s+", "", question_match.group(1))
    if len(question) < 6:
        return False
    compact = re.sub(r"\s+", "", text)
    repeats = compact.count(question)
    return repeats >= 3 and ("欢迎光临" in text or "试题" in text or "答案" in text)


def legal_template_dense(text: str) -> bool:
    if NEGATED_COURT_DOC_CONTEXT_RE.search(text):
        return False
    head = text[:260]
    title_match = COURT_DOC_TITLE_RE.search(head)
    if title_match and ("人民法院" in head or re.search(r"民\s*事|刑\s*事|行\s*政", title_match.group(0))):
        return True
    court_role_hits = sum(
        text.count(term)
        for term in [
            "上诉人",
            "被上诉人",
            "原告",
            "被告",
            "再审申请人",
            "委托诉讼代理人",
            "法定代表人",
            "公诉",
        ]
    )
    formal_court_markers = sum(
        text.count(term)
        for term in [
            "上诉人",
            "被上诉人",
            "再审申请人",
            "委托诉讼代理人",
            "法定代表人",
            "审理终结",
        ]
    )
    has_court_context = any(term in text for term in ["人民法院", "本院", "检察署", "检察院", "审理终结"])
    if formal_court_markers >= 3 and court_role_hits >= 6 and has_court_context:
        return True
    if text.count("被告") >= 5 and ("公诉" in text or "检察" in text) and has_court_context:
        return True
    article_count = len(LEGAL_ARTICLE_RE.findall(text))
    return article_count >= 16 and formal_court_markers >= 2 and has_court_context


def punctuation_dense(text: str, ctx: SourceContext, stats: dict[str, float]) -> bool:
    if stats["length"] < 80 or stats["punct_ratio"] <= 0.35:
        return False
    if ctx.source_kind in {"qa", "dialogue"} and (
        "摩斯电码" in text
        or "摩尔斯电码" in text
        or "Morse" in text
        or re.search(r"[.\-]{2,}(?:\s+[.\-]{1,}){3,}", text)
        or re.search(r"\|[^|\n]{1,80}\|", text)
        or re.search(r"```|class\s+\w+|def\s+\w+|function\s+\w+|=>|</?\w+", text)
    ):
        return False
    text_without_rules = HORIZONTAL_RULE_RE.sub(" ", text)
    text_without_rules = INLINE_HORIZONTAL_RULE_RE.sub(" ", text_without_rules)
    adjusted_stats = char_stats(text_without_rules)
    return adjusted_stats["length"] >= 80 and adjusted_stats["punct_ratio"] > 0.35


RULES: tuple[Rule, ...] = (
    Rule(
        name="too_short",
        severity="drop",
        scope=(),
        description="Text is below the source-aware minimum length.",
        false_positive_risk="medium",
        matcher=lambda text, ctx, stats, min_chars: int(stats["length"]) < min_chars_for_source(ctx, min_chars),
    ),
    Rule(
        name="short_social_dialogue",
        severity="flag",
        scope=("dialogue_short",),
        description="Short LCCC-style social chat; useful but should be mix-controlled later.",
        false_positive_risk="low",
        matcher=lambda text, ctx, stats, min_chars: int(stats["length"]) < 80,
    ),
    Rule(
        name="low_chinese_ratio_hard",
        severity="drop",
        scope=(),
        description="Very low Chinese character ratio for a Chinese pretraining corpus.",
        false_positive_risk="low",
        matcher=lambda text, ctx, stats, min_chars: stats["chinese_ratio"] < 0.15,
    ),
    Rule(
        name="low_chinese_ratio_soft",
        severity="flag",
        scope=(),
        description="Moderately low Chinese character ratio; retained for later weighting/audit.",
        false_positive_risk="medium",
        matcher=lambda text, ctx, stats, min_chars: 0.15 <= stats["chinese_ratio"] < 0.35,
    ),
    Rule(
        name="broken_encoding",
        severity="quarantine",
        scope=(),
        description="Replacement characters or dense private-use characters indicate broken decoding.",
        false_positive_risk="high",
        matcher=lambda text, ctx, stats, min_chars: text.count("\ufffd") >= 20
        or (text.count("\ufffd") >= 5 and stats["replacement_ratio"] > 0.03)
        or stats["private_ratio"] > 0.03,
    ),
    Rule(
        name="mojibake_pattern",
        severity="quarantine",
        scope=(),
        description="Known mojibake fragments from bad transcoding.",
        false_positive_risk="high",
        matcher=lambda text, ctx, stats, min_chars: MOJIBAKE_RE.search(text) is not None,
    ),
    Rule(
        name="url",
        severity="flag",
        scope=(),
        description="Contains URL-like text; retained unless another rule escalates it.",
        false_positive_risk="low",
        matcher=lambda text, ctx, stats, min_chars: URL_RE.search(text) is not None,
    ),
    Rule(
        name="dense_social_shortlinks",
        severity="quarantine",
        scope=(),
        description="Dense t.cn shortlink and social-account list, usually giveaway/forwarding noise.",
        false_positive_risk="medium",
        matcher=lambda text, ctx, stats, min_chars: dense_social_shortlinks(text),
    ),
    Rule(
        name="exam_keyword_stuffing",
        severity="quarantine",
        scope=("general_web", "qa", "unknown"),
        description="Question-bank SEO page with repeated prompt keywords and boilerplate.",
        false_positive_risk="medium",
        matcher=lambda text, ctx, stats, min_chars: exam_keyword_stuffing(text),
    ),
    Rule(
        name="many_media_filenames",
        severity="quarantine",
        scope=(),
        description="Dense media filenames usually indicate scraped galleries, markdown dumps, or course listings.",
        false_positive_risk="medium",
        matcher=lambda text, ctx, stats, min_chars: len(MEDIA_RE.findall(text)) >= 5,
    ),
    Rule(
        name="markup_noise",
        severity="quarantine",
        scope=(),
        description="HTML/XML markup appears repeatedly in the training text.",
        false_positive_risk="medium",
        matcher=lambda text, ctx, stats, min_chars: html_markup_noise(text),
    ),
    Rule(
        name="long_repeated_char",
        severity="flag",
        scope=(),
        description="A single character repeats at least 10 times; retained because comments/dialogue often use this naturally.",
        false_positive_risk="high",
        matcher=lambda text, ctx, stats, min_chars: re.search(r"(.)\1{9,}", text) is not None,
    ),
    Rule(
        name="repeated_span",
        severity="quarantine",
        scope=(),
        description="Repeated long spans indicate duplicated boilerplate or broken extraction.",
        false_positive_risk="high",
        matcher=lambda text, ctx, stats, min_chars: has_repeated_span(text),
    ),
    Rule(
        name="low_char_diversity",
        severity="quarantine",
        scope=(),
        description="Very low unique-character diversity in long text.",
        false_positive_risk="high",
        matcher=lambda text, ctx, stats, min_chars: low_diversity(text),
    ),
    Rule(
        name="gambling_spam",
        severity="quarantine",
        scope=(),
        description="Lottery/gambling spam, betting keywords, or casino promotion.",
        false_positive_risk="high",
        matcher=lambda text, ctx, stats, min_chars: GAMBLING_RE.search(text) is not None,
    ),
    Rule(
        name="mixed_topic_soup",
        severity="quarantine",
        scope=("general_web",),
        description="CCI3-style stitched multi-topic fragments.",
        false_positive_risk="high",
        matcher=lambda text, ctx, stats, min_chars: "cci3" in ctx.source_key and mixed_topic_soup(text),
    ),
    Rule(
        name="ad_contact_template",
        severity="flag",
        scope=(),
        description="Contact-heavy promotional or service-template text; retained unless another stronger rule escalates it.",
        false_positive_risk="high",
        matcher=lambda text, ctx, stats, min_chars: ad_contact_template(text),
    ),
    Rule(
        name="seo_boilerplate_dense",
        severity="quarantine",
        scope=("general_web", "qa", "wiki", "unknown"),
        description="Dense SEO/footer boilerplate.",
        false_positive_risk="medium",
        matcher=lambda text, ctx, stats, min_chars: len(SEO_RE.findall(text)) >= 3,
    ),
    Rule(
        name="commercial_promo_template",
        severity="quarantine",
        scope=("general_web", "qa", "comment", "unknown"),
        description="Commercial promotion, agency registration, paid course, or supplier template.",
        false_positive_risk="high",
        matcher=lambda text, ctx, stats, min_chars: commercial_promo_template(text),
    ),
    Rule(
        name="medical_promo_template",
        severity="quarantine",
        scope=("general_web", "unknown"),
        description="Medical provider or remedy advertorial with strong promotional claims.",
        false_positive_risk="high",
        matcher=lambda text, ctx, stats, min_chars: medical_promo_template(text),
    ),
    Rule(
        name="download_site_template",
        severity="quarantine",
        scope=("general_web", "unknown"),
        description="Software/game download page or download-directory boilerplate.",
        false_positive_risk="medium",
        matcher=lambda text, ctx, stats, min_chars: download_site_template(text),
    ),
    Rule(
        name="cracked_software_template",
        severity="quarantine",
        scope=("general_web", "qa", "unknown"),
        description="Cracked software, password cracker, registration-code, or similar download page.",
        false_positive_risk="medium",
        matcher=lambda text, ctx, stats, min_chars: cracked_software_template(text),
    ),
    Rule(
        name="content_farm_footer",
        severity="quarantine",
        scope=("general_web", "wiki", "unknown"),
        description="Known content-farm or scraped encyclopedia footer.",
        false_positive_risk="low",
        matcher=lambda text, ctx, stats, min_chars: CONTENT_FARM_FOOTER_RE.search(text) is not None,
    ),
    Rule(
        name="machine_translation_phrase_repetition",
        severity="quarantine",
        scope=("general_web", "qa", "wiki", "unknown"),
        description="High-density repeated helper phrases typical of machine-translated pages.",
        false_positive_risk="medium",
        matcher=lambda text, ctx, stats, min_chars: machine_translation_phrase_repetition(text),
    ),
    Rule(
        name="legal_template_dense",
        severity="quarantine",
        scope=("general_web", "unknown"),
        description="Dense legal/court-document template.",
        false_positive_risk="medium",
        matcher=lambda text, ctx, stats, min_chars: legal_template_dense(text),
    ),
    Rule(
        name="punctuation_dense",
        severity="quarantine",
        scope=(),
        description="Punctuation ratio is unusually high.",
        false_positive_risk="medium",
        matcher=lambda text, ctx, stats, min_chars: punctuation_dense(text, ctx, stats),
    ),
    Rule(
        name="many_short_docs_structural",
        severity="drop",
        scope=(),
        description="Many embedded document separators indicate structural concatenation.",
        false_positive_risk="low",
        matcher=lambda text, ctx, stats, min_chars: text.count(DOC_SEPARATOR) >= 8,
    ),
)


def rule_registry_metadata() -> list[dict[str, Any]]:
    return [
        {
            "name": rule.name,
            "severity": rule.severity,
            "scope": list(rule.scope) if rule.scope else ["all"],
            "false_positive_risk": rule.false_positive_risk,
            "description": rule.description,
        }
        for rule in RULES
    ]


def classify_text(
    text: str,
    *,
    min_chars: int,
    source_name: str = "",
    source_type: str = "",
) -> Classification:
    stats = char_stats(text)
    ctx = SourceContext.from_row(source_name=source_name, source_type=source_type)
    hits: list[RuleHit] = []
    for rule in RULES:
        if not rule.applies_to(ctx):
            continue
        if rule.matcher(text, ctx, stats, min_chars):
            hits.append(RuleHit(name=rule.name, severity=rule.severity))
    flags = [hit.name for hit in hits]
    decision = POLICY.decide(hits, ctx)
    return Classification(
        decision=decision,
        flags=flags,
        stats=stats,
        source_kind=ctx.source_kind,
        rule_hits=hits,
    )


def load_first_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    value = json.loads(line)
                    return value if isinstance(value, dict) else {}
    except Exception:
        return {}
    return {}


def choose_input_files(input_dir: Path, *, dedupe_families: bool) -> list[Path]:
    files = sorted(path for path in input_dir.glob("*.jsonl") if path.stat().st_size > 0)
    if not dedupe_families:
        return files
    by_family: dict[str, Path] = {}
    for path in files:
        family = prepared_cache_family(path)
        current = by_family.get(family)
        if current is None or path.stat().st_size > current.stat().st_size:
            by_family[family] = path
    return [by_family[key] for key in sorted(by_family)]


def preload_hashes(output_dir: Path, *, skip_names: set[str]) -> set[str]:
    hashes: set[str] = set()
    if not output_dir.exists():
        return hashes
    for path in sorted(output_dir.glob("*.jsonl")):
        if path.name in skip_names:
            continue
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                h = str(row.get("clean_text_hash") or "").strip()
                if not h:
                    h = compact_hash(str(row.get("text", "")))
                hashes.add(h)
    return hashes


def example_add(examples: dict[str, deque], flag: str, row: dict[str, Any], limit: int) -> None:
    if len(examples[flag]) < limit:
        examples[flag].append(row)


def clean_file(
    path: Path,
    *,
    output_path: Path | None,
    quarantine_path: Path | None,
    max_docs: int | None,
    start_line: int,
    max_input_lines: int | None,
    min_chars: int,
    clean_version: str,
    global_hashes: set[str],
    example_limit: int,
    status_interval: int,
    append_output: bool,
) -> dict[str, Any]:
    started = time.time()
    counters = Counter()
    flag_counts = Counter()
    kept_chars = 0
    raw_chars = 0
    source_names = Counter()
    source_types = Counter()
    examples: dict[str, deque] = defaultdict(lambda: deque(maxlen=example_limit))
    kept_examples: list[dict[str, Any]] = []
    write_handle = None
    quarantine_handle = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_handle = output_path.open("a" if append_output else "w", encoding="utf-8")
    if quarantine_path is not None:
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        quarantine_handle = quarantine_path.open("a" if append_output else "w", encoding="utf-8")

    def maybe_print_status() -> None:
        if status_interval > 0 and counters["seen"] % status_interval == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"clean source={path.name} seen={counters['seen']:,} kept={counters['kept']:,} "
                f"quarantine={counters['quarantined']:,} drop={counters['dropped']:,} "
                f"dup={counters['duplicates']:,} docs/s={counters['seen'] / elapsed:.1f}",
                flush=True,
            )

    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            stop_before_line = start_line + max_input_lines if max_input_lines is not None else None
            for line_number, line in enumerate(f, start=1):
                if line_number < start_line:
                    continue
                if stop_before_line is not None and line_number >= stop_before_line:
                    break
                if max_docs is not None and counters["seen"] >= max_docs:
                    break
                if not line.strip():
                    continue
                counters["seen"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    counters["dropped"] += 1
                    flag_counts["json_decode_error"] += 1
                    maybe_print_status()
                    continue
                text = normalize_clean_text(str(row.get("text", "")))
                raw_chars += len(text)
                source_name = str(row.get("source_name") or path.stem)
                source_type = str(row.get("source_type") or row.get("category") or "")
                classification = classify_text(
                    text,
                    min_chars=min_chars,
                    source_name=source_name,
                    source_type=source_type,
                )
                flags = classification.flags
                stats = classification.stats
                if classification.decision != "keep":
                    counters["quarantined" if classification.decision == "quarantine" else "dropped"] += 1
                    for flag in flags or ["dropped_unknown"]:
                        flag_counts[flag] += 1
                        example_add(
                            examples,
                            flag,
                            {
                                "source_name": source_name,
                                "source_type": source_type,
                                "source_id": row.get("source_id") or row.get("id"),
                                "decision": classification.decision,
                                "source_kind": classification.source_kind,
                                "flags": flags,
                                "stats": {k: round(v, 4) for k, v in stats.items()},
                                "text": short_text(text),
                            },
                            example_limit,
                        )
                    if classification.decision == "quarantine" and quarantine_handle is not None:
                        q_row = dict(row)
                        q_row["text"] = text
                        q_row["char_count"] = len(text)
                        q_row["chinese_ratio"] = chinese_ratio(text)
                        q_row["clean_decision"] = "quarantine"
                        q_row["clean_flags"] = flags
                        q_row["clean_text_hash"] = compact_hash(text)
                        q_row["clean_version"] = clean_version
                        q_row["source_cache_path"] = str(path)
                        quarantine_handle.write(json.dumps(q_row, ensure_ascii=False, sort_keys=True) + "\n")
                    maybe_print_status()
                    continue

                h = compact_hash(text)
                if h in global_hashes:
                    counters["duplicates"] += 1
                    flag_counts["duplicate_compact_hash"] += 1
                    maybe_print_status()
                    continue
                global_hashes.add(h)

                row["text"] = text
                row["char_count"] = len(text)
                row["chinese_ratio"] = chinese_ratio(text)
                row["clean_flags"] = flags
                row["clean_decision"] = "keep"
                row["clean_text_hash"] = h
                row["clean_version"] = clean_version
                row["source_cache_path"] = str(path)
                if write_handle is not None:
                    write_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                counters["kept"] += 1
                kept_chars += len(text)
                source_names[source_name] += 1
                source_types[source_type or "unknown"] += 1
                for flag in flags:
                    flag_counts[f"kept_with_{flag}"] += 1
                if len(kept_examples) < example_limit:
                    kept_examples.append(
                        {
                            "source_name": source_name,
                            "source_type": source_type,
                            "char_count": len(text),
                            "text": short_text(text),
                        }
                    )

                maybe_print_status()
    finally:
        if write_handle is not None:
            write_handle.close()
        if quarantine_handle is not None:
            quarantine_handle.close()

    elapsed = max(time.time() - started, 1e-6)
    return {
        "path": str(path),
        "output_path": str(output_path) if output_path else None,
        "start_line": start_line,
        "max_input_lines": max_input_lines,
        "append_output": append_output,
        "seen": int(counters["seen"]),
        "kept": int(counters["kept"]),
        "dropped": int(counters["dropped"]),
        "quarantined": int(counters["quarantined"]),
        "duplicates": int(counters["duplicates"]),
        "keep_rate": counters["kept"] / counters["seen"] if counters["seen"] else 0,
        "raw_chars": raw_chars,
        "kept_chars": kept_chars,
        "chars_keep_rate": kept_chars / raw_chars if raw_chars else 0,
        "source_names": dict(source_names.most_common(10)),
        "source_types": dict(source_types.most_common(10)),
        "flag_counts": dict(flag_counts.most_common()),
        "examples": {flag: list(rows) for flag, rows in examples.items()},
        "kept_examples": kept_examples,
        "elapsed_sec": elapsed,
        "docs_per_sec": counters["seen"] / elapsed,
    }


def render_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        f"# {report['output_version']} Clean Prepared Sources Report",
        "",
        f"- input dir: `{report['input_dir']}`",
        f"- output dir: `{report['output_dir']}`",
        f"- dry run: `{report['dry_run']}`",
        f"- files: `{report['files_processed']}`",
        f"- docs seen: `{report['totals']['seen']:,}`",
        f"- docs kept: `{report['totals']['kept']:,}` ({report['totals']['keep_rate']:.2%})",
        f"- docs dropped: `{report['totals']['dropped']:,}`",
        f"- docs quarantined: `{report['totals'].get('quarantined', 0):,}`",
        f"- duplicates: `{report['totals']['duplicates']:,}`",
        f"- kept chars: `{report['totals']['kept_chars']:,}`",
        "",
        "## Drop / Flag Counts",
        "",
        "| flag | count | share of seen |",
        "|---|---:|---:|",
    ]
    seen = max(int(report["totals"]["seen"]), 1)
    for flag, count in report["flag_counts"].items():
        lines.append(f"| `{flag}` | {count:,} | {count / seen:.2%} |")

    lines.extend(
        [
            "",
            "## Per Source File",
            "",
            "| file | seen | kept | keep rate | dropped | quarantined | duplicates | kept chars |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["sources"]:
        lines.append(
            f"| `{Path(item['path']).name}` | {item['seen']:,} | {item['kept']:,} | "
            f"{item['keep_rate']:.2%} | {item['dropped']:,} | {item.get('quarantined', 0):,} | "
            f"{item['duplicates']:,} | {item['kept_chars']:,} |"
        )

    lines.extend(["", "## Examples", ""])
    for flag, rows in report["examples"].items():
        lines.extend([f"### {flag}", ""])
        for row in rows[:5]:
            lines.append(
                f"- decision=`{row.get('decision')}` source=`{row.get('source_name')}` "
                f"kind=`{row.get('source_kind')}` flags=`{row.get('flags')}`: {row.get('text')}"
            )
        lines.append("")
    lines.extend(["", "## Rule Registry", "", "| rule | severity | scope | risk | description |", "|---|---|---|---|---|"])
    for rule in report.get("rules", []):
        lines.append(
            f"| `{rule['name']}` | `{rule['severity']}` | `{', '.join(rule['scope'])}` | "
            f"`{rule['false_positive_risk']}` | {rule['description']} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean local prepared source JSONL caches without re-downloading datasets.")
    parser.add_argument("--input-dir", default=str(PREPARED_SOURCE_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-version", default="prepared_sources_clean_v2")
    parser.add_argument("--clean-version", default=None)
    parser.add_argument("--max-docs-per-source", type=int, default=None)
    parser.add_argument(
        "--start-line",
        type=int,
        default=1,
        help="1-based physical input line to start from; useful for chunked long-file cleans.",
    )
    parser.add_argument(
        "--max-input-lines-per-source",
        type=int,
        default=None,
        help="Maximum physical input lines to scan per selected source file.",
    )
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--source", action="append", default=[], help="Only process files whose name/family/source matches this string.")
    parser.add_argument("--no-dedupe-families", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-existing-output-dir",
        action="store_true",
        help="Allow writing into a non-empty output directory. Intended for resumable/chunked cleans.",
    )
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Append keep/quarantine rows instead of truncating output files.",
    )
    parser.add_argument("--no-write-quarantine", action="store_true")
    parser.add_argument(
        "--preload-hashes-from-output-dir",
        action="store_true",
        help="When continuing a partial clean, load hashes from existing output JSONL files not being regenerated.",
    )
    parser.add_argument("--example-limit", type=int, default=8)
    parser.add_argument("--status-interval-docs", type=int, default=100_000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_BASE_DIR / args.output_version
    if args.start_line < 1:
        raise ValueError("--start-line must be >= 1")
    if args.max_input_lines_per_source is not None and args.max_input_lines_per_source < 1:
        raise ValueError("--max-input-lines-per-source must be >= 1")
    if args.append_output and args.force:
        raise ValueError("--append-output and --force are intentionally incompatible")
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not args.force
        and not args.dry_run
        and not args.allow_existing_output_dir
    ):
        raise FileExistsError(
            f"{output_dir} already exists and is not empty. "
            "Use --allow-existing-output-dir for resumable/chunked writes, --force, or choose a new --output-dir."
        )
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    files = choose_input_files(input_dir, dedupe_families=not args.no_dedupe_families)
    if args.source:
        needles = [item.lower() for item in args.source]
        selected = []
        for path in files:
            first = load_first_json(path)
            haystack = " ".join(
                [
                    path.name,
                    prepared_cache_family(path),
                    str(first.get("source_name") or ""),
                    str(first.get("source_type") or first.get("category") or ""),
                ]
            ).lower()
            if any(needle in haystack for needle in needles):
                selected.append(path)
        files = selected
    if not files:
        raise RuntimeError("no input files matched")

    selected_output_names = {path.name for path in files}
    preload_skip_names = set() if args.append_output else selected_output_names
    global_hashes: set[str] = (
        preload_hashes(output_dir, skip_names=preload_skip_names)
        if args.preload_hashes_from_output_dir
        else set()
    )
    if global_hashes:
        print(f"preloaded hashes from existing clean outputs: {len(global_hashes):,}", flush=True)
    source_reports = []
    totals = Counter()
    flag_counts = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    started = time.time()
    status_path = STATUS_DIR / f"{args.output_version}_clean_status.json"

    for index, path in enumerate(files, start=1):
        output_path = None if args.dry_run else output_dir / path.name
        quarantine_path = None
        if not args.dry_run and not args.no_write_quarantine:
            quarantine_path = output_dir / "_quarantine" / path.name
        if output_path is not None and args.force and output_path.exists():
            output_path.unlink()
        if quarantine_path is not None and args.force and quarantine_path.exists():
            quarantine_path.unlink()
        print(f"cleaning [{index}/{len(files)}] {path.name}", flush=True)
        item = clean_file(
            path,
            output_path=output_path,
            quarantine_path=quarantine_path,
            max_docs=args.max_docs_per_source,
            start_line=args.start_line,
            max_input_lines=args.max_input_lines_per_source,
            min_chars=args.min_chars,
            clean_version=args.clean_version or args.output_version,
            global_hashes=global_hashes,
            example_limit=args.example_limit,
            status_interval=args.status_interval_docs,
            append_output=args.append_output,
        )
        source_reports.append(item)
        for key in ["seen", "kept", "dropped", "quarantined", "duplicates", "raw_chars", "kept_chars"]:
            totals[key] += int(item[key])
        flag_counts.update(item["flag_counts"])
        for flag, rows in item["examples"].items():
            if len(examples[flag]) < args.example_limit:
                examples[flag].extend(rows[: max(0, args.example_limit - len(examples[flag]))])
        write_json(
            status_path,
            {
                "state": "running",
                "output_version": args.output_version,
                "current_file": path.name,
                "files_processed": index,
                "files_total": len(files),
                "totals": dict(totals),
                "updated_at_unix": time.time(),
            },
        )

    totals["keep_rate"] = totals["kept"] / totals["seen"] if totals["seen"] else 0
    totals["chars_keep_rate"] = totals["kept_chars"] / totals["raw_chars"] if totals["raw_chars"] else 0
    report = {
        "output_version": args.output_version,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "dry_run": args.dry_run,
        "dedupe_families": not args.no_dedupe_families,
        "max_docs_per_source": args.max_docs_per_source,
        "start_line": args.start_line,
        "max_input_lines_per_source": args.max_input_lines_per_source,
        "append_output": args.append_output,
        "min_chars": args.min_chars,
        "write_quarantine": not args.no_write_quarantine,
        "files_processed": len(files),
        "totals": dict(totals),
        "flag_counts": dict(flag_counts.most_common()),
        "rules": rule_registry_metadata(),
        "sources": source_reports,
        "examples": dict(examples),
        "elapsed_sec": time.time() - started,
    }
    json_path = REPORT_DIR / f"{args.output_version}_clean_report.json"
    md_path = REPORT_DIR / f"{args.output_version}_clean_report.md"
    write_json(json_path, report)
    render_report(report, md_path)
    write_json(status_path, {"state": "complete", **report, "report_path": str(json_path), "markdown_path": str(md_path)})
    print("report:", json_path)
    print("markdown:", md_path)
    print("status:", status_path)
    print(
        f"seen={totals['seen']:,} kept={totals['kept']:,} keep_rate={totals['keep_rate']:.2%} "
        f"dropped={totals['dropped']:,} quarantined={totals['quarantined']:,} duplicates={totals['duplicates']:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.clean_prepared_sources import classify_text, clean_file, normalize_clean_text, preload_hashes  # noqa: E402


class CleanPreparedSourcesGoldenTest(unittest.TestCase):
    def assert_classification(
        self,
        *,
        text: str,
        source_name: str,
        source_type: str,
        decision: str,
        flags: set[str] | None = None,
        absent_flags: set[str] | None = None,
    ) -> None:
        result = classify_text(normalize_clean_text(text), min_chars=20, source_name=source_name, source_type=source_type)
        self.assertEqual(result.decision, decision)
        if flags:
            self.assertTrue(flags.issubset(set(result.flags)), result.flags)
        if absent_flags:
            self.assertTrue(set(result.flags).isdisjoint(absent_flags), result.flags)

    def test_keep_short_lccc_dialogue(self) -> None:
        self.assert_classification(
            text="甲：好赞\n乙：你看了？甲：看了",
            source_name="silver/lccc/lccc_large.jsonl.gz",
            source_type="lccc_social_dialogue",
            decision="keep",
            flags={"short_social_dialogue"},
            absent_flags={"too_short"},
        )

    def test_keep_short_wikipedia_entry(self) -> None:
        self.assert_classification(
            text="崇德路 崇德路可以指：一条位于市区的道路。",
            source_name="Chinese Wikipedia",
            source_type="encyclopedia",
            decision="keep",
            absent_flags={"too_short"},
        )

    def test_keep_normal_business_news(self) -> None:
        self.assert_classification(
            text=(
                "侯孝海卸任，华润雪花啤酒发生法定代表人变更。公司表示，"
                "本次调整属于治理架构优化，后续仍将推进啤酒和白酒两大事业部建设。"
                "公开资料显示，新任负责人具有多年消费品行业管理经验。"
            ),
            source_name="Morton-Li/ChineseWebText2.0-HighQuality",
            source_type="streamed_general",
            decision="keep",
            absent_flags={"commercial_promo_template", "ad_contact_template"},
        )

    def test_keep_normal_product_review(self) -> None:
        self.assert_classification(
            text=(
                "这款车型的售价区间为十九万元到二十八万元，外观采用封闭式前脸，"
                "内饰保留实体按键，空间表现适合家庭通勤。文章主要比较续航、底盘和安全配置，"
                "并没有提供购买链接或联系方式。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"commercial_promo_template"},
        )

    def test_keep_price_heavy_business_analysis(self) -> None:
        self.assert_classification(
            text=(
                "瑞幸咖啡的创业案例主要讨论价格策略和品牌定位。文章反复分析价格，"
                "包括咖啡价格、低价竞争、品牌溢价、购买便利性和扫码免费品尝活动，"
                "但它没有导流入口或服务销售话术。这类文本属于正常商业分析，"
                "适合作为通用网页语料保留。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"commercial_promo_template"},
        )

    def test_keep_payment_safety_news(self) -> None:
        self.assert_classification(
            text=(
                "微信转账时出现风险提醒，千万不要急着付款。警方表示，"
                "类似案例多与冒充熟人、虚假交易有关，用户在支付宝转账或微信转账前"
                "应核对收款方身份。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"commercial_promo_template"},
        )

    def test_keep_price_discussion_without_sales_template(self) -> None:
        self.assert_classification(
            text=(
                "商业插画报价要结合画师专业能力、项目周期、版权范围和商家预期收益。"
                "文章讨论价格如何形成，也分析多少钱才合理，但没有购买入口、客服微信、"
                "限时优惠或招商加盟信息。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"commercial_promo_template"},
        )

    def test_keep_industry_analysis_with_manufacturer_and_wholesale_terms(self) -> None:
        self.assert_classification(
            text=(
                "中国汽车行业的产销数据长期存在统计口径差异。文章反复提到厂家、"
                "经销商、厂家批发给经销商的数据、真实上牌数据和广告流量数据，"
                "讨论的是行业分析和数据治理，不是面向消费者的销售广告。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"commercial_promo_template"},
        )

    def test_keep_business_registration_qa_without_service_pitch(self) -> None:
        self.assert_classification(
            text=(
                "一个公司能注册几个网站备案？有人回答说，公司名下可以备案多个域名，"
                "如果营业执照发生变更，备案号也可能不同。建议到工信部网站查询备案信息。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"commercial_promo_template"},
        )

    def test_keep_enterprise_software_article_with_procurement_terms(self) -> None:
        self.assert_classification(
            text=(
                "SAP自学指南介绍案例公司的产品数据管理、生产计划和供应链计划。"
                "文章讨论采购流程、供应数据、库存积压和订单核算，也留下作者QQ:2651000673，"
                "但主体是ERP实施知识，不是商品促销模板。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"commercial_promo_template", "ad_contact_template"},
        )

    def test_keep_crime_news_with_purchase_terms(self) -> None:
        self.assert_classification(
            text=(
                "顺藤摸瓜，网购农药牵出造假窝案。杨某在微信公众号看到报道后，"
                "想起自己曾向网店询问价格并下单购买农药，随后通过新闻链接"
                "http://t.cn/AiExample 发现发货地与假农药生产地点相同，于是报警。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            flags={"url"},
            absent_flags={"commercial_promo_template", "ad_contact_template"},
        )

    def test_keep_public_recruitment_notice(self) -> None:
        self.assert_classification(
            text=(
                "甘肃省兰州实验小学2023年公开招聘事业编制教师公告。"
                "根据事业单位公开招聘有关规定，现面向高校毕业生招聘教学人员6名。"
                "各岗位招聘基本情况及要求见附件，报名时间和资格审查安排以公告为准。"
                "资格审查及笔试、面试均不收取任何费用，也不委托任何机构进行辅导培训。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"commercial_promo_template", "ad_contact_template"},
        )

    def test_keep_public_notice_with_contact_as_soft_flag(self) -> None:
        self.assert_classification(
            text=(
                "公开招聘工作人员公告发布后，考生可按要求提交报名材料。"
                "报名咨询电话：13812345678，资格审查和面试安排以官方网站通知为准。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            flags={"ad_contact_template"},
            absent_flags={"commercial_promo_template"},
        )

    def test_keep_wiki_legal_explanation_not_court_template(self) -> None:
        self.assert_classification(
            text=(
                "幼女是一种年龄段或身体形态的描述语。条目在法律解释中提到，"
                "刑法对侵害未成年人的犯罪有从重处罚规定，也会讨论第十七条等法条背景，"
                "但这是百科说明，不是法院判决书或裁定书模板。"
            ),
            source_name="acg_wiki",
            source_type="wiki",
            decision="keep",
            absent_flags={"legal_template_dense"},
        )

    def test_keep_legal_news_not_court_template(self) -> None:
        self.assert_classification(
            text=(
                "铁路刺警案宣判后引发讨论。报道介绍，法院引用刑法第十九条和"
                "第八十七条说明监护处分依据，检察官随后提出上诉。文章主要是"
                "新闻报道和法律背景解释，不是判决书或裁定书正文。"
            ),
            source_name="Skywork/SkyPile-150B",
            source_type="streamed_general",
            decision="keep",
            absent_flags={"legal_template_dense"},
        )

    def test_keep_legal_qa_not_court_template(self) -> None:
        self.assert_classification(
            text=(
                "2000年江苏省婚假为多少天？回答介绍全国性规定婚假是三天，"
                "并引用江苏省人口与计划生育条例第三十条说明晚婚婚假的变化。"
                "这是法律问答，不包含原告、被告、上诉人等诉讼文书结构。"
            ),
            source_name="opencsg/Fineweb-Edu-Chinese-V2.1",
            source_type="streamed_general",
            decision="keep",
            absent_flags={"legal_template_dense"},
        )

    def test_quarantine_court_judgement_template(self) -> None:
        self.assert_classification(
            text=(
                "辽宁省葫芦岛市中级人民法院民事裁定书，上诉人某村全体村民，"
                "被上诉人某公司，法定代表人赵某。原审原告与原审被告排除妨害纠纷一案，"
                "本院依法组成合议庭进行了审理。上诉人、被上诉人、原告、被告、"
                "委托诉讼代理人等诉讼角色反复出现，现已审理终结。"
            ),
            source_name="Morton-Li/ChineseWebText2.0-HighQuality",
            source_type="streamed_general",
            decision="quarantine",
            flags={"legal_template_dense"},
        )

    def test_keep_normal_tax_policy_news(self) -> None:
        self.assert_classification(
            text=(
                "国家陆续出台一系列税收优惠政策，生态税收体系覆盖资源开采、生产、流通、消费、"
                "排放等环节。税务部门落实减税降费政策，助力企业调整优化产业结构，"
                "企业所得税和增值税政策变化缓解了企业资金压力。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"commercial_promo_template", "ad_contact_template"},
        )

    def test_keep_article_with_single_url(self) -> None:
        self.assert_classification(
            text=(
                "国家卫健委介绍疫情防控政策调整情况，强调当前工作的重点是保健康、防重症。"
                "发布会全文可见 http://t.cn/A6Kmrtf0，文章主体是正常新闻报道。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            flags={"url"},
            absent_flags={"dense_social_shortlinks"},
        )

    def test_quarantine_dense_social_shortlinks(self) -> None:
        self.assert_classification(
            text=(
                "1 @重生爆炸丸_ 447 http://t.cn/A6aWwgAw "
                "2 @小丸是你爹爹639 http://t.cn/A6aWwgA7 "
                "3 @蕊蕊要早睡_ 618 http://t.cn/A6aWwgAh "
                "4 @时迩啊十二w 533 http://t.cn/A6aWwgAP "
                "5 @汪汪小分队队长541 http://t.cn/A6aWwgAz"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"dense_social_shortlinks"},
        )

    def test_quarantine_social_giveaway_shortlinks(self) -> None:
        self.assert_classification(
            text=(
                "第六日跨年应援大礼包，领取地点为场馆门口。"
                "抽奖要求：关注两位老师，转评赞抽奖。"
                "开奖日期：http://t.cn/AiF2zR5j。本次应援已备案。"
                "http://t.cn/AiFAD09Y http://t.cn/AiFAkAq4"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"dense_social_shortlinks"},
        )

    def test_quarantine_social_commerce_shortlinks(self) -> None:
        self.assert_classification(
            text=(
                "手指滑板第二批来了，RMB65每个。购买地址http://t.cn/AiezDOrK，"
                "截止时间http://t.cn/AiezsASk，全国包邮。转发本条微博12月30日抽3人送出指板1件。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"dense_social_shortlinks"},
        )

    def test_keep_public_education_notice_with_many_shortlinks(self) -> None:
        self.assert_classification(
            text=(
                "2022年西安市中考录取分数线公布，城六区普通高中录取最低控制线523分。"
                "考生可登录西安市教育局官网查询成绩，也可到毕业学校查询。"
                "网上填报志愿时间为http://t.cn/A6aOJOPI，五区二县时间为http://t.cn/A6aOJOPe。"
                "未被录取的考生请留意http://t.cn/A6aOJOhI，http://t.cn/A6aOJOhE，"
                "http://t.cn/A6aOJO7x，http://t.cn/A6aOJO7d。请家长不要听信违规招生宣传。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            flags={"url"},
            absent_flags={"dense_social_shortlinks"},
        )

    def test_quarantine_social_commerce_shortlinks_even_with_many_links(self) -> None:
        self.assert_classification(
            text=(
                "天猫精选好物合集，前10分钟第2件5折，抢￥90券后下单。"
                "http://t.cn/Aa000001 http://t.cn/Aa000002 http://t.cn/Aa000003 "
                "http://t.cn/Aa000004 http://t.cn/Aa000005，服装美食母婴数码每日更新。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"dense_social_shortlinks"},
        )

    def test_keep_long_article_with_multiple_links(self) -> None:
        self.assert_classification(
            text=(
                "研究人员介绍，小麦根系中的真菌可以提高关键营养元素吸收，"
                "相关结果发表于学术期刊。资料链接见 http://t.cn/Ai1e7vbM，"
                "项目介绍见 http://t.cn/Ai1e7vbN，数据说明见 http://t.cn/Ai1e7vbP。"
                "文章主体讨论农业减肥增效和气候变化背景下的作物管理，并非商品推广或抽奖。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            flags={"url"},
            absent_flags={"dense_social_shortlinks"},
        )

    def test_keep_news_mentioning_wechat_and_complaint_phone(self) -> None:
        self.assert_classification(
            text=(
                "内蒙古的李霞通过某微信号的集赞活动获得免费游香港的机会，"
                "但这次出游让她大呼上当。记者致电广东省旅游局，工作人员称，"
                "目前已接到很多投诉电话，提醒旅游者不要相信不实宣传。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"ad_contact_template"},
        )

    def test_keep_product_news_with_scan_qr_terms(self) -> None:
        self.assert_classification(
            text=(
                "摩拜单车宣布免扫码解锁功能正式上线。官方称，该功能能提高用户在"
                "遇到无码无号车辆时的开锁便捷性，同时有助于减少二维码被破坏、"
                "车辆被私占等问题。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"ad_contact_template"},
        )

    def test_quarantine_direct_contact_ad(self) -> None:
        self.assert_classification(
            text=(
                "学历提升课程限时优惠，扫码报名即可领取资料。"
                "加微信 abc123 免费咨询，联系电话13812345678，名额有限。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"ad_contact_template"},
        )

    def test_quarantine_product_seo_ad(self) -> None:
        self.assert_classification(
            text=(
                "厂家批发定制环保袋，货真价实，价格优惠。"
                "想知道哪家好可立即购买或免费咨询客服，服务热线13812345678，"
                "支持采购供应，客户案例丰富。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"commercial_promo_template"},
        )

    def test_quarantine_exam_keyword_stuffing(self) -> None:
        self.assert_classification(
            text=(
                "封闭式询问常用的是（）。欢迎光临“封闭式询问常用的是（）。”，"
                "如有问题请及时联系我。封闭式询问常用的是试题答案解析，"
                "封闭式询问常用的是相关知识点，封闭式询问常用的是题库页面。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"exam_keyword_stuffing"},
        )

    def test_quarantine_exam_keyword_stuffing_missing_quote(self) -> None:
        self.assert_classification(
            text=(
                "MS图上出现m/e 74的强峰，IR光谱在3400cm有一宽峰，"
                "欢迎光临“MS图上出现m/e 74的强峰，IR光谱在3400cm如有问题请及时联系我。"
                "MS图上出现m/e 74的强峰题库答案解析和相关试题。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"exam_keyword_stuffing"},
        )

    def test_keep_private_use_spacing_after_normalization(self) -> None:
        self.assert_classification(
            text=(
                "南充市党政代表团赴我市开展主题教育暨区域协同发展交流活动。"
                "本报讯，双方举行座谈会并签署合作协议，推动区域协同发展。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"broken_encoding"},
        )

    def test_keep_sparse_replacement_character(self) -> None:
        self.assert_classification(
            text=(
                "春歌是动画作品中的登场角色。她喜欢舞蹈，并且精通日本古典文化，"
                "为了成为一名真正的淑女，她从小练习礼仪。文本中偶尔出现一个�字符，"
                "不应该导致整篇正常条目被直接删除。"
            ),
            source_name="acg_wiki",
            source_type="wiki",
            decision="keep",
            absent_flags={"broken_encoding"},
        )

    def test_keep_short_qa_item(self) -> None:
        self.assert_classification(
            text="用户：果实 : 树木 :: 子女 : ? 助手：父母",
            source_name="BELLE",
            source_type="qa",
            decision="keep",
            absent_flags={"too_short"},
        )

    def test_quarantine_gambling_spam(self) -> None:
        self.assert_classification(
            text="天天彩票平台提供 时时彩 开奖结果 下注 赔率 计划群 精准预测，注册送彩金。",
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"gambling_spam"},
        )

    def test_quarantine_hongkong_special_code_chart_spam(self) -> None:
        self.assert_classification(
            text=(
                "香港管家婆特码图更新，开奖直播和特码图资料每日整理，"
                "平台提供号码预测、开奖结果查询和相关论坛入口。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"gambling_spam"},
        )

    def test_quarantine_sports_lottery_tip(self) -> None:
        self.assert_classification(
            text=(
                "长崎航海对阵新泻天鹅赛事前瞻，快乐购彩，理性投注，"
                "中国体育彩票提醒广大球迷注意控制购彩金额，本文提供竞彩推荐。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"gambling_spam"},
        )

    def test_quarantine_special_code_riddle_spam(self) -> None:
        self.assert_classification(
            text=(
                "特码为单数，欲钱买呼风唤雨得意忘形的动物。"
                "传真之杀肖为兔，传真之杀尾为八，二十八码一玄机。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"gambling_spam"},
        )

    def test_keep_colloquial_special_code_homophone(self) -> None:
        self.assert_classification(
            text=(
                "这特码放在现代，什么仿古街道都比不上。人物一边走一边吐槽，"
                "觉得路边的青砖黑瓦很有年代感。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"gambling_spam"},
        )

    def test_quarantine_casino_deposit_spam(self) -> None:
        self.assert_classification(
            text="捷豹娱乐城首存优惠活动，存款送彩金，百家乐和老虎机平台入口。",
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"gambling_spam"},
        )

    def test_quarantine_mixed_topic_soup(self) -> None:
        self.assert_classification(
            text=(
                "银行在技术革新浪潮中冲在第一线。明星导演的新电影票房继续攀升。"
                "中超球队完成换人，篮球比赛进入最后阶段。医院医生讨论感染治疗方案。"
                "景区游客数量增加，旅行社推出酒店套餐。大学教师公布考试课程安排。"
                "芯片和人工智能产业继续发展，5G运营商推出新服务。"
                "A股基金净利润和新股申购数据也在同一页面出现。"
            )
            * 5,
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"mixed_topic_soup"},
        )

    def test_quarantine_mojibake(self) -> None:
        self.assert_classification(
            text="銆€銆€濞变箰鍦埚コ鏄熸暣瀹圭殑鏂伴椈，鍙戜綔镄勫.",
            source_name="p208p2002/wudao",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"mojibake_pattern"},
        )

    def test_keep_repeated_comment_or_expressive_text(self) -> None:
        self.assert_classification(
            text="蛇山的现货来了！！！！！！！！！！！！！！！！可动人偶玩家的进阶路线。",
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            flags={"long_repeated_char"},
        )

    def test_quarantine_business_service_ad(self) -> None:
        self.assert_classification(
            text=(
                "青羊区公司注册地址变更可以进行地址托管吗？我们提供公司注册、工商注册、"
                "代理记账、税务注销、营业执照办理、基本户开户、做账报税等服务，"
                "经验丰富，可为您建账记账报税。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"commercial_promo_template"},
        )

    def test_quarantine_download_site(self) -> None:
        self.assert_classification(
            text=(
                "软件分类：赛车竞速。游戏大小：2GB。更新时间：2024年。"
                "运行环境：Windows。下载地址如下，点击下载即可安装，配置要求见页面说明。"
            ),
            source_name="Morton-Li/ChineseWebText2.0-HighQuality",
            source_type="streamed_general",
            decision="quarantine",
            flags={"download_site_template"},
        )

    def test_quarantine_cracked_password_software_page(self) -> None:
        self.assert_classification(
            text=(
                "QQ密码破解QQExplorer下载v1.26破解版，这是一款针对忘记QQ密码或QQ被盗"
                "情况下进行分析破解的密码暴力破解器。使用QQExplorer破解版就可以强行"
                "找回密码，页面提供下载和安装说明。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"cracked_software_template"},
        )

    def test_keep_security_warning_about_cracked_software(self) -> None:
        self.assert_classification(
            text=(
                "安全部门提醒，不要下载和使用所谓密码破解器、破解版软件或破解工具。"
                "这类程序可能携带木马，用户应通过官方渠道找回账号密码。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"cracked_software_template"},
        )

    def test_keep_online_library_resource_discussion_not_cracked_software(self) -> None:
        self.assert_classification(
            text=(
                "有哪些比较全的在线古籍查找网站？回答介绍国家数字图书馆、方志数据库"
                "和在线阅读资源，也提到有些资料可以下载或注册账号后阅读。"
                "这是资源介绍，不是破解工具下载页。"
            ),
            source_name="Zhihu-KOL",
            source_type="qa",
            decision="keep",
            absent_flags={"cracked_software_template"},
        )

    def test_keep_game_console_modding_discussion_not_cracked_software_download(self) -> None:
        self.assert_classification(
            text=(
                "Switch 是买卡带还是数字版？回答提到有些玩家会讨论破解、软破或硬破，"
                "但重点是卡带、数字版、联网和账号差异，没有提供破解工具下载或安装步骤。"
            ),
            source_name="Zhihu-KOL",
            source_type="qa",
            decision="keep",
            absent_flags={"cracked_software_template"},
        )

    def test_keep_official_registration_activation_code_not_cracked_software(self) -> None:
        self.assert_classification(
            text=(
                "中国商标网注册网上用户后，系统邮件会提醒审核结果，并告知PIN码和激活码。"
                "用户点击证书下载完成官方账号开通流程。"
            ),
            source_name="opencsg/Fineweb-Edu-Chinese-V2.1",
            source_type="streamed_general",
            decision="keep",
            absent_flags={"cracked_software_template"},
        )

    def test_keep_morse_code_qa_not_punctuation_noise(self) -> None:
        self.assert_classification(
            text=(
                "用户：将以下文本转化为摩斯电码：“Hello World!” Hello World! "
                "助手：.... . .-.. .-.. --- / .-- --- .-. .-.. -.. -.-.-- "
                "用户：把“.... . .-.. .-.. --- / .-- --- .-. .-.. -.. -.-.--”翻译成英文。"
                "助手：这就是摩斯电码中的“Hello World！”"
            ),
            source_name="BELLE",
            source_type="qa",
            decision="keep",
            absent_flags={"punctuation_dense"},
        )

    def test_keep_morse_code_table_qa_not_punctuation_noise(self) -> None:
        self.assert_classification(
            text=(
                "用户：将一段文字转换成摩尔斯电码。助手：下面是转换过程。"
                "A .- N -. 0 ----- B -... O --- 1 .---- C -.-. P .--. 2 ..--- "
                "D -.. Q --.- 3 ...-- E . R .-. 4 ....-。"
                "根据表格，Hello World 可转换为 .... . .-.. .-.. --- / .-- --- .-. .-.. -..。"
            ),
            source_name="BELLE",
            source_type="qa",
            decision="keep",
            absent_flags={"punctuation_dense"},
        )

    def test_quarantine_punctuation_dense_general_web_noise(self) -> None:
        self.assert_classification(
            text=(
                "栏目导航:::::: 新闻////下载////更多////推荐//// "
                "------::::::;;;;;;------::::::;;;;;;------::::::;;;;;; "
                "页面内容混杂大量符号，正文信息很少，更多相关、上一篇、下一篇反复出现。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"punctuation_dense"},
        )

    def test_keep_qa_software_discussion_not_download_directory(self) -> None:
        self.assert_classification(
            text=(
                "指令：为什么Google Play提示某应用是有害应用并建议卸载？"
                "回答：可能是安卓版加入了无法过审的功能。建议等待官方回应，"
                "不要从不可信渠道下载安装，可以使用其他过审的新闻类软件。"
            ),
            source_name="Zhihu-KOL",
            source_type="qa",
            decision="keep",
            absent_flags={"download_site_template"},
        )

    def test_keep_wiki_game_entry_with_version_and_environment(self) -> None:
        self.assert_classification(
            text=(
                "东方年代记是由制作组创作的同人RPG。基本信息包括版本号Ver 1.000，"
                "需求硬盘空间535MB，推荐运行环境WIN7。条目介绍游戏类型、制作人员、"
                "剧情流程和角色系统，并非下载目录页面。"
            ),
            source_name="acg_wiki",
            source_type="wiki",
            decision="keep",
            absent_flags={"download_site_template"},
        )

    def test_keep_insurance_article_with_deductible_odds_word(self) -> None:
        self.assert_classification(
            text=(
                "车辆投保的保险包括交强险和商业险。商业险又包括基本险和附加险，"
                "其中不计免赔率特约条款、车上人员责任险等险种需要分别理解。"
                "文章讨论的是交通事故理赔流程和保险责任边界。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"gambling_spam"},
        )

    def test_keep_radio_training_with_bit_symbol_word(self) -> None:
        self.assert_classification(
            text=(
                "QPSK调制器输入的数据是二进制数字序列，需要把每两个比特分成一组，"
                "共有四种组合，即00、01、10、11，其中每一组称为双比特码元。"
                "该资料介绍无线电通信和数字调制技术。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"gambling_spam"},
        )

    def test_keep_macau_tourism_forum_not_gambling_spam(self) -> None:
        self.assert_classification(
            text=(
                "澳门世界旅游经济论坛参加广东国际旅游产业博览会，向参会者介绍"
                "粤港澳大湾区、一带一路沿线国家和全球国际旅游平台的合作功能。"
                "论坛期间还设置互动游戏，帮助参会人士了解旅游经济议题。"
            ),
            source_name="p208p2002/wudao",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"gambling_spam"},
        )

    def test_keep_snow_driving_safety_article_not_betting_spam(self) -> None:
        self.assert_classification(
            text=(
                "雪天开车很危险，主要是路况复杂多变，车辆容易打滑。"
                "下面小编就雪天开车说一下注意事项：减速慢行、禁止急刹车、"
                "保持车内视线清晰，转弯时还要特别注意行人和车辆。"
            ),
            source_name="p208p2002/wudao",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"gambling_spam"},
        )

    def test_keep_bingo_culture_article_not_lottery_spam(self) -> None:
        self.assert_classification(
            text=(
                "宾戈游戏是一种常见的美国风俗，文章介绍游戏玩法和历史背景。"
                "有些地区也把宾戈作为低风险彩票活动，规则包括号码选择和多期投注，"
                "但这里没有平台入口、送彩金、预测计划或开户链接。"
            ),
            source_name="p208p2002/wudao",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"gambling_spam"},
        )

    def test_keep_acca_study_experience_with_more_work_more_gain_phrase(self) -> None:
        self.assert_classification(
            text=(
                "如何通过自学考过ACCA？有的paper是典型的多劳多得，"
                "只要写到点子上就能得分，因此要留意marking scheme怎么分配考点和分值。"
                "这是一篇备考经验，不是兼职接单广告。"
            ),
            source_name="webtext2019zh",
            source_type="qa_short_answer",
            decision="keep",
            absent_flags={"commercial_promo_template"},
        )

    def test_keep_qa_answer_with_markdown_horizontal_rules(self) -> None:
        self.assert_classification(
            text=(
                "目前基因测序体检的风险是什么？为什么会被叫停？"
                "这个问题需要区分科研测序、临床诊断和消费级检测。"
                "----------------------------------------------------------"
                "如果缺少准入标准和质量控制，检测结果可能被过度解读，也可能影响后续治疗决策。"
                "----------------------------------------------------------"
                "因此政策重点不是否定技术，而是要求应用场景、报告解释和伦理审查更加规范。"
            ),
            source_name="webtext2019zh",
            source_type="qa_short_answer",
            decision="keep",
            absent_flags={"punctuation_dense"},
        )

    def test_keep_qa_answer_with_hash_separator(self) -> None:
        self.assert_classification(
            text=(
                "指令：如何评价Apple Watch推出的全新彩虹表带？"
                "回答：是对LGBT人群的重视吧，很好看，支持爱平等的可以买。"
                "##################分割##################"
                "但是对于黑色表盘的用户不太友好，表带边上是白色的。"
                "我买过其他颜色的，如图，是真的很丑！！！！！！！！！！"
            ),
            source_name="Zhihu-KOL",
            source_type="qa",
            decision="keep",
            absent_flags={"punctuation_dense"},
        )

    def test_quarantine_hospital_advertorial(self) -> None:
        self.assert_classification(
            text=(
                "上饶市第五人民医院是上饶地区规模最大的一家现代化专业医疗机构，"
                "拥有一流的检验中心、一流的诊疗设备、一流的专家团队和服务理念，"
                "专业治疗疾病，并荣获全国最受欢迎疾病医院、全国最值得信赖疾病医院。"
            ),
            source_name="p208p2002/wudao",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"medical_promo_template"},
        )

    def test_quarantine_medical_product_advertorial(self) -> None:
        self.assert_classification(
            text=(
                "静脉曲张中医疗法日益受到关注。周围血管疾病专家研制成功脉活修复液，"
                "解决静脉曲张中医疗法的难题。专家介绍，静脉曲张症状主要发生于下肢，"
                "想了解更多请继续阅读本文。"
            ),
            source_name="p208p2002/wudao",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"medical_promo_template"},
        )

    def test_quarantine_long_medical_product_advertorial(self) -> None:
        self.assert_classification(
            text=(
                "静脉曲张久坐一族健康杀手，严重威胁身体健康，而关于静脉曲张中医疗法"
                "的问题也日益受到关注。解决这个问题不仅是患者的梦想，周围血管疾病专家"
                "也深感责任重大，经过多年钻研，脉活修复液的研制成功解决了静脉曲张中医疗法"
                "的难题。专家介绍，静脉曲张症状主要发生于下肢，有些毛细血管扩张严格说"
                "并不是真正的静脉曲张，可以不用治疗，但想了解更多请继续阅读本文。"
            ),
            source_name="p208p2002/wudao",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"medical_promo_template"},
        )

    def test_quarantine_dental_service_lead_ad(self) -> None:
        self.assert_classification(
            text=(
                "宁波隐形矫正哪里好，开颌矫正前后对比。宁波齿科医院推荐，"
                "时代天使隐形牙套价格，成人正畸大概多少费用？想知道具体医生口碑可以问我，"
                "也可以上爱优牙平台找好牙医，联系账号可推荐全国优秀正畸医生。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="quarantine",
            flags={"medical_promo_template"},
        )

    def test_keep_dental_health_article_not_service_lead(self) -> None:
        self.assert_classification(
            text=(
                "隐形矫正需要医生评估牙齿排列、牙周健康和咬合关系。"
                "文章介绍正畸治疗的适应症、复诊周期和清洁注意事项，没有推荐医院、"
                "平台、价格或联系方式。"
            ),
            source_name="BAAI/CCI3-Data",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"medical_promo_template"},
        )

    def test_keep_child_fluoride_price_experience_not_service_lead(self) -> None:
        self.assert_classification(
            text=(
                "带孩子上牙医诊所涂氟是一件大事。五岁前涂氟有健保给付，"
                "每半年可以涂一次，自费涂氟的价格差异不小。文章记录医生检查牙齿、"
                "清除齿垢和提醒半小时内不要喝水的过程，没有推荐诊所、平台或联系方式。"
            ),
            source_name="opencsg/Fineweb-Edu-Chinese-V2.1",
            source_type="streamed_general",
            decision="keep",
            absent_flags={"medical_promo_template"},
        )

    def test_keep_normal_health_article(self) -> None:
        self.assert_classification(
            text=(
                "黄瓜、茄子、绿豆等食物含有不同营养成分。文章讨论饮食结构、"
                "运动习惯和慢性病预防，没有推荐医院、产品或治疗服务。"
            ),
            source_name="p208p2002/wudao",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"medical_promo_template"},
        )

    def test_keep_normal_hospital_public_health_news(self) -> None:
        self.assert_classification(
            text=(
                "某市人民医院发布科研进展，医生介绍了慢病管理和随访制度，"
                "重点是公共卫生服务流程，没有宣传诊疗设备或招揽患者。"
            ),
            source_name="p208p2002/wudao",
            source_type="general_web_backbone",
            decision="keep",
            absent_flags={"medical_promo_template"},
        )

    def test_chunked_append_preloads_hashes_and_skips_overlap(self) -> None:
        rows = [
            {
                "text": f"这是第{i}段正常训练文本，用来验证分片追加清洗时不会重复写入重叠数据。",
                "source_name": "BAAI/CCI3-Data",
                "source_type": "general_web_backbone",
            }
            for i in range(1, 6)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "BAAI_CCI3-Data_test.jsonl"
            output_dir = tmp_path / "clean"
            output_path = output_dir / input_path.name
            input_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            hashes: set[str] = set()
            first = clean_file(
                input_path,
                output_path=output_path,
                quarantine_path=None,
                max_docs=None,
                start_line=1,
                max_input_lines=2,
                min_chars=20,
                clean_version="test",
                global_hashes=hashes,
                example_limit=2,
                status_interval=0,
                append_output=True,
            )
            self.assertEqual(first["seen"], 2)
            self.assertEqual(first["kept"], 2)

            hashes = preload_hashes(output_dir, skip_names=set())
            second = clean_file(
                input_path,
                output_path=output_path,
                quarantine_path=None,
                max_docs=None,
                start_line=2,
                max_input_lines=3,
                min_chars=20,
                clean_version="test",
                global_hashes=hashes,
                example_limit=2,
                status_interval=0,
                append_output=True,
            )
            self.assertEqual(second["seen"], 3)
            self.assertEqual(second["kept"], 2)
            self.assertEqual(second["duplicates"], 1)
            written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(written), 4)
            self.assertEqual([row["text"] for row in written], [row["text"] for row in rows[:4]])


if __name__ == "__main__":
    unittest.main()

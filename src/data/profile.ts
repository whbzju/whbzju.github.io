export const profile = {
  name: "Wujia / Haibo Wu",
  mark: "WH",
  eyebrow: "AI product builder · Open-source operator · Writer",
  headline: "I build AI products for real workflows.",
  summary:
    "I have spent the past decade building products at the intersection of AI, internet commerce, creative workflows, SaaS delivery, and commercialization. Today I work on WeShop AI for global e-commerce creative production and Memex for local-first personal AI memory.",
  chinese:
    "过去十年，我主要在互联网产品、AI 应用和商业化一线工作。现在的重点是 WeShop AI 和 Memex：前者服务全球电商创意生产，后者探索 local-first 的个人 AI 记忆。",
  links: {
    github: "https://github.com/whbzju",
    linkedin: "https://www.linkedin.com/in/haibo-wu-97b60759/",
    zhihu: "https://www.zhihu.com/people/wu-hai-bo",
    memex: "https://github.com/memex-lab/memex",
    weshop: "https://www.weshop.ai/",
  },
};

export const nav = [
  { label: "Work", labelZh: "作品", href: "/works/" },
  { label: "Blog", labelZh: "文章", href: "/blog/" },
  { label: "Events", labelZh: "活动", href: "/events/" },
  { label: "About", labelZh: "经历", href: "/about/" },
];

export const works = [
  {
    title: "Memex",
    type: "Open-source AI product",
    typeZh: "开源 AI 产品",
    href: "https://github.com/memex-lab/memex",
    logo: "/images/logos/memex-logo.png",
    logoAlt: "Memex logo",
    image: "/images/products/memex-homepage.jpg",
    imageAlt: "Memex homepage screenshot",
    summary:
      "An open-source, local-first AI journal that captures text, screenshots, voice, and images, then organizes them into timeline cards, insights, and personal memory.",
    summaryZh:
      "一个开源、local-first 的 AI 日志产品，用来记录文字、截图、语音和图片，再由 AI 整理成时间线、洞察和个人记忆。",
    thesis:
      "Personal AI needs trust, ownership, and durable context before it can become useful over time.",
    thesisZh:
      "个人 AI 想要长期有用，首先要解决信任、所有权和上下文沉淀。",
    proof: ["Open source", "Local-first", "AI journal", "Personal agents"],
    proofZh: ["开源", "Local-first", "AI 日志", "个人智能体"],
  },
  {
    title: "WeShop AI",
    type: "Commercial AI product",
    typeZh: "商业化 AI 产品",
    href: "https://www.weshop.ai/",
    logo: "/images/logos/weshop-logo.svg",
    logoAlt: "WeShop AI logo",
    image: "/images/products/weshop-homepage.jpg",
    imageAlt: "WeShop AI homepage screenshot",
    summary:
      "A commercial AI image and video platform for global e-commerce teams creating product, model, and advertising assets.",
    summaryZh:
      "面向全球电商团队的 AI 图片与视频生产平台，服务商品图、模特图和广告创意。",
    thesis:
      "Vertical AI products win through workflow depth, quality control, distribution, and domain data.",
    thesisZh:
      "垂直 AI 产品的壁垒，来自工作流深度、质量控制、分发能力和行业数据。",
    proof: ["Vertical AI SaaS", "Creative workflow", "Global users", "Media covered"],
    proofZh: ["垂直 AI SaaS", "创意工作流", "全球用户", "媒体报道"],
  },
];

export const principles = [
  {
    title: "AI applications must enter workflows",
    titleZh: "AI 应用必须进入工作流",
    text:
      "The valuable layer is not a prettier model interface. It is the repeated task, data structure, collaboration loop, and delivery expectation around the model.",
    textZh:
      "真正有价值的层不是更好看的模型界面，而是围绕模型形成的重复任务、数据结构、协作流程和交付预期。",
  },
  {
    title: "Vertical AI SaaS compounds through context",
    titleZh: "垂直 AI SaaS 通过上下文复利",
    text:
      "As model capability becomes widely available, durable product advantage shifts toward domain data, workflow closure, distribution, and operational taste.",
    textZh:
      "当模型能力越来越普及时，长期产品优势会转向行业数据、工作流闭环、分发能力和运营品味。",
  },
  {
    title: "Personal agents require local-first trust",
    titleZh: "个人智能体需要 local-first 信任",
    text:
      "When an agent touches screenshots, voice, private notes, and long-term memory, ownership and controllability become product foundations.",
    textZh:
      "当智能体开始处理截图、语音、私人笔记和长期记忆时，所有权和可控性会变成产品底座。",
  },
  {
    title: "Open-source growth starts with a real problem",
    titleZh: "开源增长从真实问题开始",
    text:
      "A small project can still be discovered when the problem is concrete, the progress is visible, and the community can participate in the work.",
    textZh:
      "一个小项目也可以被看见，前提是问题足够具体、进展足够可见，社区也能参与到真实工作里。",
  },
];

export const experience = [
  {
    period: "2026-now",
    periodZh: "2026 至今",
    title: "Memex and local-first personal AI",
    titleZh: "Memex 与 local-first 个人 AI",
    text:
      "Building Memex, an open-source local-first AI journal for private context, screenshots, voice, photos, long-term memory, and personal agents.",
    textZh:
      "在做 Memex，一个开源、local-first 的 AI 日志产品，核心关注私人上下文、截图、语音、图片、长期记忆和个人智能体。",
    href: "https://github.com/memex-lab/memex",
    source: "GitHub",
    sourceZh: "GitHub",
  },
  {
    period: "2023-now",
    periodZh: "2023 至今",
    title: "WeShop AI and commercial creative workflows",
    titleZh: "WeShop AI 与商业创意工作流",
    text:
      "Leading WeShop as GM, turning AI commercial photography into a product for e-commerce merchants, brands, creators, and key accounts across global markets.",
    textZh:
      "担任 WeShop 总经理，主导 AI 商拍产品发布和增长，服务电商商家、品牌、创作者和大客户，重点放在工作流、质量、全球化 SaaS 和交付。",
    href: "https://m.36kr.com/p/3262118027476743",
    source: "36Kr talk",
    sourceZh: "36氪演讲",
  },
  {
    period: "2021-2023",
    periodZh: "2021-2023",
    title: "Large models, overseas commerce, and the path to WeShop",
    titleZh: "大模型、海外电商和 WeShop 的前期探索",
    text:
      "Started working on large-model applications in 2021 and explored how AI could serve overseas commerce before WeShop became a public product.",
    textZh:
      "2021 年开始投入大模型应用方向，围绕海外电商和 AI 商拍做前期探索，最终推进 WeShop 产品化。",
    href: "https://m.36kr.com/p/3262118027476743",
    source: "36Kr talk",
    sourceZh: "36氪演讲",
    extraHref: "https://hznews.hangzhou.com.cn/kejiao/content/2024-02/02/content_8683997_4.htm",
    extraSource: "Hangzhou News feature",
    extraSourceZh: "杭州新闻报道",
  },
  {
    period: "2014-2021",
    periodZh: "2014-2021",
    title: "Mogujie: search, recommendation, ads, and product",
    titleZh: "蘑菇街：搜索、推荐、广告和产品",
    text:
      "Joined Mogujie in 2014. Led search, recommendation, and advertising algorithm systems, and later took responsibility for core product work.",
    textZh:
      "2014 年加入蘑菇街，先后负责搜索、推荐、广告算法体系，也承担过蘑菇街核心产品工作。",
    href: "https://cloud.tencent.com/tvp/member/927?userType=0",
    source: "Tencent Cloud TVP profile",
    sourceZh: "腾讯云 TVP 个人简介",
  },
  {
    period: "Before 2014",
    periodZh: "2014 以前",
    title: "Huawei 2012 Lab and terminal AI applications",
    titleZh: "华为 2012 实验室与终端 AI 应用",
    text:
      "Worked in Huawei's 2012 Lab on AI applications for terminal devices before the current large-model wave.",
    textZh:
      "早年在华为 2012 实验室做终端侧 AI 应用，那是大模型浪潮之前的一段应用 AI 经历。",
    href: "https://www.weshop.com/blog/post-2633",
    source: "Jixin interview",
    sourceZh: "极新访谈",
  },
  {
    period: "Education",
    periodZh: "教育背景",
    title: "Computer science at Zhejiang University",
    titleZh: "浙江大学计算机专业",
    text:
      "Studied computer science at Zhejiang University.",
    textZh:
      "浙江大学计算机专业。",
    href: "https://cloud.tencent.com/tvp/member/927?userType=0",
    source: "Tencent Cloud TVP profile",
    sourceZh: "腾讯云 TVP 个人简介",
  },
];

export const writing = [
  {
    title: "一个没资源的开源项目，是怎么被人看见的",
    source: "Zhihu column",
    sourceZh: "知乎专栏",
    href: "https://zhuanlan.zhihu.com/p/2046314257738887994",
    noteEn:
      "A postmortem on Memex's open-source cold start: how a project with no budget or partnerships got discovered through real work and useful discussion.",
    note: "Memex 开源冷启动复盘：没有预算、没有合作，如何靠真实作品和有效讨论被看见。",
  },
  {
    title: "我们做了一个关于“记录自己”的东西，然后决定把它开源",
    source: "Zhihu column",
    sourceZh: "知乎专栏",
    href: "https://zhuanlan.zhihu.com/p/2016169349749166998",
    noteEn:
      "The product thinking behind Memex: why personal records, privacy, and long-term trust matter again in the AI era.",
    note: "关于 Memex 的产品理念：AI 时代，个人记录、隐私和长期信任为什么重新变重要。",
  },
  {
    title: "我们发布了WeShop商拍1.5版----分享一些最近的思考",
    source: "Zhihu column",
    sourceZh: "知乎专栏",
    href: "https://zhuanlan.zhihu.com/p/680507675",
    noteEn:
      "Reflections on WeShop's product evolution, AI commercial photography, and e-commerce creative workflows.",
    note: "关于 WeShop 产品演进、AI 商拍场景和电商创意工作流的阶段性思考。",
  },
];

export const events = [
  {
    title: "AI应用 WeShop 卷到海外，智能时代的新外贸故事",
    titleEn: "WeShop's overseas expansion and the new foreign-trade story in the AI era",
    titleZh: "AI应用 WeShop 卷到海外，智能时代的新外贸故事",
    date: "2025-11",
    kind: "Feature",
    kindZh: "专题报道",
    source: "WeShop",
    sourceZh: "WeShop",
    href: "https://www.weshop.com/blog/post-3126",
    noteEn:
      "A feature on WeShop's overseas market expansion, local partnerships, and global AI SaaS distribution.",
    note: "关于 WeShop 出海、本地合作和全球 AI SaaS 分发的专题报道。",
  },
  {
    title: "WeShop唯象总经理吴海波：AI创业已非“套壳应用”时代",
    titleEn: "AI entrepreneurship is no longer about wrapper apps",
    titleZh: "WeShop唯象总经理吴海波：AI创业已非“套壳应用”时代",
    date: "2025-04",
    kind: "Talk",
    kindZh: "演讲",
    source: "36Kr talk",
    sourceZh: "36氪演讲",
    href: "https://m.36kr.com/p/3262118027476743",
    noteEn:
      "A public talk on AI application startups, vertical use cases, and commercial product depth.",
    note: "关于 AI 应用创业、垂直场景和商业化产品深度的公开分享。",
  },
  {
    title: "AI Entrepreneurship Era Is No Longer About Wrapper Apps",
    titleEn: "AI Entrepreneurship Era Is No Longer About Wrapper Apps",
    titleZh: "36Kr Global 英文报道：AI 创业已不是套壳应用时代",
    date: "2025-04",
    kind: "English coverage",
    kindZh: "英文报道",
    source: "36Kr Global",
    sourceZh: "36Kr Global",
    href: "https://eu.36kr.com/en/p/3262118027476743",
    noteEn:
      "English coverage of the same product thinking for an international audience.",
    note: "面向国际读者的英文报道，延展 36氪演讲中的 AI 应用创业判断。",
  },
  {
    title: "从“模型即应用”到场景深耕：WeShop 唯象吴海波解码 AI 商拍的破局密码",
    titleEn: "From model-as-app to scenario depth in AI commercial photography",
    titleZh: "从“模型即应用”到场景深耕：WeShop 唯象吴海波解码 AI 商拍的破局密码",
    date: "2025-04",
    kind: "Event coverage",
    kindZh: "活动报道",
    source: "TOM / 36Kr AI Partner",
    sourceZh: "TOM / 36氪 AI Partner",
    href: "https://xiaofei.tom.com/202504/4016633183.html",
    noteEn:
      "Coverage of the AI Partner conference talk on WeShop's global path and the shift from model access to scenario depth.",
    note: "报道 36氪 AI Partner 大会分享，讨论 WeShop 唯象的全球化路径，以及从模型能力走向场景深耕。",
  },
  {
    title: "组织的野望：AI 公司全员远程办公可行吗？",
    titleEn: "Can an AI company run fully remote?",
    titleZh: "组织的野望：AI 公司全员远程办公可行吗？",
    date: "2025",
    kind: "Podcast",
    kindZh: "播客",
    source: "苔藓之火",
    sourceZh: "苔藓之火",
    href: "https://podcasts.apple.com/tr/podcast/6-%E7%BB%84%E7%BB%87%E7%9A%84%E9%87%8E%E6%9C%9B-ai%E5%85%AC%E5%8F%B8%E5%85%A8%E5%91%98%E8%BF%9C%E7%A8%8B%E5%8A%9E%E5%85%AC%E5%8F%AF%E8%A1%8C%E5%90%97-ai%E5%BA%94%E7%94%A8%E9%BB%91%E9%A9%ACweshop-%E5%90%B4%E6%B5%B7%E6%B3%A2/id1839281356?i=1000726376776",
    noteEn:
      "A podcast conversation on remote work, AI-native organizations, startup autonomy, and product cadence.",
    note: "关于全员远程、AI 原生组织、创业团队自主性和产品节奏的播客讨论。",
  },
  {
    title: "AI 商拍用上智能算力，天翼云助力 WeShop 唯象“点击就成片”",
    titleEn: "AI commercial photography, cloud compute, and WeShop's click-to-create workflow",
    titleZh: "AI 商拍用上智能算力，天翼云助力 WeShop 唯象“点击就成片”",
    date: "2024-11",
    kind: "Case study",
    kindZh: "案例报道",
    source: "China.com repost",
    sourceZh: "中华网转载",
    href: "https://m.tech.china.com/hea/article/20241211/122024_1615378.html",
    noteEn:
      "A media case study, originally from Xinhua, on WeShop's AI commercial photography workflow, regional adaptation, and infrastructure support.",
    note: "新华网原文的转载版本，关于 WeShop AI 商拍工作流、不同地区数据训练和算力基础设施支持。",
  },
  {
    title: "2024 腾讯数字生态大会：WeShop 总经理吴海波展望 AI 商拍未来",
    titleEn: "The future of AI commercial photography at Tencent Global Digital Ecosystem Summit",
    titleZh: "2024 腾讯数字生态大会：WeShop 总经理吴海波展望 AI 商拍未来",
    date: "2024-09",
    kind: "Conference",
    kindZh: "大会分享",
    source: "Tencent Global Digital Ecosystem Summit",
    sourceZh: "腾讯全球数字生态大会",
    href: "https://www.laohu8.com/post/349922498707752",
    noteEn:
      "A conference appearance on AI commercial photography, product-market fit, multimodal models, and vertical application innovation.",
    note: "在互联网 AI 应用专场分享 AI 商拍、PMF、多模态模型和垂直应用创新的实践。",
  },
  {
    title: "WeShop 唯象 GM 吴海波：创新才有红利，产品驱动增长",
    titleEn: "Innovation creates the dividend; products drive growth",
    titleZh: "WeShop 唯象 GM 吴海波：创新才有红利，产品驱动增长",
    date: "2024-07",
    kind: "Interview",
    kindZh: "访谈",
    source: "创氪网",
    sourceZh: "创氪网",
    href: "http://www.chuanganwang.cn/dskx/yw/20240705/173858.html",
    noteEn:
      "An interview on WeShop's global users, product growth, cross-border commerce, and AI product innovation.",
    note: "围绕 WeShop 全球用户、产品增长、跨境电商和 AI 产品创新的访谈。",
  },
  {
    title: "Sora 爆火将如何改变广告行业？",
    titleEn: "How will Sora change the advertising industry?",
    titleZh: "Sora 爆火将如何改变广告行业？",
    date: "2024-02",
    kind: "Media interview",
    kindZh: "媒体采访",
    source: "36Kr",
    sourceZh: "36氪",
    href: "https://m.36kr.com/p/2674977768683266",
    noteEn:
      "A discussion on video generation models, advertising, and commercial creative workflows.",
    note: "围绕视频生成模型、广告行业和商业创意工作流的讨论。",
  },
  {
    title: "AI 落地进行时，让创新真正触手可及",
    titleEn: "AI in implementation: making innovation genuinely accessible",
    titleZh: "AI 落地进行时，让创新真正触手可及",
    date: "2024-01",
    kind: "Panel",
    kindZh: "圆桌",
    source: "Zhihu AI Pioneer Salon",
    sourceZh: "知乎 AI 先行者沙龙",
    href: "https://zhuanlan.zhihu.com/p/676903031",
    noteEn:
      "A salon panel on how AI applications cross the last mile from model capability to real industry implementation.",
    note: "知乎 AI 先行者沙龙圆桌，讨论 AI 应用如何完成从模型能力到产业落地的最后一公里。",
  },
  {
    title: "我们是独立于蘑菇街的产品，改老的东西比做新东西更难",
    titleEn: "WeShop as an independent product, and why changing old systems is harder than building new ones",
    titleZh: "我们是独立于蘑菇街的产品，改老的东西比做新东西更难",
    date: "2023-11",
    kind: "Interview",
    kindZh: "访谈",
    source: "Tencent News",
    sourceZh: "腾讯新闻",
    href: "https://news.qq.com/rain/a/20231113A04U5B00",
    noteEn:
      "An interview about WeShop, product independence, organization, and business evolution.",
    note: "关于 WeShop、产品独立性、组织和业务演进的访谈。",
  },
  {
    title: "极新对话 WeShop 总经理吴海波｜投身 AI，大胆创新",
    titleEn: "A conversation with Jixin on AI entrepreneurship and product judgment",
    titleZh: "极新对话 WeShop 总经理吴海波｜投身 AI，大胆创新",
    date: "2023-10",
    kind: "Interview",
    kindZh: "访谈",
    source: "Jixin / WeShop",
    sourceZh: "极新 / WeShop",
    href: "https://www.weshop.com/blog/post-2633",
    noteEn:
      "A long-form interview on building AI products, developing product taste for AI, and commercializing AI commercial photography.",
    note: "一次关于 AI 产品构建、AI 感、产品判断和 AI 商拍商业化的长访谈。",
  },
  {
    title: "替代 OR 进化——人工智能时代的服装设计",
    titleEn: "Replacement or evolution: fashion design in the AI era",
    titleZh: "替代 OR 进化——人工智能时代的服装设计",
    date: "2023-09",
    kind: "Salon talk",
    kindZh: "沙龙演讲",
    source: "Beijing Fashion Week / 36Kr",
    sourceZh: "北京时装周 / 36氪",
    href: "https://www.weshop.com/blog/post-2559",
    noteEn:
      "A digital fashion salon talk on AI, fashion, e-commerce creative workflows, and the shift from replacement anxiety to production evolution.",
    note: "在“虚实共生，数位重建”数字时尚沙龙上，分享 AI、时尚、电商创意工作流和生产方式演进。",
  },
  {
    title: "WeShop 受邀参加东方卫视 AI 主题对话节目",
    titleEn: "AI-themed dialogue on Dragon TV's finance program",
    titleZh: "WeShop 受邀参加东方卫视 AI 主题对话节目",
    date: "2023-08",
    kind: "TV program",
    kindZh: "电视节目",
    source: "Dragon TV",
    sourceZh: "东方卫视",
    href: "https://cloud.tencent.com/developer/news/1149786",
    noteEn:
      "A public TV discussion on how AI technology changes e-commerce, with live demonstrations of model and scene replacement.",
    note: "在《来点财经范儿》AI 主题节目中，讨论 AI 技术对电商行业的影响，并现场展示换模特和换场景能力。",
  },
  {
    title: "WeShop 亮相 2023 世界人工智能大会直播",
    titleEn: "WeShop at the 2023 World Artificial Intelligence Conference livestream",
    titleZh: "WeShop 亮相 2023 世界人工智能大会直播",
    date: "2023-07",
    kind: "Livestream",
    kindZh: "大会直播",
    source: "WAIC 2023",
    sourceZh: "2023 世界人工智能大会",
    href: "https://wap.bjd.com.cn/news/2023/07/07/10488899.shtml",
    noteEn:
      "A livestream appearance discussing AIGC product scenarios and productivity changes in commercial photography.",
    note: "在 2023 世界人工智能大会直播间，讨论 AIGC 产品应用场景和商拍生产力变化。",
  },
  {
    title: "谈谈做 WeShop 过程中对 AIGC 产品的一些思考",
    titleEn: "Product reflections from building WeShop's AIGC commercial photography product",
    titleZh: "谈谈做 WeShop 过程中对 AIGC 产品的一些思考",
    date: "2023-06",
    kind: "Launch note",
    kindZh: "产品发布",
    source: "Zhihu column",
    sourceZh: "知乎专栏",
    href: "https://zhuanlan.zhihu.com/p/633513013",
    noteEn:
      "A launch-period note after the WeShop beta opened registration, covering growth, user feedback, and the product's move toward official release.",
    note: "WeShop beta 开放注册后的产品复盘，记录增长、用户反馈和正式版上线前后的产品判断。",
  },
  {
    title: "和大家汇报下我们电商 AI 模特产品 WeShop beta 版本开放测试",
    titleEn: "WeShop AI model beta opened for public testing",
    titleZh: "和大家汇报下我们电商 AI 模特产品 WeShop beta 版本开放测试",
    date: "2023-04",
    kind: "Product beta",
    kindZh: "产品内测",
    source: "Zhihu column",
    sourceZh: "知乎专栏",
    href: "https://zhuanlan.zhihu.com/p/629144911",
    noteEn:
      "The early public beta note for WeShop, showing e-commerce model replacement and scene generation use cases.",
    note: "WeShop 早期公开内测说明，展示电商换模特、换场景等核心使用场景。",
  },
  {
    title: "电商数字模特生成技术实践分享",
    titleEn: "Technical practice notes on e-commerce digital model generation",
    titleZh: "电商数字模特生成技术实践分享",
    date: "2023-04",
    kind: "Technical note",
    kindZh: "技术分享",
    source: "Zhihu column",
    sourceZh: "知乎专栏",
    href: "https://zhuanlan.zhihu.com/p/621970429",
    noteEn:
      "An early technical write-up on the digital-model generation work that led into WeShop.",
    note: "WeShop 早期电商数字模特生成方向的技术实践记录。",
  },
  {
    title: "WeShop 团队从大模型研发走向 AI 商拍产品化",
    titleEn: "From large-model R&D to productizing AI commercial photography",
    titleZh: "WeShop 团队从大模型研发走向 AI 商拍产品化",
    date: "2021-2023",
    kind: "Milestone",
    kindZh: "阶段节点",
    source: "Hangzhou News / 36Kr",
    sourceZh: "杭州新闻 / 36氪",
    href: "https://hznews.hangzhou.com.cn/kejiao/content/2024-02/02/content_8683997_4.htm",
    noteEn:
      "Background references describe the team's early large-model R&D and the path toward the 2023 WeShop product launch.",
    note: "背景报道提到团队从早期大模型研发逐步走向 2023 年 WeShop 产品化发布。",
  },
];

export const writingHighlights = [
  {
    group: "AI products and agents",
    groupZh: "AI 产品与 Agent",
    type: "Zhihu answer",
    typeZh: "知乎回答",
    title: "在研究编程 Agent，Agent 核心就几十行代码，那剩下的几万行到底在解决什么问题？",
    titleEn: "If an agent core is only a few dozen lines, what are the other tens of thousands of lines for?",
    href: "https://www.zhihu.com/question/1994057559171167995/answer/2021164259149714484",
    noteEn:
      "A product-engineering answer on production agents: tools, permissions, validation, application structure, and the gap between a demo loop and a usable product.",
    note: "从产品工程角度解释生产级 Agent 为什么需要工具、权限、校验和完整应用结构。",
  },
  {
    group: "AI products and agents",
    groupZh: "AI 产品与 Agent",
    type: "Zhihu answer",
    typeZh: "知乎回答",
    title: "怎么成为一个 AI agent 工程师？",
    titleEn: "How to become an AI agent engineer",
    href: "https://www.zhihu.com/question/1936375725931361485/answer/1951304824327996800",
    noteEn:
      "A practical answer connecting agent engineering with Memex, local-first personal AI, and the product work behind reliable agent experiences.",
    note: "从 AI Agent 工程聊到 Memex、local-first 个人 AI 和产品构建。",
  },
  {
    group: "AI products and agents",
    groupZh: "AI 产品与 Agent",
    type: "Zhihu answer",
    typeZh: "知乎回答",
    title: "为什么一定要频繁记录自己？",
    titleEn: "Why record yourself so often?",
    href: "https://www.zhihu.com/question/1925936172162605365/answer/2016262442049634843",
    noteEn:
      "A Memex-adjacent answer about personal records, AI anxiety, memory, and agency in an era when private context becomes useful again.",
    note: "延伸 Memex 的产品理念，讨论个人记录、AI 焦虑、记忆和主体性。",
  },
  {
    group: "WeShop and commercial AI",
    groupZh: "WeShop 与商业 AI",
    type: "Zhihu answer",
    typeZh: "知乎回答",
    title: "AI 技术代替电商模特，现在可以实现了吗？",
    titleEn: "Can AI replace e-commerce models now?",
    href: "https://www.zhihu.com/question/590884963/answer/2983851310",
    noteEn:
      "A WeShop-grounded answer on virtual models, e-commerce photography, production constraints, and where generative AI was already useful before it was perfect.",
    note: "关于虚拟试衣、AI 模特和电商商拍生产约束的代表性回答。",
  },
  {
    group: "WeShop and commercial AI",
    groupZh: "WeShop 与商业 AI",
    type: "Zhihu answer",
    typeZh: "知乎回答",
    title: "现在 AI 很火，但实际落地的行业应用并不多，问题出在哪里？",
    titleEn: "Why are there still so few real industry AI applications?",
    href: "https://www.zhihu.com/question/635835356/answer/3388639454",
    noteEn:
      "A product answer on why AI applications struggle to move from impressive demos into repeatable industry workflows, using WeShop as context.",
    note: "用 WeShop 经验回答 AI 应用为什么难从 demo 进入行业工作流。",
  },
  {
    group: "WeShop and commercial AI",
    groupZh: "WeShop 与商业 AI",
    type: "Zhihu column",
    typeZh: "知乎专栏",
    title: "WeShopAI 全员远程协作一年记",
    titleEn: "A year of remote collaboration at WeShop AI",
    href: "https://zhuanlan.zhihu.com/p/1910357861290210251",
    noteEn:
      "A founder/operator note on running a remote AI product team: cadence, autonomy, communication, and the kind of organization AI work requires.",
    note: "从团队 leader 视角记录 WeShop 全员远程协作一年的体感和组织判断。",
  },
  {
    group: "Technical foundations",
    groupZh: "技术理解",
    type: "Zhihu column",
    typeZh: "知乎专栏",
    title: "Diffusion Models 导读",
    titleEn: "A guide to diffusion models",
    href: "https://zhuanlan.zhihu.com/p/591720296",
    noteEn:
      "A technical primer for readers entering image generation, useful as a marker of the technical base behind later commercial AI image work.",
    note: "面向图像生成入门读者的 Diffusion Models 技术导读。",
  },
  {
    group: "Technical foundations",
    groupZh: "技术理解",
    type: "Zhihu column",
    typeZh: "知乎专栏",
    title: "应用视角下 ChatGPT 背后的关键技术讨论",
    titleEn: "The core technologies behind ChatGPT from an application perspective",
    href: "https://zhuanlan.zhihu.com/p/609624863",
    noteEn:
      "A technical-product discussion of ChatGPT from the perspective of application builders, not only model observers.",
    note: "从应用构建者视角讨论 ChatGPT 背后的关键技术。",
  },
  {
    group: "Technical foundations",
    groupZh: "技术理解",
    type: "Zhihu answer",
    typeZh: "知乎回答",
    title: "推荐系统中如何做 User Embedding？",
    titleEn: "How to build user embeddings in recommender systems",
    href: "https://www.zhihu.com/question/336110178/answer/848660398",
    noteEn:
      "A representative search/recommendation answer from earlier machine-learning work, connecting modeling concepts with production recommendation practice.",
    note: "把 User Embedding 和推荐系统实践联系起来的代表性回答。",
  },
  {
    group: "Technical foundations",
    groupZh: "技术理解",
    type: "Zhihu answer",
    typeZh: "知乎回答",
    title: "在你做推荐系统的过程中都遇到过什么坑？",
    titleEn: "Pitfalls from building recommender systems in real business",
    href: "https://www.zhihu.com/question/32218407/answer/555385513",
    noteEn:
      "A practical answer on what breaks in recommender systems once algorithms meet product constraints, traffic, incentives, and operations.",
    note: "来自真实业务经验的推荐系统踩坑总结。",
  },
];

export const writingCollections = [
  {
    title: "Zhihu columns and answers",
    titleZh: "知乎专栏与回答",
    description:
      "A long-running thinking trail across AI products, e-commerce creative workflows, agents, and open-source growth.",
    descriptionZh:
      "围绕 AI 产品、电商创意工作流、智能体和开源增长的一条长期思考线索。",
    items: [
      ["Zhihu column", "和大家汇报下我们电商AI模特产品WeShop beta版本开放测试", "https://zhuanlan.zhihu.com/p/629144911"],
      ["Zhihu column", "我们发布了WeShop商拍1.5版----分享一些最近的思考", "https://zhuanlan.zhihu.com/p/680507675"],
      ["Zhihu answer", "AI 技术代替电商模特，现在可以实现了吗？", "https://www.zhihu.com/question/590884963/answer/2983851310"],
      ["Zhihu answer", "怎么成为一个 AI agent 工程师？", "https://www.zhihu.com/question/1936375725931361485/answer/1951304824327996800"],
    ],
  },
];
